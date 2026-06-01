"""Math/asset runtime embedding for the HTML report (KaTeX, MathJax, Plotly),
local CSS asset inlining, and template path resolution.

Extracted from ``html_exporter.py`` (Phase 2 of the god-file split). All methods
are stateless ``@staticmethod`` helpers that read bundled runtime files and emit
``<script>``/``<style>`` blobs. ``HTMLExporter`` mixes this in, so existing
``HTMLExporter._plotly_bundle()`` style call sites keep working via the MRO.

The three compiled regexes used only by these helpers travel with them as class
attributes; internal cross-calls reference ``_AssetsMixin`` directly to avoid any
import back into ``html_exporter`` (no circular import).
"""
import base64
import re
import sys
from pathlib import Path

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger(name)


logger = get_logger(__name__)


class _AssetsMixin:
    """Stateless math/asset/template helpers mixed into ``HTMLExporter``."""

    _INLINE_LATEX_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
    _BEGIN_ENV_RE = re.compile(r"\\\\begin\{[^}]+\}")
    _CSS_URL_RE = re.compile(r"url\((?P<quote>['\"]?)(?P<path>[^\)\"']+)(?P=quote)\)", re.IGNORECASE)

    @staticmethod
    def _has_latex_syntax(value: str) -> bool:
        if not isinstance(value, str):
            return False
        if _AssetsMixin._BEGIN_ENV_RE.search(value):
            return True
        return bool(_AssetsMixin._INLINE_LATEX_RE.search(value))

    @staticmethod
    def _requires_math_rendering(results: dict, hero: dict | None = None) -> bool:
        strings_to_scan = []

        if isinstance(results, dict):
            for key in [
                "title", "subtitle", "dataset_name", "column_name", "dependent_variable",
                "unit", "units", "x_label", "xlabel", "y_label", "ylabel",
            ]:
                value = results.get(key)
                if isinstance(value, str):
                    strings_to_scan.append(value)

        if isinstance(hero, dict):
            for key in ["title", "subtitle", "test_name"]:
                value = hero.get(key)
                if isinstance(value, str):
                    strings_to_scan.append(value)

        return any(_AssetsMixin._has_latex_syntax(text) for text in strings_to_scan)

    @staticmethod
    def _math_bundle(preferred: str = "katex") -> tuple[str, str]:
        order = [preferred, "mathjax"] if preferred != "mathjax" else ["mathjax"]
        for engine in order:
            if engine == "katex":
                bundle, status = _AssetsMixin._katex_bundle()
            else:
                bundle, status = _AssetsMixin._mathjax_bundle()
            if status.startswith("loaded"):
                return bundle, status

        missing_bundle = (
            "<script>"
            "window.BioMedStatXMath={enabled:false,status:'missing-local-runtime'};"
            "window.BioMedStatXTypesetMath=function(){return Promise.resolve();};"
            "</script>"
        )
        return missing_bundle, "missing-local-runtime"

    @staticmethod
    def _katex_runtime_candidates() -> list[Path]:
        templates_dir = _AssetsMixin._templates_dir()
        source_root = Path(__file__).resolve().parent
        candidates = [
            templates_dir / "vendor" / "katex",
            source_root / "templates" / "vendor" / "katex",
        ]
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            candidates.append(Path(frozen_root) / "templates" / "vendor" / "katex")
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _katex_bundle() -> tuple[str, str]:
        for candidate_root in _AssetsMixin._katex_runtime_candidates():
            try:
                css_path = candidate_root / "katex.min.css"
                js_path = candidate_root / "katex.min.js"
                autorender_path = candidate_root / "auto-render.min.js"
                if not (css_path.exists() and js_path.exists() and autorender_path.exists()):
                    continue

                css_text = css_path.read_text(encoding="utf-8")
                css_text = _AssetsMixin._inline_local_css_assets(css_text, candidate_root)
                js_text = js_path.read_text(encoding="utf-8")
                autorender_text = autorender_path.read_text(encoding="utf-8")

                bootstrap = (
                    "<style>" + css_text + "</style>"
                    "<script>"
                    "window.BioMedStatXMath={enabled:false,status:'loaded',engine:'katex'};"
                    "window.BioMedStatXTypesetMath=function(root){"
                    "if(typeof renderMathInElement==='function'){"
                    "renderMathInElement(root||document.body,{"
                    "delimiters:["
                    "{left:'$$',right:'$$',display:true},"
                    "{left:'$',right:'$',display:false},"
                    "{left:'\\\\(',right:'\\\\)',display:false},"
                    "{left:'\\\\[',right:'\\\\]',display:true}"
                    "]"
                    "});"
                    "}"
                    "return Promise.resolve();"
                    "};"
                    "</script>"
                )
                runtime = f"<script>{js_text}</script><script>{autorender_text}</script>"
                finalize = (
                    "<script>"
                    "window.BioMedStatXMath={enabled:true,status:'loaded',engine:'katex'};"
                    "document.addEventListener('DOMContentLoaded',function(){"
                    "window.BioMedStatXTypesetMath(document.body);"
                    "});"
                    "</script>"
                )
                return bootstrap + runtime + finalize, "loaded-katex"
            except Exception as exc:
                logger.warning("failed to load KaTeX runtime %r: %s", candidate_root, exc, exc_info=True)
        return "", "missing-katex-runtime"

    @staticmethod
    def _inline_local_css_assets(css_text: str, assets_root: Path) -> str:
        if not isinstance(css_text, str) or not isinstance(assets_root, Path):
            return css_text

        def replace_url(match: re.Match) -> str:
            raw_path = (match.group("path") or "").strip()
            if not raw_path:
                return match.group(0)
            if raw_path.startswith(("data:", "http:", "https:", "//", "#")):
                return match.group(0)

            cleaned_path = raw_path.split("?", 1)[0].split("#", 1)[0]
            candidate = (assets_root / cleaned_path).resolve()
            if not candidate.exists() or not candidate.is_file():
                return match.group(0)

            mime_type = _AssetsMixin._guess_embedded_asset_mime(candidate.suffix.lower())
            if mime_type is None:
                return match.group(0)

            try:
                encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
            except Exception:
                return match.group(0)

            return f"url('data:{mime_type};base64,{encoded}')"

        return _AssetsMixin._CSS_URL_RE.sub(replace_url, css_text)

    @staticmethod
    def _guess_embedded_asset_mime(extension: str) -> str | None:
        mapping = {
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
            ".otf": "font/otf",
            ".eot": "application/vnd.ms-fontobject",
            ".svg": "image/svg+xml",
        }
        return mapping.get(extension)

    @staticmethod
    def _templates_dir() -> Path:
        module_templates = Path(__file__).resolve().parent.parent / "templates"
        if module_templates.exists():
            return module_templates
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            frozen_templates = Path(frozen_root) / "templates"
            if frozen_templates.exists():
                return frozen_templates
        return module_templates

    @staticmethod
    def _template_name(mode: str) -> str:
        return "report_multi.html.j2" if mode == "multi" else "report_single.html.j2"

    @staticmethod
    def _read_template(template_name: str) -> str:
        return (_AssetsMixin._templates_dir() / template_name).read_text(encoding="utf-8")

    @staticmethod
    def _plotly_bundle() -> str:
        try:
            from plotly.offline.offline import get_plotlyjs

            return f"<script>{get_plotlyjs()}</script>"
        except Exception as exc:
            logger.warning("plotly bundle unavailable: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def _mathjax_runtime_candidates() -> list[Path]:
        templates_dir = _AssetsMixin._templates_dir()
        source_root = Path(__file__).resolve().parent
        candidates = [
            templates_dir / "vendor" / "mathjax" / "tex-svg.js",
            templates_dir / "vendor" / "mathjax" / "tex-mml-chtml.js",
            source_root / "templates" / "vendor" / "mathjax" / "tex-svg.js",
            source_root / "templates" / "vendor" / "mathjax" / "tex-mml-chtml.js",
        ]
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            frozen_path = Path(frozen_root)
            candidates.extend([
                frozen_path / "templates" / "vendor" / "mathjax" / "tex-svg.js",
                frozen_path / "templates" / "vendor" / "mathjax" / "tex-mml-chtml.js",
            ])
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _mathjax_bundle() -> tuple[str, str]:
        for candidate in _AssetsMixin._mathjax_runtime_candidates():
            try:
                if not candidate.exists():
                    continue
                runtime = candidate.read_text(encoding="utf-8")
                bootstrap = (
                    "<script>"
                    "window.MathJax={"
                    "tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]},"
                    "svg:{fontCache:'none'},"
                    "options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}"
                    "};"
                    "window.BioMedStatXMath={enabled:false,status:'loaded'};"
                    "window.BioMedStatXTypesetMath=function(root){"
                    "if(window.MathJax&&window.MathJax.typesetPromise){"
                    "return window.MathJax.typesetPromise(root?[root]:undefined);"
                    "}"
                    "return Promise.resolve();"
                    "};"
                    "</script>"
                )
                runtime_script = f"<script>{runtime}</script>"
                finalize = (
                    "<script>"
                    "window.BioMedStatXMath={enabled:true,status:'loaded',engine:'mathjax'};"
                    "document.addEventListener('DOMContentLoaded',function(){"
                    "window.BioMedStatXTypesetMath(document.body);"
                    "});"
                    "</script>"
                )
                return bootstrap + runtime_script + finalize, "loaded-mathjax"
            except Exception as exc:
                logger.warning("failed to load MathJax runtime %r: %s", candidate, exc, exc_info=True)
        return "", "missing-mathjax-runtime"

# Offline KaTeX Runtime (Preferred)

KaTeX is the preferred runtime for offline report math rendering because it is generally smaller and faster than MathJax for standard formula use.

Required files in this folder:
- katex.min.css
- katex.min.js
- auto-render.min.js

Runtime behavior:
- The HTML exporter injects KaTeX only when LaTeX syntax is detected in report-relevant strings (for example labels, units, titles).
- No CDN fallback is used.
- If KaTeX files are missing, exporter falls back to local MathJax runtime in src/templates/vendor/mathjax/.
- If no local runtime exists, reports expose missing-local-runtime and continue without typesetting.

Typical source:
- KaTeX distribution assets

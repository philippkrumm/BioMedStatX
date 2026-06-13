# Offline MathJax Runtime (Fallback)

MathJax is used as a fallback runtime when LaTeX syntax is detected and no local KaTeX runtime is available.

Supported filenames:
- `tex-svg.js` (recommended)
- `tex-mml-chtml.js`

Runtime behavior:
- The HTML exporter only inlines local files from this folder.
- No CDN fallback is used.
- If no local KaTeX and no local MathJax runtime are present, reports expose a `missing-local-runtime` status and continue without typesetting.

Typical source:
- MathJax v3 distribution (`es5/tex-svg.js`)

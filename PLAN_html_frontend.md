# HTML Export — Frontend Design Improvements

> Plan erstellt: 2026-04-01  
> Branch: `feature/advanced-stats-automation`

## Context

Der HTML-Export erzeugt vollständig offline-fähige, self-contained Reports (Jinja2-Template mit eingebettetem CSS/JS/Plotly). Die Grundstruktur ist solide, aber das Design kann an mehreren Stellen spürbar verbessert werden:

- **Keine Navigation** bei langen Reports → Nutzer scrollen blind durch alle Sections
- **Dark Mode fehlt** → schlechte Lesbarkeit bei Nachtschichten
- **Kein Print-Stylesheet** → Drucken für Paper/Supervisor-Meeting sieht schlecht aus
- **P-Werte in Tabellen sind farbneutral** → schnelles Scannen für Signifikanz ist mühsam
- **Assumption-Status ist nur Text** → kein visueller Hinweis auf Pass/Fail-Rate
- **Stat-Werte nutzen keine monospace Schrift** → Tabellen schwerer vertikal zu lesen

---

## Änderungen (alle in `src/html_exporter.py`)

### 1. Sticky TOC Navigation

Fixed Dot-Navigation rechts (`position:fixed; right:1rem; top:50%; transform:translateY(-50%)`), verschwindet auf `<1100px`.

- Jeder `<section>` bekommt `id`-Attribut (`sec-decision`, `sec-results`, `sec-assumptions`, etc.)
- `<nav id="toc">` mit `<a>` Dot pro Section + `data-label` Tooltip
- JS: IntersectionObserver hebt aktiven Dot hervor (`is-active` class)
- `@media (max-width:1100px) { #toc { display:none } }`

### 2. Dark Mode

`@media (prefers-color-scheme: dark)` Override für CSS-Variablen:

```css
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #0f1a1c;
    --surface-2: #162428;
    --ink: #e8f0f2;
    --muted: #8ba4ac;
    --line: rgba(232,240,242,.1);
    --accent: #2dd4bf;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
    --shadow: 0 18px 40px rgba(0,0,0,.35);
  }
  body { background: #0a1214; }
  .hero { background: linear-gradient(135deg,rgba(10,18,20,.98),rgba(15,118,110,.6)); }
  th { background: rgba(232,240,242,.06); }
}
```

### 3. Print-Stylesheet

`@media print` für sauberes A4-Layout:

```css
@media print {
  #toc, .toolbar button, .tree-button, .modal-backdrop { display:none !important; }
  .section { opacity:1 !important; transform:none !important; page-break-inside:avoid; }
  .hero { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .page { max-width:100%; padding:0; }
  .decision-layout, .hero-grid { grid-template-columns:1fr; }
  body { font-size:11pt; }
}
```

### 4. P-Wert Heat-Coloring

Python-seitig in `_build_pairwise_rows()` ein `p_value_style`-Feld hinzufügen:

```python
def _p_heat_style(p_val: float) -> str:
    if p_val < 0.001: return "background:rgba(31,122,90,.22)"
    if p_val < 0.01:  return "background:rgba(31,122,90,.13)"
    if p_val < 0.05:  return "background:rgba(183,121,31,.13)"
    if p_val < 0.1:   return "background:rgba(159,58,56,.08)"
    return ""
```

Im Template: `<td style="{{ row.p_value_style }}">{{ row.p_value }}</td>`

Gilt analog für p-value Zellen in `statistical_rows`.

### 5. Assumption Traffic Lights + Monospace Werte

**A — Traffic Light Icons** in `_build_assumption_summary()`:
```python
icons = {"is-significant": "✓ ", "is-danger": "✗ ", "is-neutral": "~ "}
row["status_label"] = icons.get(row["status_class"], "") + row["status_label"]
```

**B — Monospace CSS-Klasse** für alle numerischen Zellen:
```css
.num-cell { font-family:"Cascadia Mono",Consolas,"Courier New",monospace; font-variant-numeric:tabular-nums; }
```
Im Template alle `<td>` mit Statistiken, p-values, Effektgrößen mit `class="num-cell"` versehen.

---

## Critical Files

| Datei | Methode | Was ändert sich |
|-------|---------|----------------|
| `src/html_exporter.py` | `_template()` Z.771–808 | CSS (dark mode, print, toc, num-cell) + HTML (section ids, toc nav, num-cell classes) + JS (toc observer) |
| `src/html_exporter.py` | `_build_pairwise_rows()` | `p_value_style` Feld hinzufügen |
| `src/html_exporter.py` | `_build_assumption_summary()` | `status_label` mit Icon-Prefix |
| `src/html_exporter.py` | `_build_statistical_rows()` | ggf. p-value Erkennung für heat style |

---

## Verification

1. BioMedStatX starten, Analyse mit paarweisen Vergleichen → HTML exportieren
2. Im Browser:
   - TOC-Dots rechts sichtbar, verschwinden auf schmalem Screen
   - OS Dark Mode → Report wechselt Farbschema
   - Ctrl+P → sauberes A4, keine interaktiven Elemente
   - Pairwise-Tabelle: Grün bei p<0.001, gelblich bei p≈0.05
   - Assumptions: ✓/✗ Präfixe
   - Numerische Werte: Monospace-Font
3. `validation/test_html_exporter.py` → keine Regressions

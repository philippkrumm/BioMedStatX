# A2 + A3: Plot Aesthetics Dialog Fixes — Design

**Date:** 2026-07-08
**Findings:** GD8 (journal palette WCAG contrast), GD12 (`get_config()` silently drops invalid filename)
**File:** `src/ui/dialogs/plot_aesthetics_dialog.py`

## Context

Both findings live in `PlotAestheticsDialog` and were flagged as small, independent
product/UX decisions in `docs/superpowers/specs/2026-07-07-audit-fix-clustering-design.md`
(Tier A, packages A2/A3). Neither shares a function or file section with the other, or with
any Tier B package. Once decided, both fold into one small mechanical implementation package
(B8) rather than needing separate `subagent-driven-development` dispatch.

## A2 — GD8: journal palette WCAG contrast

### Problem

`ColorsTab.journal_palettes` (line 388) bundles 4 journal-styled color presets (Nature,
Science, NEJM, Lancet) — 29 hex values total. 11 of them fail WCAG's 3:1 non-text contrast
floor against a white plot background, several badly (`#FFDC91` at 1.32:1, `#EDC948` at
1.61:1). These are presented to the user as "Professional palettes" for publication-ready
figures, so a low-contrast pick undermines that promise with no in-app warning.

### Decision

Swap the 11 failing entries for same-hue, darker/more-saturated variants that clear
WCAG's 4.5:1 body-text floor (not just the 3:1 non-text floor), matching the target range
already used for this session's `DEFAULT_COLORS` fix (VZ4). Hue is preserved exactly
(confirmed via `colorsys.rgb_to_hls`/`hls_to_rgb` round-trip — e.g. `#FF9DA7` and its
replacement `#EB0018` are both 353.9°); only lightness and saturation change. The 18
already-passing entries in these 4 palettes are untouched.

This means the palettes no longer match each journal's exact published house color
(a deliberate tradeoff — legibility wins over source-fidelity, confirmed with the user).

### Exact replacements

| Palette | Index (0-based) | Old | Old ratio | New | New ratio |
|---|---|---|---|---|---|
| Nature | 1 | `#F28E2B` | 2.42:1 | `#B75C03` | 4.62:1 |
| Nature | 3 | `#76B7B2` | 2.29:1 | `#3F817C` | 4.52:1 |
| Nature | 5 | `#EDC948` | 1.61:1 | `#907207` | 4.57:1 |
| Nature | 7 | `#FF9DA7` | 1.98:1 | `#EB0018` | 4.62:1 |
| Science | 4 | `#56B4E9` | 2.31:1 | `#0F7CBA` | 4.55:1 |
| Science | 5 | `#E69F00` | 2.25:1 | `#9C6C00` | 4.61:1 |
| Science | 6 | `#999999` | 2.85:1 | `#767676` | 4.54:1 |
| NEJM | 2 | `#E18727` | 2.73:1 | `#B16310` | 4.50:1 |
| NEJM | 6 | `#FFDC91` | 1.32:1 | `#9C6A00` | 4.69:1 |
| Lancet | 2 | `#42B540` | 2.65:1 | `#2C882A` | 4.50:1 |
| Lancet | 5 | `#FDAF91` | 1.79:1 | `#D73C00` | 4.63:1 |

Resulting dict (full replacement of `self.journal_palettes` in `ColorsTab.__init__`,
line 388-393):

```python
self.journal_palettes = {
    'Nature': ['#4E79A7', '#B75C03', '#E15759', '#3F817C', '#59A14F', '#907207', '#B07AA1', '#EB0018'],
    'Science': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#0F7CBA', '#9C6C00', '#767676'],
    'NEJM': ['#BC3C29', '#0072B5', '#B16310', '#20854E', '#7876B1', '#6F99AD', '#9C6A00'],
    'Lancet': ['#00468B', '#ED0000', '#2C882A', '#0099B4', '#925E9F', '#D73C00', '#AD002A']
}
```

### Testing

A pure data fix, same shape as VZ4 — one test asserting every hex value in every palette
in `ColorsTab.journal_palettes` clears the 3:1 WCAG non-text floor against white, using
the same relative-luminance/contrast-ratio helpers already established in
`tests/test_default_colors_contrast.py`.

## A3 — GD12: `get_config()` silently drops invalid filename

### Problem

`get_config()` (line 1809) is called from 3 places: the two live-preview paths
(`update_preview_immediately` line 1757, `_do_update_preview` line 1803 — both fire on
*any* settings change in *any* tab, not just the filename field) and the final
accept-time read (line 1915, after `dialog.exec_() == Accepted`).

The current invalid-filename branch (line 1868-1874) shows a blocking `QMessageBox.warning`
and returns an incomplete config dict (missing `file_name`, `create_plot`, `dependent`).
Because `get_config()` runs during live preview, this modal can fire repeatedly — once per
unrelated settings change (color, size, style, ...) — for as long as an invalid character
sits in the filename field, not just once at submission. This is worse than GD12's original
write-up assumed (which only traced the final accept-time call site).

### Decision

Sanitize inline, no modal, ever. Replace each invalid character
(`<>:"/\|?*`) with `_` in the raw filename, write the sanitized string back into
`file_name_edit`'s text (so the field visibly reflects what will actually be used — not a
silent substitution the user never sees), and continue populating the rest of the config
dict exactly as the valid path already does. This fixes GD12 and eliminates the live-preview
modal-spam behavior in the same change, with no new UI state or persistent indicator.

### Fix shape

Replace (line 1868-1875):

```python
if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
    raw_name = self.file_name_edit.text().strip()
    if raw_name and _re.search(r'[<>:"/\\|?*]', raw_name):
        QMessageBox.warning(self, "Invalid filename",
            'File name contains invalid characters: < > : " / \\ | ? *\n'
            'Please use only letters, digits, spaces, hyphens, or underscores.')
        return config
    config['file_name'] = raw_name or None
```

with:

```python
if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
    raw_name = self.file_name_edit.text().strip()
    if raw_name and _re.search(r'[<>:"/\\|?*]', raw_name):
        sanitized_name = _re.sub(r'[<>:"/\\|?*]', '_', raw_name)
        self.file_name_edit.setText(sanitized_name)
        raw_name = sanitized_name
    config['file_name'] = raw_name or None
```

(`_re` is the module's existing `re` import alias, confirmed at the top of the file —
no new import needed. `QMessageBox` import may become unused elsewhere in the file; check
before removing anything, since it's likely used by other dialogs/branches too.)

### Testing

Unit test on `get_config()` directly: construct the dialog (or a minimal fake with just
`file_name_edit`), set an invalid filename (e.g. `bad<>name`), call `get_config()`, assert
no modal is shown (mock `QMessageBox.warning`, assert not called), assert
`config['file_name']` is the sanitized value (`bad__name`), and assert `create_plot` and
`dependent` are still populated (the bug this closes). A second test confirms a valid
filename passes through unchanged.

## Scope note

Both fixes land in one implementation package (B8), sequenced after A1/Tier B since neither
touches a file any other package uses. No shared code between A2 and A3 — they can be
implemented as two independent commits within the same plan.

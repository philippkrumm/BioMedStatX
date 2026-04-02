# Developer UI Design System

Audience: maintainers and contributors.

Purpose: provide a single internal reference for how BioMedStatX should look and feel.
This file is not part of end-user documentation.

## 1. Scope

BioMedStatX has two UI surfaces with different rendering stacks:

1. Desktop app UI (PyQt)
2. HTML report UI (offline export)

Use this document to keep both surfaces coherent while respecting their technical differences.

## 2. Source of Truth

Primary style source files:

1. Desktop theme: assets/BioMedStatX_2_0.qss
2. Desktop layout and motion behavior: src/statistical_analyzer.py
3. HTML report style and interaction behavior: src/html_exporter.py

Planning documents:

1. HTML frontend roadmap: PLAN_html_frontend.md
2. Help Hub deep-link UX concept: docs/DEV_UX_HELP_HUB_DEEP_LINKING.md (if present)

## 3. Design Principles

1. Scientific clarity over decoration.
2. Visual hierarchy by tone and spacing, not heavy borders.
3. Progressive disclosure for complex guidance.
4. Motion should support orientation, never distract.
5. Keep interaction states explicit (default, hover, active, disabled).

## 4. Core Visual Tokens

### 4.1 Color direction

Desktop palette is teal-first with neutral blue-gray support.

Common semantic accents:

1. Primary action/accent: #0f766e
2. Success: #1f7a5a
3. Warning: #b7791f
4. Danger: #9f3a38
5. Info: #0369a1 (desktop) / #38bdf8 (HTML dark mode info accent)

Base surfaces/text:

1. Main backgrounds around #f0f4f8 and #eef5fb
2. Panels mostly #ffffff
3. Primary text around #16313a
4. Muted text around #6b7c84

### 4.2 Spacing scale

Adopt the scale already declared in QSS:

4 / 8 / 12 / 16 / 24 / 32 / 48

Rule:

1. Prefer this scale for margins, paddings, gaps.
2. Avoid one-off spacing values unless there is a strong layout reason.

### 4.3 Shape language

1. Panels/cards: rounded, mostly 12-18 px radius.
2. Inputs/buttons: 8-10 px radius.
3. Buckets: larger rounded outlines (16 px) to signal drop zones.

### 4.4 Elevation and shadows

QSS does not provide real drop shadows, so Python effects are used where needed.

Default elevation helper parameters:

1. blurRadius: 18
2. yOffset: 4
3. opacity: ~0.18

Do not stack multiple shadows unless required for a specific component state.

## 5. Motion and Interaction Feel

### 5.1 Desktop motion

1. Pipeline pulse animation is used for analysis-running feedback.
2. Some card animations are intentionally disabled for platform stability.

Design implication:

1. Favor reliable, minimal motion over aggressive transitions.
2. Keep reduced-motion behavior graceful where possible.

### 5.2 HTML motion

1. Section reveal transitions and tree replay are present.
2. prefers-reduced-motion handling is defined in template CSS.

Keep motion tied to comprehension:

1. Reveals for reading flow.
2. Replay for decision path understanding.

## 6. Component Rules (Desktop)

### 6.1 Panels and cards

1. Use white panel surfaces on soft app background.
2. Keep borders subtle or removed; rely on spacing and contrast.

### 6.2 Buttons

1. Primary actions: filled teal buttons.
2. Secondary actions: ghost style with thin border.
3. Danger actions: neutral background plus red text/border accents.

### 6.3 Mapping buckets

1. Dashed outline and soft fill in idle state.
2. Stronger accent + clearer fill on drag hover.
3. Info button style must remain compact and discoverable.

### 6.4 Help Hub

1. Two-column layout (navigation + content).
2. Search-first navigation.
3. Copy example data action as explicit utility button.

## 7. Component Rules (HTML Report)

1. Keep report readable in long-scroll format.
2. Maintain dark-mode and print behavior.
3. Preserve sticky TOC behavior on wide viewports and hide on small screens.
4. Keep info-panels optional and lightweight.
5. Keep numeric-heavy cells in tabular/monospace style where applied.

## 8. UX Consistency Guardrails

When changing UI, verify:

1. Color semantics are consistent (success/warning/danger/info meaning unchanged).
2. Spacing scale is respected.
3. Button hierarchy remains obvious.
4. Interactive states are visible and accessible.
5. New help affordances align with progressive disclosure.
6. Desktop and HTML themes feel like the same product family.

## 9. Change Workflow for UI Work

For any non-trivial UI change:

1. Update QSS and related Python behavior together if both are affected.
2. Add or update internal docs when introducing new interaction patterns.
3. Include screenshots or short video/GIF in PR when visual behavior changes.
4. Note platform-specific caveats (Windows/macOS/Linux) in PR description.

## 10. Quick QA Checklist

1. Visual hierarchy is readable at first glance.
2. Hover/focus/disabled states are clearly distinguishable.
3. No clipped text in common dialog sizes.
4. Spacing feels consistent between major sections.
5. Shadow usage is subtle and not muddy.
6. Animations do not cause instability or jank.
7. HTML export remains readable in light, dark, and print contexts.
8. Every text has to be in englisch!

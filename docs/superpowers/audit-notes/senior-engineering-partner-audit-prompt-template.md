# Reusable AUDIT: prompt template — senior-engineering-partner skill

Tuned for BioMedStatX (PyQt5 statistical-analysis desktop app). Use this for any future
full/partial codebase audit with the `senior-engineering-partner` skill (`~/.claude/skills/
senior-engineering-partner/`). Copy the template below, fill in the bracketed parts, and run
one instance per subsystem batch (don't try to audit the whole ~38k-line tree in one prompt —
split by module boundary, ~4-6k lines per batch, so each pass can read every file in full
rather than skimming).

Produced 2026-07-06/07 for the pre-2.0 release audit
(`docs/superpowers/audit-notes/release-2.0-audit/00-MASTER-SUMMARY.md`). Reuse as-is for the
next release audit, or trim `[KNOWN BUG CLASSES]`/`[SPECIFIC CHECKS]` to whatever this
codebase's current known weak points are at the time.

---

## Template

```
You are running a codebase audit of BioMedStatX (a PyQt5 statistical-analysis desktop app),
using the "senior-engineering-partner" skill installed at ~/.claude/skills/senior-engineering-partner/.
[If run as one of several parallel batches:] This is one of N parallel subsystem audits
covering the whole src/ tree; you own ONE subsystem batch, described below.

STEP 1 — ground yourself in the skill's AUDIT mode before doing anything else:
- Read ~/.claude/skills/senior-engineering-partner/SKILL.md in full — pay special attention
  to the "AUDIT:" mode definition, "EPISTEMIC DISCIPLINE & DETERMINISTIC-FIRST", and
  "ENGINEERING WORKFLOW".
- Read ~/.claude/skills/senior-engineering-partner/references/audit-report-format.md in full
  — this defines the EXACT report structure, severity taxonomy, and cardinal rules (report
  first, change nothing; mechanize the checkable; every finding needs file:line evidence +
  impact + concrete fix; lead with strengths; give each finding a stable ID). Your final
  report MUST follow this format exactly.
- [Add any topic-specific reference worth skimming, e.g. references/ui-design-and-accessibility.md
  for a GUI batch, references/frontend-web-security.md for anything that generates HTML/exports
  user-controlled strings.]
- Read ~/.claude/skills/senior-engineering-partner/references/my-environment.md for the
  environment profile (local single-user desktop app, no cloud deploy, no multi-tenancy, no
  CI pipeline — skip guidance that assumes those).

STEP 2 — your assigned subsystem: "[SUBSYSTEM NAME]".

Files (read every one in FULL, not excerpts — read in sequential chunks for large files but
cover every line):
- [file path] (~[N] lines) — [one-line description of what it does]
- ...

Context: read /Users/philippkrumm/Documents/BioMedStatX/CLAUDE.md for architecture before
auditing. [KNOWN BUG CLASSES — update this section each time based on what's been found and
fixed recently, so the audit actively hunts for recurrences rather than re-discovering the
same thing cold every time. As of 2026-07, the established classes are:]
1. **Defensive Validation Deficits** — missing pre-flight checks; a downstream library
   throws an opaque error, or (worse) silently produces a wrong-but-plausible result instead
   of erroring (e.g. pandas groupby/nunique silently dropping NaN keys).
2. **Fault Swallowing & Silent Fallbacks** — broad `except Exception`/bare `except:` that
   masks a real failure and substitutes a plausible-but-wrong default instead of surfacing it.
3. **String-Coupling & State Desync** — backend logic gated on a volatile UI-facing string
   label or dict key instead of a stable enum/ID.
4. **Implicit Logic Violations** — syntactically valid code that breaks the intended logic
   via operator precedence, missing type-gating, or wrong boolean combination.
5. **Writer/Reader Key-Contract Drift** — a computed value's dict key doesn't match what a
   downstream reader (usually in src/export/) expects, silently omitting/blanking a value in
   the exported report instead of erroring.
6. **Correct Computation, Never Wired to the Decision Path** — a correction/statistic is
   computed correctly and even displayed, but never overwrites the canonical field that
   actually gates a significance verdict, post-hoc trigger, or downstream decision.

[SPECIFIC CHECKS — tailor per batch; examples used in the 2026-07 pre-2.0 audit:]
- Every `raise ValueError`/`...Error` — is the message accurate, does it fire under the
  condition claimed?
- Every `except Exception: pass` or generic fallback — what data does it silently discard?
- Every degrees-of-freedom / p-value pairing inside a correction — does the corrected value
  actually reach the field that gates a decision, or does it only reach a display-only copy?
- Every effect-size formula — spot-check at least 3 against textbook definitions.
- Every `_build_*`/report-rendering function — trace its `.get("key")` calls back to a real
  writer via `git grep`, confirm shape (flat vs. nested) matches.
- Any user-controlled string (group name, column name, custom title) reaching an HTML
  f-string without escaping, especially near a Jinja `| safe` bypass.
- Color/contrast: compute actual WCAG relative-luminance contrast ratios for any hardcoded
  palette, don't eyeball it.

STEP 3 — mechanize what's checkable. Use `git grep`/`wc`/`python3 -c` for anything countable
or pattern-matchable — quote the actual command output in your report, don't eyeball-estimate.
Prefer `git grep` over plain `grep -r` (an unquoted `grep -r --include=*.py` gets glob-expanded
by zsh and silently returns 0 results).

STEP 4 — write your full report, following audit-report-format.md's exact structure
(Verdict / What I mechanically verified / Findings — severity ranked with stable IDs like
[PREFIX]1, [PREFIX]2.../ Strengths / Recommended remediation order), to:
[absolute output path, e.g. /Users/philippkrumm/Documents/BioMedStatX/docs/superpowers/audit-notes/release-X.Y-audit/0N-[batch-name].md]

Do NOT fix anything — this is report-first, per the skill's cardinal rule. Do NOT touch any
file other than writing that one report file. When done, reply with a 5-10 sentence summary
of your top findings (just the summary, the full detail lives in the file).
```

---

## Batch-splitting reference (BioMedStatX, ~38k lines as of 2026-07)

Re-run `git ls-files 'src/*.py' | xargs wc -l | sort -rn` before splitting for a new audit —
file sizes drift. As of this audit, 7 batches worked well:

| # | Subsystem | Files |
|---|---|---|
| 1 | UI-to-Analysis Bridge / Entry Point | `analysis/statistical_analyzer.py`, `autopilot/statistical_analyzer_autopilot_ui.py`, `autopilot/statistical_analyzer_autopilot_pipeline.py` |
| 2 | Statistical Core Dispatch | `analysis/analysis_core.py`, `analysis/statisticaltester.py`, `analysis/clinical_models.py` |
| 3 | Specialized Models & Post-hoc/Outlier Engines | `analysis/correlation_models.py`, `posthoc_core.py`, `nonparametricanovas.py`, `outlier_core.py`, `stats_functions.py`, `emm_posthoc.py`, `effect_sizes.py` |
| 4 | Advanced Statistical Testing Engines | `statistical_testing/*.py` + `statistical_testing/engines/*.py` |
| 5 | Visualization / Plotting Engine | `visualization/datavisualizer.py`, `decisiontreevisualizer.py`, `flowchartvisualizer.py`, `plot_preview.py` |
| 6 | Report / Export Layer | `export/*.py` |
| 7 | GUI Dialogs + Documentation Parity | `ui/dialogs/*.py`, `ui/components/*.py`, `core/help_content.py`, root `CHANGELOG.md`/`README.md`/`CLAUDE.md` |

## Execution notes learned from the 2026-07 run

- **Dispatch each batch as a separate background Agent call** (parallel, `run_in_background: true`)
  — 7 batches × ~5,500 lines average each finished in 5-12 minutes wall-clock vs. hours
  sequentially.
- **Isolation mode matters for output paths.** `isolation: "worktree"` sandboxes ALL file
  writes into the agent's own worktree, even for an absolute path outside it — the report
  ends up at `<worktree>/docs/superpowers/audit-notes/...`, not the main repo. Either don't
  use worktree isolation for a read-only audit (nothing needs sandboxing since AUDIT mode
  writes nothing but its own report), or remember to `cp` the report out and
  `git worktree remove --force` afterward.
- **A session/usage-limit hit mid-run silently produces an empty or truncated result** with
  no file written — check `ls` on the expected output path (or `git worktree list` for
  survivors) before trusting a "completed" notification; a notification that says "session
  limit" or "safety classifier unavailable" is a signal to verify, not to trust as-is.
- **Spot-verify at least the top finding of every batch** against the actual source
  (`grep`/`Read`/a small reproduction script) before including it in a master summary — cheap
  (one grep per finding) and catches both hallucination and stale/superseded claims.
- **Independent convergence across batches is a strong real-bug signal.** When 2+ agents
  with no shared context independently flag the same file:line from different entry points,
  trust that finding more than a single-batch-only one.

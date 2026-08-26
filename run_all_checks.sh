#!/usr/bin/env bash
# Every automated check this repo has, in one pass.
#
# There are four of them now, and the ones that are not pytest are easy to
# forget when clicking through by hand -- which is exactly how a fuzzer stops
# being run. Counts are printed per stage; a non-zero exit means at least one
# stage found something. Override the seed counts for a quicker pass:
#
#   FUZZ_SEEDS=50 IMPORT_SEEDS=30 VISUAL_SEEDS=15 ./run_all_checks.sh
#
# The visual stage drives a headless browser, so it costs roughly five seconds
# a seed -- an order of magnitude more than the other two. Its default count is
# small for that reason, not because it matters less.
set -uo pipefail
cd "$(dirname "$0")"

FUZZ_SEEDS="${FUZZ_SEEDS:-300}"
IMPORT_SEEDS="${IMPORT_SEEDS:-250}"
VISUAL_SEEDS="${VISUAL_SEEDS:-60}"
status=0

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "1/5  consistency validator"
python tools/validate_consistency.py || status=1

banner "2/5  test suite (tests/ + validation/)"
python -m pytest tests/ validation/ -q || status=1

banner "3/5  analysis fuzzer  (${FUZZ_SEEDS} seeds: DataFrame -> report)"
python -m fuzzing.run_fuzzer --count "$FUZZ_SEEDS" || status=1

banner "4/5  import fuzzer  (${IMPORT_SEEDS} seeds: file -> mapping -> analysis -> report)"
python -m fuzzing.run_import_fuzzer --count "$IMPORT_SEEDS" || status=1

banner "5/5  visual fuzzer  (${VISUAL_SEEDS} seeds: report -> browser -> figure -> export)"
if ! python -c "import playwright" 2>/dev/null; then
    printf 'Playwright is missing -- it is a development-only dependency:\n'
    printf '    pip install playwright && python -m playwright install chromium\n'
    status=1
else
    # The self-check runs first: a fuzzer whose oracles cannot fail turns the
    # stage below into an expensive way of printing OK, so it is worth the
    # twenty seconds before the five minutes.
    python -m fuzzing.visual_selfcheck || status=1
    python -m fuzzing.run_visual_fuzzer --count "$VISUAL_SEEDS" || status=1
fi

if [ "$status" -eq 0 ]; then
    printf '\n\033[32mAll five stages clean.\033[0m\n'
else
    printf '\n\033[31mAt least one stage reported findings -- see the sections above.\033[0m\n'
fi
exit "$status"

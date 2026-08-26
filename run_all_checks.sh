#!/usr/bin/env bash
# Every automated check this repo has, in one pass.
#
# There are three of them now, and the second one is easy to forget when
# clicking through by hand -- which is exactly how a fuzzer stops being run.
# Counts are printed per stage; a non-zero exit means at least one stage found
# something. Override the seed counts for a quicker pass:
#
#   FUZZ_SEEDS=50 IMPORT_SEEDS=30 ./run_all_checks.sh
#
set -uo pipefail
cd "$(dirname "$0")"

FUZZ_SEEDS="${FUZZ_SEEDS:-300}"
IMPORT_SEEDS="${IMPORT_SEEDS:-250}"
status=0

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "1/4  consistency validator"
python tools/validate_consistency.py || status=1

banner "2/4  test suite (tests/ + validation/)"
python -m pytest tests/ validation/ -q || status=1

banner "3/4  analysis fuzzer  (${FUZZ_SEEDS} seeds: DataFrame -> report)"
python -m fuzzing.run_fuzzer --count "$FUZZ_SEEDS" || status=1

banner "4/4  import fuzzer  (${IMPORT_SEEDS} seeds: file -> mapping -> analysis -> report)"
python -m fuzzing.run_import_fuzzer --count "$IMPORT_SEEDS" || status=1

if [ "$status" -eq 0 ]; then
    printf '\n\033[32mAll four stages clean.\033[0m\n'
else
    printf '\n\033[31mAt least one stage reported findings -- see the sections above.\033[0m\n'
fi
exit "$status"

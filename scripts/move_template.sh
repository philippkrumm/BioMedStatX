#!/usr/bin/env bash
set -e
# Move the Excel template into docs/ and update git index
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/StatisticalAnalyzer_Excel_Template.xlsx"
DEST="$REPO_ROOT/docs/StatisticalAnalyzer_Excel_Template.xlsx"

if [ ! -f "$SRC" ]; then
  echo "Source template not found at $SRC"
  echo "If you already moved it manually, nothing to do."
  exit 1
fi

mkdir -p "$REPO_ROOT/docs"
if command -v git >/dev/null 2>&1; then
  git mv "$SRC" "$DEST" || {
    echo "git mv failed; trying plain move"
    mv "$SRC" "$DEST"
  }
  echo "Template moved to docs/ and staged with git." 
else
  mv "$SRC" "$DEST"
  echo "Template moved to docs/ (git not available, please `git add` and commit manually)."
fi

echo "Done. Please commit the change:"
echo "  git add docs/StatisticalAnalyzer_Excel_Template.xlsx && git commit -m 'Move excel template to docs/'"

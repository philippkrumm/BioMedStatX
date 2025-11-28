#!/usr/bin/env bash
set -e
# start.sh - portable launcher for BioMedStatX
# Usage: ./start.sh [args]

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

run_python_entrypoint() {
  echo "Falling back to Python entrypoint..."
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    echo "No Python interpreter found. Please install Python 3 and try again." >&2
    exit 1
  fi
  exec "$PY" "$REPO_ROOT/Source_Code/statistical_analyzer.py" "$@"
}

# Try common binary locations/names
try_exec() {
  local path="$1"
  if [ -x "$path" ]; then
    echo "Starting binary: $path"
    exec "$path" "$@"
  fi
}

# Check some likely places for precompiled binaries
try_exec "$REPO_ROOT/BioMedStatX"
try_exec "$REPO_ROOT/BioMedStatX.app/Contents/MacOS/BioMedStatX"
try_exec "$REPO_ROOT/BioMedStatX.exe"
try_exec "$REPO_ROOT/dist/BioMedStatX"
try_exec "$REPO_ROOT/dist/BioMedStatX.exe"
try_exec "$REPO_ROOT/build/BioMedStatX"

# If running on macOS, try .app bundle via open
if [ "$(uname -s)" = "Darwin" ]; then
  if [ -d "$REPO_ROOT/BioMedStatX.app" ]; then
    echo "Opening macOS app bundle..."
    open "$REPO_ROOT/BioMedStatX.app"
    exit $?
  fi
fi

# No binary found — instructive fallback to Python
echo "No native binary found in repository."
echo "You can:
 - Run the Python source directly (recommended for Linux users): ./start.sh
 - Or run a precompiled binary if you downloaded one for your OS."

run_python_entrypoint "$@"

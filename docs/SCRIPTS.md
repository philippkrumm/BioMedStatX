# Scripts and How to Use Them

This document describes the helper scripts included in this repository, what they do, and how to run them on Windows and Unix-like systems. All examples assume you run the commands from the repository root.

## Purpose

- Make it easy for developers to run the application from source.
- Provide a reproducible way to generate documentation PDFs locally.
- Offer small convenience wrappers which handle common platform differences (venv activation, pandoc location).

## Files Covered

- `start.sh` (repo root) — optional launcher for Linux/macOS (not present by default in this repo).
- `start.bat` (repo root) — optional launcher for Windows (not present by default in this repo).
- `Source_Code/start_from_source.sh` — helper to run the application from source inside a Python virtual environment (optional).
- `tools/generate_pdfs.bat` — helper batch file to regenerate PDFs from Markdown using `pandoc` + `xelatex` (present).

> Note: Some start scripts may be missing if they haven't been added yet. This document explains expected behavior and usage so you can create/modify them if needed.

## `tools/generate_pdfs.bat` (existing)

Purpose: convert Markdown files in `docs/` to PDF using `pandoc` and your LaTeX engine. The script prefers a system `pandoc` on `PATH`, otherwise it falls back to a bundled/anaconda `pandoc` path.

Usage (Windows, cmd.exe / PowerShell):

```cmd
.\tools\generate_pdfs.bat
```

What it does:
- Invokes `pandoc` with `--pdf-engine=xelatex` and `-V geometry:margin=1in`.
- Passes `--resource-path=docs` so images under `docs/HowToScreenshots/` are found.

Requirements:
- `pandoc` must be installed (or available in the fallback path inside Anaconda). If you want the script to use `pandoc` from `PATH`, ensure `pandoc.exe` is in your PATH.
- A TeX engine (XeLaTeX/LuaLaTeX) installed and available.

Troubleshooting:
- If images are missing in the produced PDFs, verify that image files exist under `docs/` (or the path referenced in the Markdown) and re-run the script.

## Start scripts (recommended layout)

The repository should include simple start scripts so users can run the app without remembering long commands. Below are suggested behaviors and example implementations.

### `start.sh` (Linux/macOS)
- Detect OS and look for a precompiled binary (`BioMedStatX` or `BioMedStatX.app`) in the repo root or `releases/`.
- If not found, try to activate a Python virtual environment `.venv` and run `python Source_Code/statistical_analyzer.py`.

Example usage:

```bash
chmod +x ./start.sh   # once
./start.sh
```

Example pseudo-implementation (bash):

```bash
#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$ROOT/BioMedStatX" ]; then
  exec "$ROOT/BioMedStatX"
fi
if [ -d "$ROOT/.venv" ]; then
  source "$ROOT/.venv/bin/activate"
  python "$ROOT/Source_Code/statistical_analyzer.py"
else
  echo "No binary and no .venv found. To run from source, create a venv: python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt";
fi
```

### `start.bat` (Windows)

Suggested behavior (cmd.exe):

```bat
@echo off
set ROOT=%~dp0
if exist "%ROOT%\BioMedStatX.exe" (
  "%ROOT%\BioMedStatX.exe"
  exit /b
)
if exist "%ROOT%\.venv\Scripts\activate.bat" (
  call "%ROOT%\.venv\Scripts\activate.bat"
  python "%ROOT%\Source_Code\statistical_analyzer.py"
  exit /b
)
echo No binary or virtual environment found. Create one with: python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt
```

## `Source_Code/start_from_source.sh` (recommended)

Purpose: developer helper to run the main application while ensuring a venv is active.

Example usage:

```bash
cd Source_Code
./start_from_source.sh
```

Example implementation:

```bash
#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")" && pwd)/.."
if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv "$ROOT/.venv"
  echo "Created .venv; activate it and install requirements: pip install -r requirements.txt"
fi
source "$ROOT/.venv/bin/activate"
python "$ROOT/Source_Code/statistical_analyzer.py"
```

## Virtual environment and dependency tips

1. Create a virtual environment (cross-platform):

```bash
python -m venv .venv
# Windows (cmd.exe)
.\.venv\Scripts\activate
# PowerShell
.\.venv\Scripts\Activate.ps1
# Unix
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. If you only want to run the GUI binary distributed in Releases, you don't need a Python environment.

## Security and path notes

- Scripts use relative paths so they work from any current working directory. In Bash use `$(dirname "$0")` and in Windows use `%~dp0`.
- Be cautious when running scripts obtained from the internet. Review scripts before executing.

## Where to document more

- This file is intended as a quick reference. If you want, we can add the actual `start.sh` and `start.bat` files to the repo and keep this doc in sync.

---
If you'd like, I can (A) add `start.sh` and `start.bat` to the repository now, or (B) only document them and leave creation to you. Reply with `A` or `B`.

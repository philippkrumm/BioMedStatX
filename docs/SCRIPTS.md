# Scripts and How to Use Them

This document describes the helper scripts included in this repository, what they do, and how to run them. All examples assume you run the commands from the repository root.

## Purpose

- Let developers start the application from source without typing the full Python command.
- Offer platform-specific convenience wrappers for Linux/macOS and Windows.

## Files Covered

- `start.sh` (repo root): launcher for Linux/macOS. Tries binary locations first and falls back to the Python entrypoint.
- `run.bat` (repo root): developer convenience script for Windows. Calls a hardcoded virtual environment path — see the Windows section below before using it.

## Start Scripts

### `start.sh` (Linux/macOS)

- Tries binary locations in this order: `BioMedStatX`, `BioMedStatX.app/Contents/MacOS/BioMedStatX`, `BioMedStatX.exe`, `dist/BioMedStatX`, `dist/BioMedStatX.exe`, `build/BioMedStatX`.
- On macOS, if a `.app` bundle is found, opens it via `open`.
- If no binary is found, falls back to `python3` (or `python`) and runs `src/analysis/statistical_analyzer.py`.

Example usage:

```bash
chmod +x ./start.sh
./start.sh
```

### `run.bat` (Windows)

`run.bat` is a minimal developer script. It calls a hardcoded Python interpreter path:

```bat
@echo off
cd /d "%~dp0"
"C:\bmx_venv\Scripts\python.exe" src\analysis\statistical_analyzer.py %*
```

Before using it, edit the path `C:\bmx_venv\Scripts\python.exe` to match your local virtual environment. If your venv is at `.venv`, change it to `.venv\Scripts\python.exe`.

Example (after editing):

```bat
.\run.bat
```

End users should use the packaged `BioMedStatX.exe` from the GitHub Releases page instead.

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

- `start.sh` uses `$(dirname "$0")` to resolve paths relative to the script location.
- `run.bat` uses `%~dp0` for the same purpose, but the Python interpreter path is hardcoded and must be updated manually.
- Review scripts obtained from the internet before running them.

## Notes

- End users on Windows and macOS will usually use the packaged app from the GitHub Releases page.
- The launcher scripts are for running from a cloned repository or a local source build.

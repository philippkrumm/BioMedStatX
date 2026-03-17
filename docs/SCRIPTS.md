# Scripts and How to Use Them

This document describes the helper scripts included in this repository, what they do, and how to run them on Windows and Unix-like systems. All examples assume you run the commands from the repository root.

## Purpose

- Make it easy for developers to run the application from source.
- Provide a reproducible way to generate documentation PDFs locally.
- Offer small convenience wrappers which handle common platform differences (venv activation, pandoc location).

## Files Covered

- `Start_BioMedStatX_on_Linux.sh` (repo root) — launcher for Linux/macOS. It prefers a native binary and otherwise runs the Python entrypoint.
- `start.bat` (repo root) — launcher for Windows. It prefers a native `.exe` and otherwise runs the Python entrypoint.

There is currently no `tools/generate_pdfs.bat` file in this repository, and there is no `Source_Code/start_from_source.sh`.

## Start Scripts

The repository includes two launcher scripts so the app can be started from the repository root without typing the full Python command.

### `Start_BioMedStatX_on_Linux.sh` (Linux/macOS)
- Looks for common binary locations first, including `BioMedStatX`, `BioMedStatX.app`, `BioMedStatX.exe`, and `dist/` builds.
- On macOS it can open a `.app` bundle via `open`.
- If no binary is found, it falls back to `python3` or `python` and runs `Source_Code/statistical_analyzer.py`.

Example usage:

```bash
chmod +x ./Start_BioMedStatX_on_Linux.sh
./Start_BioMedStatX_on_Linux.sh
```

### `start.bat` (Windows)
- Looks for `BioMedStatX.exe` in the repo root.
- Looks for `dist\BioMedStatX.exe` as a secondary location.
- If no native executable is found, it falls back to `python Source_Code\statistical_analyzer.py`.

Example usage:

```bat
.\start.bat
```

Behavior summary:

```bat
@echo off
if exist "%REPO_ROOT%\BioMedStatX.exe" (
  start "" "%REPO_ROOT%\BioMedStatX.exe"
  goto :eof
)
if exist "%REPO_ROOT%\dist\BioMedStatX.exe" (
  start "" "%REPO_ROOT%\dist\BioMedStatX.exe"
  goto :eof
)
python "%REPO_ROOT%\Source_Code\statistical_analyzer.py"
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

## Notes

- End users on Windows and macOS will usually use the packaged app from the GitHub Releases page.
- The launcher scripts are mainly useful for running from a cloned repository or a local unpacked build.

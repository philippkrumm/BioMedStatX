@echo off
rem start.bat - launcher for BioMedStatX on Windows
set REPO_ROOT=%~dp0

if exist "%REPO_ROOT%\BioMedStatX.exe" (
  echo Starting BioMedStatX.exe
  start "" "%REPO_ROOT%\BioMedStatX.exe" %*
  goto :eof
)

if exist "%REPO_ROOT%\dist\BioMedStatX.exe" (
  echo Starting dist\BioMedStatX.exe
  start "" "%REPO_ROOT%\dist\BioMedStatX.exe" %*
  goto :eof
)

rem Fallback to Python entrypoint
if exist "%REPO_ROOT%\tools\run_python.cmd" (
  echo No native exe found - running Python entrypoint
  call "%REPO_ROOT%\tools\run_python.cmd" "%REPO_ROOT%\src\statistical_analyzer.py" %*
  goto :eof
)

echo No suitable binary or Python found. Please install Python or place the precompiled exe in the repository root.
exit /b 1

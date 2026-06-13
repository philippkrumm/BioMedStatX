@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "PYTHON_EXE="

rem 1. Prefer the user's known local Anaconda install
if exist "%USERPROFILE%\AppData\Local\anaconda3\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\AppData\Local\anaconda3\python.exe"
)

rem 2. Respect an active conda environment if available
if not defined PYTHON_EXE if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
  )
)

rem 3. Fallback to python on PATH
if not defined PYTHON_EXE (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PYTHON_EXE=python"
  )
)

if not defined PYTHON_EXE (
  echo No suitable Python interpreter found.
  echo Checked:
  echo   %USERPROFILE%\AppData\Local\anaconda3\python.exe
  echo   %%CONDA_PREFIX%%\python.exe
  echo   python on PATH
  exit /b 1
)

"%PYTHON_EXE%" %*
exit /b %errorlevel%

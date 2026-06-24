@echo off
cd /d "%~dp0"
"C:\bmx_venv\Scripts\python.exe" src\analysis\statistical_analyzer.py %*

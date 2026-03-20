@echo off
REM BioMedStatX Dead-Code Audit
REM Aufruf: paudit.bat          (Terminal-Output)
REM         paudit.bat --report  (+ audit_report.md schreiben)

cd /d "%~dp0"
python audit_dead_code.py %*

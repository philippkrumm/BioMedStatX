@echo off
REM Move the Excel template into docs\ and update git index (Windows)
SET REPO_ROOT=%~dp0..\
SET SRC=%~dp0..\StatisticalAnalyzer_Excel_Template.xlsx
SET DEST=%~dp0..\docs\StatisticalAnalyzer_Excel_Template.xlsx

IF NOT EXIST "%SRC%" (
  echo Source template not found at %SRC%
  echo If you already moved it manually, nothing to do.
  exit /b 1
)

IF NOT EXIST "%~dp0..\docs" (
  mkdir "%~dp0..\docs"
)

where git >nul 2>&1
IF %ERRORLEVEL%==0 (
  git mv "%SRC%" "%DEST%" || (
    echo git mv failed; trying plain move
    move "%SRC%" "%DEST%"
  )
  echo Template moved to docs\ and staged with git.
) ELSE (
  move "%SRC%" "%DEST%"
  echo Template moved to docs\ (git not available, please `git add` and commit manually).
)

echo Done. Please commit the change:
echo   git add docs\StatisticalAnalyzer_Excel_Template.xlsx && git commit -m "Move excel template to docs/"

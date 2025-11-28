@echo off
REM Generate PDFs from markdown files using pandoc + xelatex
REM Adjust PANDOC path if pandoc is not in PATH
REM Prefer system pandoc if available
where pandoc >nul 2>nul
if %errorlevel%==0 (
	pandoc docs\HowTo.md -o docs\HowTo.pdf --pdf-engine=xelatex -V geometry:margin=1in --resource-path=docs
	pandoc docs\ADVANCED_ANOVA_GUIDE.md -o docs\ADVANCED_ANOVA_GUIDE.pdf --pdf-engine=xelatex -V geometry:margin=1in --resource-path=docs
) else (
	"C:\Users\pkrumm\AppData\Local\anaconda3\Library\bin\pandoc.exe" docs\HowTo.md -o docs\HowTo.pdf --pdf-engine=xelatex -V geometry:margin=1in --resource-path=docs
	"C:\Users\pkrumm\AppData\Local\anaconda3\Library\bin\pandoc.exe" docs\ADVANCED_ANOVA_GUIDE.md -o docs\ADVANCED_ANOVA_GUIDE.pdf --pdf-engine=xelatex -V geometry:margin=1in --resource-path=docs
)

echo PDFs generated in docs\
pause
@echo off
REM Generate PDFs from markdown files using pandoc + xelatex
REM Adjust PANDOC path if pandoc is not in PATH
"C:\Users\pkrumm\AppData\Local\anaconda3\Library\bin\pandoc.exe" docs\HowTo.md -o docs\HowTo.pdf --pdf-engine=xelatex -V geometry:margin=1in
"C:\Users\pkrumm\AppData\Local\anaconda3\Library\bin\pandoc.exe" docs\ADVANCED_ANOVA_GUIDE.md -o docs\ADVANCED_ANOVA_GUIDE.pdf --pdf-engine=xelatex -V geometry:margin=1in

echo PDFs generated in docs\
pause
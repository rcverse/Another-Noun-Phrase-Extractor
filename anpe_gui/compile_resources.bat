@echo off
echo Compiling Qt resources...
pyrcc6 -o resources.rcc resources.qrc
if %ERRORLEVEL% neq 0 (
    echo Error compiling resources! Make sure PyQt6 is installed and pyrcc6 is in your PATH.
    pause
    exit /b 1
)
echo Resources compiled successfully to resources.rcc
pause 
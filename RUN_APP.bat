@echo off
REM Starts the HabiScan demo app and opens it in the browser. Close this window to stop it.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP.bat first.
    pause
    exit /b 1
)
start "" cmd /c "timeout /t 15 /nobreak >nul & start http://127.0.0.1:7860"
".venv\Scripts\python.exe" app.py
pause

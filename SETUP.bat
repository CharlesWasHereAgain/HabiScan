@echo off
REM HabiScan one-time setup: creates .venv and installs the Python packages.
REM Needs Python 3.10+ (https://www.python.org/downloads/ - tick "Add python.exe to PATH")
REM and an internet connection (downloads ~300 MB of packages, mostly PyTorch).
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during install, then run SETUP.bat again.
    pause
    exit /b 1
)

echo Using:
%PY% --version
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv || (echo Failed to create .venv & pause & exit /b 1)
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)

echo.
echo Setup finished. Double-click RUN_APP.bat to start HabiScan.
pause

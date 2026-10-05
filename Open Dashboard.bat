@echo off
rem ---------------------------------------------------------------------------
rem  Solar Flare Prediction & Space Weather Alert System - one-click launcher.
rem
rem  Starts the local dashboard server and opens it in the default browser.
rem  A desktop shortcut to this file carries the hotkey Ctrl + Alt + S, so the
rem  dashboard can be opened without touching a terminal during a review.
rem
rem  Re-create that shortcut at any time with:
rem      powershell -ExecutionPolicy Bypass -File "scripts\install_shortcut.ps1"
rem ---------------------------------------------------------------------------
setlocal
set "PORT=8791"
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3.12"
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [!] No virtual environment found at .venv
  echo     Falling back to the system Python. If this fails, run:
  echo         py -3.12 -m venv .venv
  echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo         .venv\Scripts\python.exe -m pip install -e .
  echo.
)

rem If something is already serving on the port, just open the browser.
netstat -ano | findstr /r /c:"LISTENING.*:%PORT% " >nul 2>&1
if %errorlevel%==0 (
  echo Dashboard already running on port %PORT%.
  start "" "http://localhost:%PORT%/index.html"
  exit /b 0
)

echo Starting the dashboard on http://localhost:%PORT% ...
start "Solar Flare dashboard" /min "%PY%" -m solarflare dashboard --port %PORT%

rem Give the server a moment, then open the browser.
powershell -NoProfile -Command "Start-Sleep -Milliseconds 1400" >nul 2>&1
start "" "http://localhost:%PORT%/index.html"

echo.
echo Dashboard opened. The server runs in a minimised window titled
echo "Solar Flare dashboard" - close that window to stop it.
endlocal

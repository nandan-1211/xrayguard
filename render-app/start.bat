@echo off
title XRayGuard - local server
cd /d "%~dp0"
echo.
echo  ==========================================
echo    XRayGuard - starting on your computer
echo  ==========================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo  Python is not installed. Install it from https://www.python.org/downloads/
  echo  and tick "Add Python to PATH" during setup, then run this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo  First run: setting up, this takes 1-2 minutes...
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo  Installing packages failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)
findstr /r /c:"_API_KEY=[A-Za-z0-9]" .env >nul 2>nul || del .env 2>nul
findstr /c:":\\" .env >nul 2>nul && del .env 2>nul
if not exist ".env" (
  echo.
  echo  Paste your FREE Google Gemini API key and press Enter.
  echo  Get one free at https://aistudio.google.com/apikey  ^(no card needed^)
  set /p KEY="  Gemini key: "
  call echo GEMINI_API_KEY=%%KEY%%> .env
)
echo.
echo  Opening http://localhost:8000 ...
echo  Keep this window open while you use XRayGuard. Close it to stop.
echo.
start "" http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause

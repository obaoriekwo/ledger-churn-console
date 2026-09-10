@echo off
setlocal
title Ledger - Credit Card Attrition Risk Console
cd /d "%~dp0"

echo ============================================================
echo   Ledger - starting up
echo ============================================================
echo.

REM --- Find a working Python launcher ---
where py >nul 2>nul
if %errorlevel%==0 (
    set PYCMD=py
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYCMD=python
    ) else (
        echo [ERROR] Python was not found on this computer.
        echo Install it from https://www.python.org/downloads/
        echo IMPORTANT: during install, tick "Add python.exe to PATH".
        echo.
        pause
        exit /b 1
    )
)

echo Using: %PYCMD%
echo.

REM --- Install/confirm required packages (fast if already installed) ---
echo Checking required packages (this can take a minute the first time)...
%PYCMD% -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Packages OK. Training models and starting the server...
echo (This trains 3 models with 5-fold cross-validation - genuinely takes
echo  20-40 seconds depending on your machine. The browser will open
echo  automatically ONLY once the server actually responds - no guessing.)
echo.
echo Keep this window open while you use the app.
echo Close this window, or press CTRL+C, to stop the server.
echo ============================================================
echo.

REM --- Poll the server until it actually responds, THEN open the browser.
REM     (A previous version used a fixed 8-second wait, which opened the
REM     browser before training finished and showed "connection refused" -
REM     this polls instead, so it works regardless of machine speed.) ---
start "" cmd /c "for /L %%i in (1,1,90) do (curl -s -o nul -w \"\" http://127.0.0.1:5000/ >nul 2>nul && (start http://127.0.0.1:5000/ & exit) || timeout /t 2 >nul)"

REM --- Run the app (this window stays open showing logs) ---
%PYCMD% app.py

pause

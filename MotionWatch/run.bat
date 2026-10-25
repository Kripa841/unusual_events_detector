@echo off
title MotionWatch - AI CCTV Theft Detection Station
cd /d "%~dp0"
echo ========================================================
echo   MotionWatch - CCTV Theft Detection and Alert System
echo ========================================================
echo.
echo Activating Python virtual environment...
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)
echo Launching MotionWatch Surveillance Station...
python main.py
pause

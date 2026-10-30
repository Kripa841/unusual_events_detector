@echo off
title MotionWatch AI Surveillance Dashboard
cd /d "%~dp0"
echo ========================================================
echo  MotionWatch AI — Autonomous CCTV Theft Detection Station
echo  Launching Streamlit Web Dashboard...
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [OK] Using project virtual environment: .venv
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else if exist "..\.venv\Scripts\python.exe" (
    echo [OK] Using parent virtual environment: ..\.venv
    "..\.venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo [WARN] Virtual environment not found, attempting global streamlit...
    python -m streamlit run app.py
)

pause

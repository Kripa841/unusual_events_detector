@echo off
title MotionWatch AI Surveillance Dashboard
echo ========================================================
echo  MotionWatch AI — Autonomous CCTV Theft Detection Station
echo  Launching Streamlit Web Dashboard...
echo ========================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause


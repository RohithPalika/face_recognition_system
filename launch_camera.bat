@echo off
title Biometric AI Face Recognition System
cd /d "%~dp0"
echo ================================================================
echo           BIOMETRIC AI FACE RECOGNITION SYSTEM
echo ================================================================
echo Starting Live Camera HUD...
echo.
python main.py
if errorlevel 1 (
    echo.
    echo Application exited with an error. Press any key to close...
    pause >nul
)

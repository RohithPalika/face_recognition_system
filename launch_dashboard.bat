@echo off
title Biometric AI Web Dashboard
cd /d "%~dp0"
echo ================================================================
echo           BIOMETRIC AI WEB DASHBOARD
echo ================================================================
echo Launching Streamlit Web App in browser...
echo.
streamlit run app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error. Press any key to close...
    pause >nul
)

@echo off
rem Sets up the agent on first use, then starts it and opens it in the browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo First run: creating a Python environment. This takes a few minutes.
    py -3.14 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Python 3.14 is needed. Install it from https://www.python.org/downloads/ and run this again.
        pause
        exit /b 1
    )
)
.venv\Scripts\python -m pip install --quiet --disable-pip-version-check -r requirements.lock
if errorlevel 1 (
    echo Installing dependencies failed. Check your internet connection and run this again.
    pause
    exit /b 1
)
if not exist .env copy .env.example .env >nul
.venv\Scripts\python app.py %*

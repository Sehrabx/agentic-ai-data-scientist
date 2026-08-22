@echo off
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo Virtual environment found - starting app directly...
    call venv\Scripts\activate.bat
    streamlit run app.py
) else (
    echo No virtual environment found - setting up for the first time...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip >nul
    pip install -r requirements.txt
    streamlit run app.py
)

pause
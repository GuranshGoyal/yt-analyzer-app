@echo off
echo Starting YouTube Analyzer Application...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8+ and try again.
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed or not in PATH. Please install Node.js and try again.
    exit /b 1
)

REM Install Python dependencies if needed
echo Checking Python dependencies...
if not exist "%~dp0venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing Python dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Start the backend server in a new window
start cmd /k "title YouTube Analyzer Backend && echo Starting backend server... && python backend.py"

REM Wait a moment for the backend to start
timeout /t 3 /nobreak >nul

REM Start the frontend in a new window
start cmd /k "title YouTube Analyzer Frontend && echo Starting frontend... && npm run dev"

echo YouTube Analyzer started successfully!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
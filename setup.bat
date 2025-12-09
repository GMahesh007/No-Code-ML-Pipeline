@echo off
echo ============================================
echo  No-Code ML Pipeline Builder - Setup
echo ============================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo.

echo [2/3] Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo.

echo [3/3] Setup complete!
echo.
echo ============================================
echo  Ready to run!
echo ============================================
echo.
echo To start the application, run:
echo     python app.py
echo.
echo Then open your browser to:
echo     http://localhost:5000
echo.
pause

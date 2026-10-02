@echo off
set SCRIPT_DIR=%~dp0
echo ===================================================
echo Starting Virtual Steering Wheel...
echo ===================================================

if exist "%SCRIPT_DIR%python312-embed\python.exe" (
    echo Using embedded Python 3.12 environment...
    "%SCRIPT_DIR%python312-embed\python.exe" "%SCRIPT_DIR%virtual_steering_wheel.py"
) else (
    echo Searching for Python 3.12...
    py -3.12 "%SCRIPT_DIR%virtual_steering_wheel.py" 2>nul || python "%SCRIPT_DIR%virtual_steering_wheel.py"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while running the script.
    echo Please make sure dependencies are installed.
)
pause

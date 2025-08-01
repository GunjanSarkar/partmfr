@echo off
:: Script to run the API and test the bidirectional sliding window implementation
echo ===============================================================================
echo Motor Part API Test with Bidirectional Sliding Window
echo ===============================================================================

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not found in your PATH. Please install Python or add it to your PATH.
    exit /b 1
)

echo Checking for required packages...
python -c "import requests" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing requests package...
    python -m pip install requests
)

echo.
echo Starting the API server (press Ctrl+C to stop)...
echo.
start "Motor API Server" cmd /c "python start_api.py"

:: Wait for API to start
echo Waiting 5 seconds for API to start...
timeout /t 5 /nobreak >nul

:: Run the test script
echo.
echo Running test for part number DDE-A471014002...
python test_api_request.py DDE-A471014002

echo.
echo Test complete. The API server is still running in another window.
echo Press Ctrl+C in the API window to stop the server when you're done.
echo.
pause

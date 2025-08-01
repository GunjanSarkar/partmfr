@echo off
echo ===============================================================================
echo Unified Motor Parts API with Bidirectional Sliding Window
echo ===============================================================================

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not found in your PATH. Please install Python or add it to your PATH.
    exit /b 1
)

echo.
echo Starting the Unified API server...
echo.
echo API Documentation will be available at:
echo - Swagger UI: http://localhost:8000/docs
echo - ReDoc:      http://localhost:8000/redoc
echo - Unified API Endpoint: http://localhost:8000/api/unified
echo.
echo Main frontend available at:
echo - http://localhost:8000/
echo.
echo Press Ctrl+C to stop the server when finished
echo.

python start_api.py

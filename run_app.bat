@echo off
title DeepCSI Launcher
echo ============================================================
echo               DeepCSI Massive MIMO System Launcher
echo ============================================================

cd /d "%~dp0"

:: Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
)

:: 1. Launch FastAPI Backend in a dedicated window
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "DeepCSI - FastAPI Backend" cmd /k "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

:: Wait 2 seconds for backend to start up
timeout /t 2 /nobreak >nul

:: 2. Launch Streamlit Frontend in a dedicated window
echo [2/2] Starting Streamlit Dashboard on http://localhost:8501 ...
start "DeepCSI - Streamlit Dashboard" cmd /k "python -m streamlit run frontend/app.py"

echo ============================================================
echo [SUCCESS] Both services launched successfully!
echo   - Frontend: http://localhost:8501
echo   - Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo Keep the service terminal windows open while using the app.
echo ============================================================

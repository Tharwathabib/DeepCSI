# DeepCSI PowerShell Launcher
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          DeepCSI Massive MIMO System Launcher              " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$WorkspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $WorkspaceDir

# Detect and activate virtualenv if present
$venvPrompt = ""
if (Test-Path "$WorkspaceDir\.venv\Scripts\Activate.ps1") {
    $venvPrompt = "& '$WorkspaceDir\.venv\Scripts\Activate.ps1'; "
} elseif (Test-Path "$WorkspaceDir\venv\Scripts\Activate.ps1") {
    $venvPrompt = "& '$WorkspaceDir\venv\Scripts\Activate.ps1'; "
}

# 1. Start FastAPI Backend in a new window
Write-Host "[1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$WorkspaceDir'; $venvPrompt python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 2

# 2. Start Streamlit Frontend in a new window
Write-Host "[2/2] Launching Streamlit Dashboard on http://localhost:8501 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$WorkspaceDir'; $venvPrompt python -m streamlit run frontend/app.py"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Both services launched!" -ForegroundColor Green
Write-Host "  - Frontend: http://localhost:8501" -ForegroundColor White
Write-Host "  - Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green

#!/usr/bin/env pwsh
# Complete DiagnoGenie Startup Script
Write-Host "🏥 Starting DiagnoGenie Healthcare AI System..." -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    Set-Location "backend"
    pip install -r requirements.txt
    Set-Location ".."
}

Write-Host ""
Write-Host "🎯 Quick Start Guide:" -ForegroundColor Cyan
Write-Host "1. Run .\start-ollama.ps1 (in separate terminal)" -ForegroundColor White
Write-Host "2. Run .\start-backend.ps1 (in separate terminal)" -ForegroundColor White  
Write-Host "3. Run .\start-frontend.ps1 (in separate terminal)" -ForegroundColor White
Write-Host "4. Open http://localhost:5173 in browser" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Quick URLs:" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8001" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Note: Keep all terminals open while using the app!" -ForegroundColor Yellow
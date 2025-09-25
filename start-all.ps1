#!/usr/bin/env pwsh
# Complete DiagnoGenie Startup Script
param(
    [switch]$SkipDependencyCheck = $false
)

Write-Host "🏥 Starting DiagnoGenie Healthcare AI System..." -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

try {
    # Check if virtual environment exists
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
        & .venv\Scripts\Activate.ps1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to activate virtual environment"
        }
    } else {
        Write-Host "⚠️  Virtual environment not found. Creating one..." -ForegroundColor Yellow
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        & .venv\Scripts\Activate.ps1
        Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
        Set-Location "backend"
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python dependencies"
        }
        Set-Location ".."
    }

    # Check Node.js dependencies
    if (-not $SkipDependencyCheck) {
        Write-Host "🔍 Checking Node.js dependencies..." -ForegroundColor Yellow
        Set-Location "frontend"
        if (-not (Test-Path "node_modules")) {
            Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Yellow
            npm install
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install Node.js dependencies"
            }
        }
        Set-Location ".."
    }

    Write-Host ""
    Write-Host "🎯 DiagnoGenie is ready! Follow these steps:" -ForegroundColor Green
    Write-Host ""
    Write-Host "1. [OPTIONAL] Start enhanced AI chat:" -ForegroundColor Cyan
    Write-Host "   .\start-ollama.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "2. [REQUIRED] Start backend API server:" -ForegroundColor Cyan
    Write-Host "   .\start-backend.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "3. [REQUIRED] Start frontend application:" -ForegroundColor Cyan
    Write-Host "   .\start-frontend.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "4. Open DiagnoGenie in your browser:" -ForegroundColor Cyan
    Write-Host "   http://localhost:5173" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔗 Service URLs:" -ForegroundColor Cyan
    Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor Green
    Write-Host "  Backend:   http://localhost:8000" -ForegroundColor Green
    Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 Keep all terminal windows open while using DiagnoGenie!" -ForegroundColor Yellow
    Write-Host "💡 For best chat experience, run Ollama first" -ForegroundColor Yellow

} catch {
    Write-Host ""
    Write-Host "❌ Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running individual scripts or check the logs above" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🛠️  Troubleshooting:" -ForegroundColor Cyan
    Write-Host "  - Ensure Python 3.8+ is installed" -ForegroundColor White
    Write-Host "  - Ensure Node.js 16+ is installed" -ForegroundColor White
    Write-Host "  - Run as Administrator if permission errors occur" -ForegroundColor White
    exit 1
}
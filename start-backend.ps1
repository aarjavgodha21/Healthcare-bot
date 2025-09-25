#!/usr/bin/env pwsh
# Start Backend Server
Write-Host "🚀 Starting Backend API server..." -ForegroundColor Green

try {
    Set-Location "backend"
    & python -m uvicorn api:app --reload --port 8001
} catch {
    Write-Host "❌ Error starting backend: $_" -ForegroundColor Red
    Write-Host "💡 Make sure you've activated the virtual environment and installed dependencies." -ForegroundColor Yellow
}
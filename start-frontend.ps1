#!/usr/bin/env pwsh
# Start Frontend Development Server
Write-Host "🚀 Starting Frontend development server..." -ForegroundColor Green

try {
    Set-Location "frontend"
    npm run dev
} catch {
    Write-Host "❌ Error starting frontend: $_" -ForegroundColor Red
    Write-Host "💡 Make sure you've run 'npm install' in the frontend directory." -ForegroundColor Yellow
}
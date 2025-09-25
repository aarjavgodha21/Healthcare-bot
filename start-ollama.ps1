#!/usr/bin/env pwsh
# Start Ollama Server
Write-Host "🚀 Starting Ollama server..." -ForegroundColor Green

try {
    & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
} catch {
    Write-Host "❌ Error starting Ollama: $_" -ForegroundColor Red
    Write-Host "💡 Make sure Ollama is installed." -ForegroundColor Yellow
}
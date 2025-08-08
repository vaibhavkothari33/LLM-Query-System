@echo off
echo 🤖 LLM Query System - Starting Server
echo ========================================

echo 🛑 Killing existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo 🚀 Starting server on port 8000...
python main.py --server --port 8000

pause

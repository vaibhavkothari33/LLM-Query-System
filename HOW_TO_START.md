# 🚀 How to Start the LLM Query System

## Quick Start (Recommended)

**Just run ONE command:**

```bash
python start_server.py
```

This script will:
- ✅ Kill any existing Python processes (fixes port conflicts)
- ✅ Check all required files exist
- ✅ Build document index if needed
- ✅ Find an available port automatically
- ✅ Start the server
- ✅ Show you all the URLs to access

## What You'll See

After running the command, you'll see:
```
🤖 LLM Query System - Server Startup
========================================
✅ All required files found
✅ Killed existing Python processes
✅ Document index already exists
🔍 Found available port: 8000
🚀 Starting server on port 8000...
✅ Server started successfully on port 8000
📊 API Documentation: http://localhost:8000/docs
🌐 Frontend: http://localhost:8000/frontend.html
🔗 Health Check: http://localhost:8000/api/v1/health

==================================================
🎯 SERVER IS RUNNING!
==================================================

To stop the server, press Ctrl+C
```

## How to Use

1. **Open your browser** and go to: `http://localhost:8000/frontend.html`
2. **Type your question** in the text box (like "What is the GitHub ID of Vaibhav?")
3. **Click "Send Query"** to get your answer
4. **To stop the server**, press `Ctrl+C` in the terminal

## If Something Goes Wrong

If you get any errors:

1. **Close all terminal windows**
2. **Open a new terminal**
3. **Navigate to the project folder**
4. **Run again:** `python start_server.py`

## Files You Need

Make sure these files are in your folder:
- ✅ `main.py` (main application)
- ✅ `config.py` (configuration)
- ✅ `requirements.txt` (dependencies)
- ✅ `start_server.py` (this startup script)
- ✅ `frontend.html` (web interface)
- ✅ `documents/` folder (your resume and other documents)

## That's It!

No other commands needed. Just run `python start_server.py` and you're ready to go!

# 🚀 Quick Start Guide - LLM Query System

## 📋 Script Execution Order

### Option 1: Automated Startup (Recommended)
```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Run the automated startup script
python start_system.py
```

This will automatically:
- ✅ Check dependencies
- ✅ Verify documents
- ✅ Build index
- ✅ Start server
- ✅ Run tests
- ✅ Open frontend

### Option 2: Manual Startup

#### Step 1: Environment Setup
```bash
# Navigate to project directory
cd "D:\Desktop\Code Folder\Artifical Intelligense\Bajaj"

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

#### Step 2: Prepare Documents
```bash
# Check if documents exist
dir documents

# If no documents, add PDF/DOCX files to ./documents folder
# Then build index:
python main.py --documents ./documents --rebuild
```

#### Step 3: Start Server
```bash
# Start the API server
python main.py --server --port 8001
```

#### Step 4: Test System
```bash
# In a new terminal, test the API
python test_api.py
```

#### Step 5: Open Frontend
```bash
# Open frontend.html in your browser
# Or run:
python -c "import webbrowser; webbrowser.open('file:///path/to/frontend.html')"
```

## 🌐 Available URLs

Once the server is running:

- **Frontend Interface**: `file:///path/to/frontend.html`
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/v1/health
- **System Stats**: http://localhost:8001/api/v1/stats

## 🔧 Common Commands

### Server Management
```bash
# Start server on port 8001
python main.py --server --port 8001

# Start server on different port
python main.py --server --port 8002

# Stop server: Press Ctrl+C in the terminal
```

### Index Management
```bash
# Build new index from documents
python main.py --documents ./documents --rebuild

# Process single query via CLI
python main.py --documents ./documents --query "What are the key skills?"
```

### Testing
```bash
# Run all API tests
python test_api.py

# Test webhook specifically
python test_webhook.py

# Run setup validation
python setup.py
```

## 📁 File Structure
```
Bajaj/
├── main.py                 # Main application
├── start_system.py         # Automated startup script
├── frontend.html           # Web interface
├── test_api.py            # API testing
├── test_webhook.py        # Webhook testing
├── setup.py               # Setup validation
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── QUICK_START.md        # This file
├── documents/            # Your PDF/DOCX files
├── data/                 # Generated index files
└── venv/                 # Virtual environment
```

## 🎯 Testing Your System

### 1. Using the Frontend
1. Open `frontend.html` in your browser
2. Check system status (should show green indicators)
3. Enter a query like "What are the key skills mentioned?"
4. Click "Submit Query"
5. Review the response with confidence score and reasoning

### 2. Using API Documentation
1. Go to http://localhost:8001/docs
2. Click "Authorize" and enter your API key
3. Try the `/api/v1/query` endpoint
4. Test with different queries

### 3. Using CLI
```bash
# Interactive mode
python main.py --documents ./documents

# Single query
python main.py --documents ./documents --query "What is the educational background?"
```

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :8001

# Kill the process
taskkill /PID <process_id> /F

# Or use different port
python main.py --server --port 8002
```

### Documents Not Found
```bash
# Check documents directory
dir documents

# Rebuild index
python main.py --documents ./documents --rebuild
```

### Dependencies Missing
```bash
# Install missing packages
pip install fastapi uvicorn sentence-transformers faiss-cpu
```

## 🎉 Success Indicators

When everything is working correctly, you should see:

1. **Server logs**: "INFO: Uvicorn running on http://0.0.0.0:8001"
2. **Frontend status**: All green indicators
3. **API tests**: All tests passing
4. **Query responses**: Meaningful answers with confidence scores

## 📞 Getting Help

1. Check the logs for error messages
2. Run `python test_api.py` to diagnose issues
3. Verify all dependencies are installed
4. Ensure documents are in the correct format (PDF/DOCX)

---

**Happy Querying! 🎉**

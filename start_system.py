#!/usr/bin/env python3
"""
Startup script for LLM Query System
Runs all components in the correct order
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def check_dependencies():
    """Check if all required packages are installed"""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        'fastapi', 'uvicorn', 'sentence_transformers', 'faiss', 
        'scikit-learn', 'pandas', 'numpy', 'PyPDF2', 'docx', 
        'beautifulsoup4', 'transformers', 'openai', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_documents():
    """Check if documents directory exists and has files"""
    print_step(2, "Checking Documents")
    
    documents_dir = Path("./documents")
    if not documents_dir.exists():
        print("❌ Documents directory not found!")
        print("Creating documents directory...")
        documents_dir.mkdir(exist_ok=True)
        print("✅ Created documents directory")
        print("⚠️  Please add PDF/DOCX files to the ./documents folder")
        return False
    
    files = list(documents_dir.glob("*"))
    if not files:
        print("❌ No documents found in ./documents directory")
        print("⚠️  Please add PDF/DOCX files to the ./documents folder")
        return False
    
    print(f"✅ Found {len(files)} files in documents directory:")
    for file in files:
        print(f"   📄 {file.name}")
    
    return True

def build_index():
    """Build the document index"""
    print_step(3, "Building Document Index")
    
    try:
        # Check if index already exists
        if Path("./data/faiss_index.bin").exists() and Path("./data/documents.pkl").exists():
            print("✅ Index already exists, skipping build")
            return True
        
        print("🔨 Building new index...")
        result = subprocess.run([
            sys.executable, "main.py", 
            "--documents", "./documents", 
            "--rebuild"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Index built successfully!")
            return True
        else:
            print(f"❌ Failed to build index: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error building index: {str(e)}")
        return False

def start_server():
    """Start the API server"""
    print_step(4, "Starting API Server")
    
    # Check if port 8001 is available
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8001))
        sock.close()
        
        if result == 0:
            print("⚠️  Port 8001 is already in use")
            print("   The server might already be running")
            return True
    except:
        pass
    
    print("🚀 Starting server on port 8001...")
    print("   This will take a moment to load the models...")
    
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, "main.py", 
            "--server", "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully!")
            print(f"   Process ID: {process.pid}")
            return True, process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start: {stderr.decode()}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        return False, None

def test_system():
    """Test the system"""
    print_step(5, "Testing System")
    
    try:
        result = subprocess.run([
            sys.executable, "test_api.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests passed!")
            return True
        else:
            print(f"❌ System tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        return False

def open_frontend():
    """Open the frontend in browser"""
    print_step(6, "Opening Frontend")
    
    frontend_path = Path("./frontend.html")
    if not frontend_path.exists():
        print("❌ Frontend file not found!")
        return False
    
    try:
        # Convert to absolute path
        abs_path = frontend_path.absolute()
        url = f"file:///{abs_path.as_posix()}"
        
        print(f"🌐 Opening frontend: {url}")
        webbrowser.open(url)
        
        print("✅ Frontend opened in browser!")
        print("\n📖 Available URLs:")
        print("   Frontend: file:///path/to/frontend.html")
        print("   API Docs: http://localhost:8001/docs")
        print("   Health Check: http://localhost:8001/api/v1/health")
        
        return True
        
    except Exception as e:
        print(f"❌ Error opening frontend: {str(e)}")
        return False

def main():
    """Main startup function"""
    print("🚀 LLM Query System Startup")
    print("This script will start all components in the correct order")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return 1
    
    # Step 2: Check documents
    if not check_documents():
        print("\n⚠️  No documents found. You can still start the server, but queries won't work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Step 3: Build index (if documents exist)
    if Path("./documents").exists() and any(Path("./documents").iterdir()):
        if not build_index():
            print("\n❌ Index build failed.")
            return 1
    
    # Step 4: Start server
    server_success, server_process = start_server()
    if not server_success:
        print("\n❌ Server startup failed.")
        return 1
    
    # Step 5: Test system
    time.sleep(3)  # Wait for server to fully start
    if not test_system():
        print("\n⚠️  System tests failed, but server is running.")
    
    # Step 6: Open frontend
    open_frontend()
    
    print("\n🎉 System startup complete!")
    print("\n📋 Next steps:")
    print("   1. Use the frontend to test queries")
    print("   2. Visit http://localhost:8001/docs for API documentation")
    print("   3. Add more documents to ./documents/ and rebuild index")
    print("   4. Press Ctrl+C to stop the server")
    
    try:
        # Keep the script running
        if server_process:
            print(f"\n🔄 Server is running (PID: {server_process.pid})")
            print("Press Ctrl+C to stop...")
            server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if server_process:
            server_process.terminate()
        print("✅ Server stopped")
    
    return 0

if __name__ == "__main__":
    exit(main())

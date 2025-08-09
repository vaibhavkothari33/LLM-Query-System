#!/usr/bin/env python3
"""
Robust Server Startup Script
Automatically finds available ports and starts the FastAPI server
"""

import subprocess
import time
import socket
import sys
import os
from pathlib import Path

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def kill_python_processes():
    """Kill all Python processes to clear port conflicts"""
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, check=False)
        print("✅ Killed existing Python processes")
        time.sleep(2)  # Wait for processes to fully terminate
    except Exception as e:
        print(f"⚠️  Could not kill processes: {e}")

def check_dependencies():
    """Check if required files exist"""
    required_files = ['main.py', 'config.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def build_index_if_needed():
    """Build the document index if it doesn't exist"""
    data_dir = Path('./data')
    if not data_dir.exists() or not list(data_dir.glob('*.faiss')):
        print("📚 Building document index...")
        documents_dir = Path('./documents')
        if documents_dir.exists():
            try:
                result = subprocess.run([
                    sys.executable, 'main.py', 
                    '--documents', './documents', 
                    '--rebuild'
                ], capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Document index built successfully")
                else:
                    print(f"⚠️  Index build had issues: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("⚠️  Index build timed out")
        else:
            print("⚠️  No documents folder found - skipping index build")
    else:
        print("✅ Document index already exists")

def start_server(port):
    """Start the FastAPI server on the specified port"""
    print(f"🚀 Starting server on port {port}...")
    
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, 'main.py', 
            '--server', 
            '--port', str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Server started successfully on port {port}")
            print(f"📊 API Documentation: http://localhost:{port}/docs")
            print(f"🌐 Frontend: http://localhost:{port}/frontend.html")
            print(f"🔗 Health Check: http://localhost:{port}/api/v1/health")
            print("\n" + "="*50)
            print("🎯 SERVER IS RUNNING!")
            print("="*50)
            print("\nTo stop the server, press Ctrl+C")
            
            # Keep the script running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                print("✅ Server stopped")
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🤖 LLM Query System - Server Startup")
    print("="*40)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Kill existing processes
    kill_python_processes()
    
    # Step 3: Build index if needed
    build_index_if_needed()
    
    # Step 4: Find available port
    port = find_available_port(8000)
    if port is None:
        print("❌ No available ports found")
        sys.exit(1)
    
    print(f"🔍 Found available port: {port}")
    
    # Step 5: Start server
    if not start_server(port):
        sys.exit(1)

if __name__ == "__main__":
    main()

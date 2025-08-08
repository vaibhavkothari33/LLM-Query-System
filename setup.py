#!/usr/bin/env python3
"""
Setup script for LLM-Powered Intelligent Query-Retrieval System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'sentence_transformers', 'faiss', 
        'scikit-learn', 'pandas', 'numpy', 'PyPDF2', 'docx', 
        'beautifulsoup4', 'transformers', 'openai', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'documents', 'reports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_system():
    """Test the system with a sample query"""
    try:
        from main import IntelligentQueryRetrievalSystem, Config
        
        print("\n🧪 Testing system initialization...")
        
        # Initialize system
        config = Config()
        system = IntelligentQueryRetrievalSystem(config)
        
        # Check if documents exist
        documents_dir = "./documents"
        if os.path.exists(documents_dir) and any(Path(documents_dir).iterdir()):
            print("✅ Documents found in ./documents")
            
            # Initialize system (this will build the index)
            print("🔨 Building document index (this may take a few minutes)...")
            system.initialize(documents_dir, rebuild_index=True)
            
            # Test query
            print("🔍 Testing sample query...")
            test_query = "What are the key skills and experience mentioned?"
            result = system.process_query(test_query)
            
            print(f"✅ Query processed successfully!")
            print(f"📝 Answer: {result.answer[:100]}...")
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"⏱️  Processing time: {result.processing_time:.3f}s")
            
        else:
            print("⚠️  No documents found in ./documents")
            print("Please add some PDF or DOCX files to test the system")
            
    except Exception as e:
        print(f"❌ System test failed: {str(e)}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up LLM-Powered Intelligent Query-Retrieval System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create directories
    create_directories()
    
    # Test system
    if test_system():
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Add your documents to the ./documents folder")
        print("2. Run: python main.py --server --port 8000")
        print("3. Test API at: http://localhost:8000/docs")
        print("4. Run evaluation: python evaluation_framework.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

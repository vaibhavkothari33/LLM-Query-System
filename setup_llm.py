#!/usr/bin/env python3
"""
Setup script for LLM integration
Helps configure OpenAI API key and test the system
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with configuration"""
    env_content = """# LLM Query System Configuration
# OpenAI Configuration (Required for real LLM responses)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# System Configuration
SYSTEM_API_KEY=36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8001

# Document Processing
DOCUMENTS_DIR=./documents
DATA_DIR=./data

# LLM Settings
MAX_CONTEXT_LENGTH=4000
MAX_RESPONSE_TOKENS=500
TEMPERATURE=0.1

# Vector Search Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DIMENSION=384
TOP_K_DOCUMENTS=5
SIMILARITY_THRESHOLD=0.3
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 Please edit .env file and add your OpenAI API key")

def setup_openai():
    """Setup OpenAI API key"""
    print("\n🔑 OpenAI API Key Setup")
    print("=" * 50)
    print("To get real LLM responses, you need an OpenAI API key:")
    print("1. Go to: https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Add it to your .env file")
    print()
    
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Update .env file
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, 'r') as f:
                content = f.read()
            
            content = content.replace("OPENAI_API_KEY=your_openai_api_key_here", f"OPENAI_API_KEY={api_key}")
            
            with open(env_path, 'w') as f:
                f.write(content)
            
            print("✅ OpenAI API key configured!")
        else:
            print("❌ .env file not found. Run setup first.")
    else:
        print("⚠️  Skipping OpenAI setup. System will use local models.")

def test_llm_integration():
    """Test LLM integration"""
    print("\n🧪 Testing LLM Integration")
    print("=" * 50)
    
    try:
        # Import and test configuration
        from config import get_config
        config = get_config()
        
        print("📋 Current Configuration:")
        for key, value in config.items():
            if "key" in key.lower() and value and value != "your_openai_api_key_here":
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: {value}")
        
        # Test OpenAI if configured
        if config["openai_api_key"] and config["openai_api_key"] != "your_openai_api_key_here":
            print("\n🔍 Testing OpenAI connection...")
            try:
                import openai
                openai.api_key = config["openai_api_key"]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello! This is a test."}],
                    max_tokens=50
                )
                
                print("✅ OpenAI connection successful!")
                print(f"   Response: {response.choices[0].message.content}")
                
            except Exception as e:
                print(f"❌ OpenAI test failed: {str(e)}")
        else:
            print("\n⚠️  OpenAI not configured. System will use local models.")
        
        # Test local models
        print("\n🔍 Testing local model availability...")
        try:
            from transformers import pipeline
            print("✅ Local models available!")
        except ImportError:
            print("❌ Local models not available. Install with: pip install transformers torch")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")

def main():
    """Main setup function"""
    print("🚀 LLM Query System Setup")
    print("This script will help you configure the LLM integration")
    
    while True:
        print("\n📋 Setup Options:")
        print("1. Create .env file")
        print("2. Setup OpenAI API key")
        print("3. Test LLM integration")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            create_env_file()
        elif choice == "2":
            setup_openai()
        elif choice == "3":
            test_llm_integration()
        elif choice == "4":
            print("👋 Setup complete!")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()

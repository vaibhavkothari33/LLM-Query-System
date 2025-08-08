"""
Configuration file for LLM Query System
Set your API keys and settings here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDvqPYipnjb5jAozUqdmcboOrNqSKSZUWE")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# System Configuration
SYSTEM_API_KEY = os.getenv("SYSTEM_API_KEY", "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8001"))

# Document Processing
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
DATA_DIR = os.getenv("DATA_DIR", "./data")

# LLM Settings
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Vector Search Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))
TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

def get_config():
    """Get configuration dictionary"""
    return {
        "gemini_api_key": GEMINI_API_KEY,
        "gemini_model": GEMINI_MODEL,
        "system_api_key": SYSTEM_API_KEY,
        "server_host": SERVER_HOST,
        "server_port": SERVER_PORT,
        "documents_dir": DOCUMENTS_DIR,
        "data_dir": DATA_DIR,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
        "embedding_model": EMBEDDING_MODEL,
        "vector_dimension": VECTOR_DIMENSION,
        "top_k_documents": TOP_K_DOCUMENTS,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("🔧 Current Configuration:")
    print("=" * 50)
    for key, value in config.items():
        if "key" in key.lower() and value:
            # Mask API keys for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"{key}: {masked_value}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
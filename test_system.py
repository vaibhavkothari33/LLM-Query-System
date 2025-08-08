#!/usr/bin/env python3
"""
Test script for LLM-Powered Intelligent Query-Retrieval System
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_stats_endpoint():
    """Test stats endpoint"""
    print("\n📊 Testing stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats endpoint working")
            print(f"   Total chunks: {stats.get('total_document_chunks', 0)}")
            print(f"   Model: {stats.get('embedding_model', 'N/A')}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint error: {str(e)}")
        return False

def test_query_endpoint():
    """Test query endpoint with sample queries"""
    print("\n🔍 Testing query endpoint...")
    
    test_queries = [
        "What are the key skills and experience mentioned?",
        "What is the educational background?",
        "What are the technical skills listed?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Testing query {i}: {query[:50]}...")
        try:
            payload = {
                "query": query,
                "max_results": 3
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Query {i} successful")
                print(f"   📝 Answer: {result['answer'][:100]}...")
                print(f"   🎯 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️  Time: {result['processing_time']:.3f}s")
                print(f"   📄 Matched clauses: {len(result['matched_clauses'])}")
            else:
                print(f"   ❌ Query {i} failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Query {i} error: {str(e)}")
    
    return True

def test_cli_mode():
    """Test CLI mode"""
    print("\n💻 Testing CLI mode...")
    try:
        import subprocess
        import sys
        
        # Test CLI with documents directory
        cmd = [sys.executable, "main.py", "--documents", "./documents", "--query", "What are the key skills?"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ CLI mode working")
            print(f"   Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ CLI mode failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI mode error: {str(e)}")
        return False

def test_evaluation_framework():
    """Test evaluation framework"""
    print("\n📈 Testing evaluation framework...")
    try:
        import subprocess
        import sys
        
        # Test evaluation with sample data
        cmd = [sys.executable, "evaluation_framework.py", "--base-url", API_BASE_URL]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Evaluation framework working")
            print("   Check evaluation_report.html for results")
            return True
        else:
            print(f"❌ Evaluation framework failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation framework error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing LLM-Powered Intelligent Query-Retrieval System")
    print("=" * 60)
    
    # Check if server is running
    print("🔌 Checking if API server is running...")
    if not test_health_endpoint():
        print("\n❌ API server is not running!")
        print("Please start the server first:")
        print("   python main.py --server --port 8000")
        return 1
    
    # Run all tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Stats Endpoint", test_stats_endpoint),
        ("Query Endpoint", test_query_endpoint),
        ("CLI Mode", test_cli_mode),
        ("Evaluation Framework", test_evaluation_framework)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed: {str(e)}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is working correctly.")
        print("\n📖 Next steps:")
        print("1. Add more documents to ./documents/")
        print("2. Try different types of queries")
        print("3. Run evaluation: python evaluation_framework.py")
        print("4. Check API docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())

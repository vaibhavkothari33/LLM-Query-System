#!/usr/bin/env python3
"""
Simple API test script for the LLM Query System
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8001"  # Changed to port 8001
API_KEY = "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health")
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

def test_stats():
    """Test stats endpoint"""
    print("📊 Testing stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", headers=HEADERS)
        if response.status_code == 200:
            print("✅ Stats check passed")
            stats = response.json()
            print(f"   Total chunks: {stats.get('total_document_chunks', 0)}")
            print(f"   Model: {stats.get('embedding_model', 'N/A')}")
            return True
        else:
            print(f"❌ Stats check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats check error: {str(e)}")
        return False

def test_query():
    """Test query endpoint"""
    print("🔍 Testing query endpoint...")
    try:
        query_data = {
            "query": "What are the key skills and experience mentioned?",
            "max_results": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query", 
            headers=HEADERS,
            json=query_data
        )
        
        if response.status_code == 200:
            print("✅ Query test passed")
            result = response.json()
            print(f"   Answer: {result.get('answer', 'N/A')[:100]}...")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
            return True
        else:
            print(f"❌ Query test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query test error: {str(e)}")
        return False

def test_webhook():
    """Test webhook endpoint"""
    print("🔗 Testing webhook endpoint...")
    try:
        # Test query processed webhook
        webhook_data = {
            "event_type": "query_processed",
            "query_id": "q_12345",
            "query": "What are the key skills?",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processing_time": 0.245,
                "confidence": 0.85,
                "chunks_retrieved": 3,
                "answer_length": 150
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhook",
            headers=HEADERS,
            json=webhook_data
        )
        
        if response.status_code == 200:
            print("✅ Webhook test passed")
            result = response.json()
            print(f"   Webhook ID: {result.get('webhook_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ Webhook test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Webhook test error: {str(e)}")
        return False

def test_custom_queries():
    """Test various custom queries"""
    print("🎯 Testing custom queries...")
    
    test_queries = [
        {
            "query": "What is the educational background mentioned?",
            "max_results": 3
        },
        {
            "query": "What are the technical skills listed?",
            "max_results": 5
        },
        {
            "query": "What is the work experience?",
            "document_types": ["pdf"],
            "max_results": 3
        }
    ]
    
    passed = 0
    total = len(test_queries)
    
    for i, query_data in enumerate(test_queries, 1):
        try:
            print(f"   Testing query {i}: {query_data['query'][:50]}...")
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                headers=HEADERS,
                json=query_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Query {i} successful")
                print(f"   📝 Answer: {result['answer'][:80]}...")
                print(f"   🎯 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️  Time: {result['processing_time']:.3f}s")
                passed += 1
            else:
                print(f"   ❌ Query {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Query {i} error: {str(e)}")
    
    print(f"   📊 Custom queries: {passed}/{total} passed")
    return passed == total

def main():
    """Run all tests"""
    print("🚀 Testing LLM Query System API...")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Run tests
    health_ok = test_health()
    stats_ok = test_stats()
    query_ok = test_query()
    webhook_ok = test_webhook()
    custom_queries_ok = test_custom_queries()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   Health Check:     {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Stats Check:      {'✅ PASS' if stats_ok else '❌ FAIL'}")
    print(f"   Query Check:      {'✅ PASS' if query_ok else '❌ FAIL'}")
    print(f"   Webhook Check:    {'✅ PASS' if webhook_ok else '❌ FAIL'}")
    print(f"   Custom Queries:   {'✅ PASS' if custom_queries_ok else '❌ FAIL'}")
    
    all_passed = all([health_ok, stats_ok, query_ok, webhook_ok, custom_queries_ok])
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is working perfectly!")
        print("\n📖 Next steps:")
        print("   1. Open http://localhost:8001/docs in your browser")
        print("   2. Try the interactive API documentation")
        print("   3. Add more documents to the ./documents folder")
        print("   4. Test with your own queries!")
        print("   5. Set up webhook integrations for notifications")
    else:
        print("\n⚠️  Some tests failed. Check the server logs for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

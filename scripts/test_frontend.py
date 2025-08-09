#!/usr/bin/env python3
"""
Test script to verify frontend is working correctly
"""

import requests
import json

def test_frontend_api():
    """Test the API that the frontend uses"""
    print("🧪 Testing Frontend API Integration")
    print("=" * 50)
    
    API_BASE_URL = 'http://localhost:8002'
    API_KEY = '36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9'
    
    # Test the exact query that was failing
    test_query = "What is the GitHub ID of Vaibhav?"
    
    try:
        print(f"🔍 Testing query: {test_query}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'query': test_query,
                'max_results': 5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query successful!")
            print(f"📝 Answer: {data['answer']}")
            print(f"🎯 Confidence: {data['confidence']:.3f}")
            print(f"⏱️  Processing Time: {data['processing_time']:.3f}s")
            print(f"🔢 Token Usage: {data['token_usage']['total_tokens']}")
            print(f"📄 Matched Clauses: {len(data['matched_clauses'])}")
            
            # Show first matched clause
            if data['matched_clauses']:
                clause = data['matched_clauses'][0]
                print(f"📋 Top Match: {clause['document_source']} (Confidence: {clause['confidence']:.3f})")
                print(f"📄 Content: {clause['content'][:100]}...")
            
            return True
        else:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n🏥 Testing Health Endpoint")
    print("-" * 30)
    
    try:
        response = requests.get('http://localhost:8002/api/v1/health')
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Frontend Integration Test")
    print("This script tests the API endpoints used by the frontend")
    
    # Test health first
    if not test_health_endpoint():
        print("❌ Server is not running. Please start the server first.")
        return
    
    # Test the main query
    success = test_frontend_api()
    
    if success:
        print("\n🎉 Frontend API test passed!")
        print("📖 Next steps:")
        print("   1. Open frontend.html in your browser")
        print("   2. Try the query: 'What is the GitHub ID of Vaibhav?'")
        print("   3. You should see the correct answer: vaibhavkothari33")
    else:
        print("\n❌ Frontend API test failed!")
        print("   Please check the server logs for errors.")

if __name__ == "__main__":
    main()

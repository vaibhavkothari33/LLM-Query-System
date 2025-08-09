#!/usr/bin/env python3
"""
Test webhook endpoint
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8001"
API_KEY = "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_webhook():
    """Test webhook endpoint"""
    print("🔗 Testing webhook endpoint...")
    
    # Test data
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
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/webhook",
            headers=HEADERS,
            json=webhook_data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Webhook test passed")
            print(f"   Webhook ID: {result.get('webhook_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ Webhook test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook test error: {str(e)}")
        return False

if __name__ == "__main__":
    test_webhook()

import asyncio
import json
import requests
import time
from typing import List, Dict

class QueryTester:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_query(self, query: str, expected_keywords: List[str] = None) -> Dict:
        """Test a single query and validate response"""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            headers=self.headers,
            json={"query": query}
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code != 200:
            return {
                "query": query,
                "success": False,
                "error": response.text,
                "processing_time": processing_time
            }
        
        result = response.json()
        
        # Validate expected keywords if provided
        keyword_match = True
        if expected_keywords:
            answer_text = result["answer"].lower()
            keyword_match = any(keyword.lower() in answer_text for keyword in expected_keywords)
        
        return {
            "query": query,
            "success": True,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "processing_time": processing_time,
            "system_processing_time": result["processing_time"],
            "matched_clauses": len(result["matched_clauses"]),
            "keyword_match": keyword_match,
            "token_usage": result["token_usage"]
        }

# Sample test queries for different domains
SAMPLE_QUERIES = [
    # Insurance queries
    {
        "query": "Does this policy cover knee surgery, and what are the conditions?",
        "domain": "insurance",
        "expected_keywords": ["surgery", "knee", "coverage", "conditions"]
    },
    {
        "query": "What is the deductible for emergency room visits?",
        "domain": "insurance",
        "expected_keywords": ["deductible", "emergency", "room"]
    },
    
    # Legal queries
    {
        "query": "What are the termination conditions in the employment contract?",
        "domain": "legal",
        "expected_keywords": ["termination", "conditions", "employment"]
    },
    {
        "query": "What confidentiality obligations does the NDA impose?",
        "domain": "legal",
        "expected_keywords": ["confidentiality", "obligations", "NDA"]
    },
    
    # HR queries
    {
        "query": "How many vacation days are employees entitled to?",
        "domain": "hr",
        "expected_keywords": ["vacation", "days", "entitled"]
    },
    {
        "query": "What is the process for requesting remote work?",
        "domain": "hr",
        "expected_keywords": ["remote", "work", "process"]
    },
    
    # Compliance queries
    {
        "query": "What are the GDPR data retention requirements?",
        "domain": "compliance",
        "expected_keywords": ["GDPR", "retention", "requirements"]
    },
    {
        "query": "What documentation is required for SOX compliance?",
        "domain": "compliance",
        "expected_keywords": ["documentation", "SOX", "compliance"]
    }
]

def run_test_suite():
    """Run complete test suite"""
    tester = QueryTester(api_key="36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9")
    
    results = []
    total_time = 0
    successful_queries = 0
    
    print("🧪 Running Query Test Suite...")
    print("=" * 60)
    
    for i, test_case in enumerate(SAMPLE_QUERIES, 1):
        print(f"Test {i}/{len(SAMPLE_QUERIES)}: {test_case['domain'].upper()}")
        print(f"Query: {test_case['query']}")
        
        result = tester.test_query(
            test_case["query"], 
            test_case.get("expected_keywords")
        )
        
        if result["success"]:
            successful_queries += 1
            print(f"✅ Success (Confidence: {result['confidence']:.3f})")
            print(f"Answer: {result['answer'][:100]}...")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print(f"Processing Time: {result['processing_time']:.3f}s")
        print("-" * 40)
        
        results.append(result)
        total_time += result["processing_time"]
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Successful Queries: {successful_queries}/{len(SAMPLE_QUERIES)}")
    print(f"Success Rate: {successful_queries/len(SAMPLE_QUERIES)*100:.1f}%")
    print(f"Average Response Time: {total_time/len(SAMPLE_QUERIES):.3f}s")
    print(f"Total Time: {total_time:.3f}s")
    
    return results

if __name__ == "__main__":
    results = run_test_suite()
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Results saved to test_results.json")
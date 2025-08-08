#!/usr/bin/env python3
"""
Test script for LLM integration
Tests queries against your resume documents
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_resume_queries():
    """Test specific resume-related queries"""
    print("🧪 Testing Resume Queries")
    print("=" * 50)
    
    # Test queries related to your resume
    test_queries = [
        "What is the GitHub ID of Vaibhav?",
        "What are the key skills mentioned in the resume?",
        "What is the educational background?",
        "What is the work experience?",
        "What programming languages are mentioned?",
        "What projects are listed in the resume?"
    ]
    
    try:
        # Import the system
        from main import IntelligentQueryRetrievalSystem, Config
        
        # Initialize system
        config = Config()
        system = IntelligentQueryRetrievalSystem(config)
        
        # Check if documents exist
        documents_dir = Path("./documents")
        if not documents_dir.exists() or not any(documents_dir.iterdir()):
            print("❌ No documents found in ./documents directory")
            print("   Please add your resume (PDF/DOCX) to the documents folder")
            return
        
        # Initialize system
        print("🔧 Initializing system...")
        system.initialize("./documents", rebuild_index=False)
        print("✅ System initialized")
        
        # Test each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 40)
            
            try:
                result = system.process_query(query)
                
                print(f"✅ Query processed successfully")
                print(f"   Answer: {result.answer[:200]}...")
                print(f"   Confidence: {result.confidence:.3f}")
                print(f"   Processing Time: {result.processing_time:.3f}s")
                print(f"   Matched Clauses: {len(result.matched_clauses)}")
                
                # Show first matched clause
                if result.matched_clauses:
                    clause = result.matched_clauses[0]
                    print(f"   Top Match: {clause.document_source} (Confidence: {clause.confidence:.3f})")
                
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
        print(f"\n🎉 Testing complete! Processed {len(test_queries)} queries.")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        print("   Make sure all dependencies are installed and documents are available")

def test_llm_providers():
    """Test different LLM providers"""
    print("\n🔍 Testing LLM Providers")
    print("=" * 50)
    
    try:
        from main import LLMQueryProcessor, Config
        
        config = Config()
        llm_processor = LLMQueryProcessor(config)
        
        # Test prompt
        test_prompt = """
You are an expert document analyst. 
Query: What is the GitHub ID of Vaibhav?
Relevant Document Context: The resume shows GitHub: vaibhavkothari123
Instructions: Answer based on the context provided.

Format your response as:
ANSWER: [Your answer]
REASONING: [Your reasoning]
CONFIDENCE: [0.0 to 1.0]
"""
        
        print("🔧 Testing LLM response generation...")
        response = llm_processor._call_llm(test_prompt)
        
        print("✅ LLM response generated:")
        print("-" * 30)
        print(response)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ LLM test failed: {str(e)}")

def main():
    """Main test function"""
    print("🚀 LLM Integration Test")
    print("This script tests the LLM integration with your resume")
    
    # Test LLM providers first
    test_llm_providers()
    
    # Test resume queries
    test_resume_queries()
    
    print("\n📋 Next Steps:")
    print("1. If tests pass, your system is working correctly")
    print("2. If you want better responses, add your OpenAI API key to .env file")
    print("3. Run: python setup_llm.py to configure OpenAI")
    print("4. Test with the frontend: open frontend.html in your browser")

if __name__ == "__main__":
    main()

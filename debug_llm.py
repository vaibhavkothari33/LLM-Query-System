#!/usr/bin/env python3
"""
Debug script for LLM processor
"""

from main import LLMQueryProcessor, Config

def test_llm_processor():
    """Test LLM processor directly"""
    print("🧪 Testing LLM Processor")
    print("=" * 40)
    
    # Create config
    config = Config()
    config.gemini_api_key = "AIzaSyDvqPYipnjb5jAozUqdmcboOrNqSKSZUWE"
    
    print(f"🔑 Gemini API Key: {config.gemini_api_key[:10]}...")
    
    # Create LLM processor
    llm_processor = LLMQueryProcessor(config)
    
    # Test prompt
    test_prompt = """
You are an expert document analyst. 
Query: What is the GitHub ID of Vaibhav?
Relevant Document Context: The resume shows GitHub: github.com/vaibhavkothari33
Instructions: Answer based on the context provided.

Format your response as:
ANSWER: [Your answer]
REASONING: [Your reasoning]
CONFIDENCE: [0.0 to 1.0]
"""
    
    print("🔍 Testing LLM call...")
    
    try:
        response = llm_processor._call_llm(test_prompt)
        print("✅ LLM Response:")
        print("-" * 30)
        print(response)
        print("-" * 30)
        return True
    except Exception as e:
        print(f"❌ LLM call failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_llm_processor()

#!/usr/bin/env python3
"""
Test script for Gemini API
"""

import google.generativeai as genai

def test_gemini():
    """Test Gemini API directly"""
    print("🧪 Testing Gemini API")
    print("=" * 30)
    
    # Your API key
    api_key = ""
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
        
        print("🔍 Testing Gemini response...")
        response = model.generate_content(test_prompt)
        
        print("✅ Gemini response:")
        print("-" * 30)
        print(response.text)
        print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini()

#!/usr/bin/env python3
"""Test script to verify context handling with Ollama"""

import requests
import sys

def test_context(model="llama2"):
    # 1. Upload a test file
    test_content = """Sample Product Information:
Product: AI Assistant Pro
Price: $99/month
Features: 
- Natural language processing
- Multi-language support
- API access
- 24/7 availability"""
    
    # 2. Test with direct prompt injection
    prompt = f"""Context:
{test_content}

Question: What is the price of AI Assistant Pro?

Answer based on the context provided above:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    print(f"Model: {model}")
    print(f"Response: {response.json().get('response', 'No response')}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "llama2"
    test_context(model)
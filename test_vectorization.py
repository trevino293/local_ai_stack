#!/usr/bin/env python3
"""
Debug script to test vectorization pipeline
Save as: test_vectorization.py
Run with: python test_vectorization.py
"""

import requests
import json
import time

def test_service_health():
    """Test all service health endpoints"""
    services = [
        ("Ollama", "http://localhost:11434/api/tags"),
        ("Embedding Service", "http://localhost:8080/health"),
        ("MCP Filesystem", "http://localhost:3000/status"),
        ("Flask App", "http://localhost:5000/api/system/info")
    ]
    
    print("🔍 Testing Service Health...")
    print("=" * 50)
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                print(f"✅ {name}: Online")
                if name == "MCP Filesystem":
                    data = response.json()
                    print(f"   Vectorization: {data.get('vectorization', {}).get('enabled', 'Unknown')}")
                    print(f"   Total Vectors: {data.get('vectorization', {}).get('totalVectors', 0)}")
            else:
                print(f"❌ {name}: Error {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Connection failed - {e}")
    
    print()

def test_embedding_service():
    """Test embedding service directly"""
    print("🧠 Testing Embedding Service...")
    print("=" * 50)
    
    try:
        test_text = "This is a test for vector embedding generation"
        response = requests.post(
            "http://localhost:8080/embed",
            json={"text": test_text},
            timeout=10
        )
        
        if response.ok:
            data = response.json()
            embedding = data.get('embedding', [])
            print(f"✅ Embedding generated successfully")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   Model: {data.get('model', 'Unknown')}")
            print(f"   First 5 values: {embedding[:5]}")
        else:
            print(f"❌ Embedding failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Embedding service error: {e}")
    
    print()

def test_semantic_search():
    """Test semantic search through MCP server"""
    print("🔍 Testing Semantic Search...")
    print("=" * 50)
    
    try:
        # First check if we have any files
        files_response = requests.get("http://localhost:3000/files", timeout=5)
        if files_response.ok:
            files = files_response.json()
            print(f"📁 Found {len(files)} files")
            
            if len(files) == 0:
                print("⚠️  No files found - upload some context files first")
                return
            
            # Test search
            search_response = requests.post(
                "http://localhost:3000/search",
                json={
                    "query": "system features capabilities",
                    "topK": 3,
                    "minSimilarity": 0.3
                },
                timeout=10
            )
            
            if search_response.ok:
                data = search_response.json()
                results = data.get('results', [])
                print(f"✅ Search completed successfully")
                print(f"   Results found: {len(results)}")
                
                for i, result in enumerate(results[:3], 1):
                    print(f"\n   Result {i}:")
                    print(f"   File: {result.get('filename', 'Unknown')}")
                    print(f"   Similarity: {result.get('similarity', 0):.3f}")
                    print(f"   Chunk: {result.get('chunk', '')[:100]}...")
                    
            else:
                print(f"❌ Search failed: {search_response.status_code} - {search_response.text}")
                
        else:
            print(f"❌ Cannot list files: {files_response.status_code}")
            
    except Exception as e:
        print(f"❌ Search test error: {e}")
    
    print()

def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("🚀 Testing Full RAG Pipeline...")
    print("=" * 50)
    
    try:
        test_query = "What are the key features of this AI system?"
        
        response = requests.post(
            "http://localhost:5000/api/chat",
            json={
                "model": "llama3.2",  # Adjust based on your installed model
                "message": test_query,
                "context_files": [],  # Will use all available files
                "model_params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "seed": -1,
                    "num_predict": -1
                },
                "conversation_id": "test_conversation",
                "fast_mode": True
            },
            timeout=30
        )
        
        if response.ok:
            data = response.json()
            print(f"✅ RAG pipeline completed successfully")
            print(f"   Processing mode: {data.get('metadata', {}).get('processing_mode', 'Unknown')}")
            print(f"   Chunks used: {data.get('metadata', {}).get('context_chunks_used', 0)}")
            print(f"   Confidence: {data.get('metadata', {}).get('confidence_score', 'Unknown')}")
            print(f"   Citations: {len(data.get('citations', []))}")
            print(f"\n   Response preview: {data.get('response', '')[:200]}...")
            
        else:
            print(f"❌ RAG pipeline failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
    
    print()

def main():
    print("🔧 Local AI Stack Vectorization Debug Tool")
    print("=" * 60)
    print()
    
    test_service_health()
    test_embedding_service()
    test_semantic_search()
    test_full_rag_pipeline()
    
    print("🏁 Debug testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
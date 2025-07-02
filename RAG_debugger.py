#!/usr/bin/env python3
"""
RAG Pipeline Issue Debugger
Identifies and helps fix specific issues found in the test results
"""

import requests
import json
import time
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class RAGDebugger:
    def __init__(self):
        self.base_urls = {
            "flask": "http://localhost:5000",
            "mcp": "http://localhost:3000",
            "qdrant": "http://localhost:6333",
            "ollama": "http://localhost:11434"
        }
        
    def print_section(self, title):
        print(f"\n{Colors.CYAN}{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}{Colors.ENDC}\n")
        
    def debug_all(self):
        """Run all debug checks"""
        self.print_section("RAG PIPELINE ISSUE DEBUGGER")
        
        # 1. Check Ollama models
        self.check_ollama_models()
        
        # 2. Check Qdrant collection
        self.check_qdrant_collection()
        
        # 3. Test vector upload and retrieval
        self.test_vector_flow()
        
        # 4. Test with correct model
        self.test_correct_model()
        
    def check_ollama_models(self):
        """Check available Ollama models"""
        self.print_section("1. OLLAMA MODEL CHECK")
        
        try:
            response = requests.get(f"{self.base_urls['ollama']}/api/tags")
            if response.ok:
                data = response.json()
                models = data.get('models', [])
                
                print(f"{Colors.GREEN}Available Ollama models:{Colors.ENDC}")
                if models:
                    for model in models:
                        print(f"  • {model['name']} ({self.format_size(model.get('size', 0))})")
                else:
                    print(f"  {Colors.RED}No models found! You need to pull a model.{Colors.ENDC}")
                    print(f"\n  Run: docker exec ollama-server ollama pull llama2")
                    
                # Check if llama3.2 exists
                model_names = [m['name'] for m in models]
                if 'llama3.2' not in model_names:
                    print(f"\n{Colors.YELLOW}⚠️  'llama3.2' not found. Use one of the above models.{Colors.ENDC}")
                    if 'llama2' in model_names:
                        print(f"{Colors.GREEN}✓ 'llama2' is available and can be used instead.{Colors.ENDC}")
                        
        except Exception as e:
            print(f"{Colors.RED}Error checking Ollama: {e}{Colors.ENDC}")
            
    def check_qdrant_collection(self):
        """Check Qdrant collection details"""
        self.print_section("2. QDRANT COLLECTION CHECK")
        
        try:
            # Get collection info
            response = requests.get(f"{self.base_urls['qdrant']}/collections/documents")
            if response.ok:
                data = response.json()
                result = data.get('result', {})
                
                print(f"Collection 'documents' info:")
                print(f"  • Points count: {result.get('points_count', 0)}")
                print(f"  • Vectors count: {result.get('vectors_count', 0)}")
                print(f"  • Status: {result.get('status', 'unknown')}")
                
                # Get actual points
                scroll_response = requests.post(
                    f"{self.base_urls['qdrant']}/collections/documents/points/scroll",
                    json={"limit": 10, "with_payload": True}
                )
                
                if scroll_response.ok:
                    points = scroll_response.json().get('result', {}).get('points', [])
                    print(f"\n{Colors.CYAN}Stored vectors:{Colors.ENDC}")
                    for point in points[:5]:
                        payload = point.get('payload', {})
                        print(f"  • ID: {point['id']}, File: {payload.get('filename', 'unknown')}, "
                              f"Chunk: {payload.get('chunk_index', 0)}")
                    if len(points) > 5:
                        print(f"  ... and {len(points)-5} more")
                        
            else:
                print(f"{Colors.RED}Collection 'documents' not found{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.RED}Error checking Qdrant: {e}{Colors.ENDC}")
            
    def test_vector_flow(self):
        """Test vector upload and search"""
        self.print_section("3. VECTOR UPLOAD & SEARCH TEST")
        
        test_content = "Vector databases enable semantic search in AI applications"
        
        # Upload test file
        print(f"{Colors.YELLOW}Uploading test content...{Colors.ENDC}")
        files = {'file': ('vector_test.txt', test_content, 'text/plain')}
        
        try:
            response = requests.post(f"{self.base_urls['flask']}/api/files", files=files)
            if response.ok:
                data = response.json()
                print(f"{Colors.GREEN}✓ Upload successful{Colors.ENDC}")
                print(f"  • Chunks created: {data.get('chunksCreated', 0)}")
                print(f"  • Status: {data.get('vectorizationStatus', 'unknown')}")
                
                # Wait for processing
                time.sleep(1)
                
                # Test search
                print(f"\n{Colors.YELLOW}Testing vector search...{Colors.ENDC}")
                search_response = requests.post(
                    f"{self.base_urls['mcp']}/search",
                    json={"query": "semantic search", "topK": 5}
                )
                
                if search_response.ok:
                    results = search_response.json().get('results', [])
                    print(f"{Colors.GREEN}✓ Search returned {len(results)} results{Colors.ENDC}")
                    for i, result in enumerate(results[:3]):
                        print(f"  [{i+1}] {result.get('filename')} - similarity: {result.get('similarity', 0):.3f}")
                else:
                    print(f"{Colors.RED}✗ Search failed: {search_response.status_code}{Colors.ENDC}")
                    
        except Exception as e:
            print(f"{Colors.RED}Error in vector flow: {e}{Colors.ENDC}")
            
    def test_correct_model(self):
        """Test chat with correct model"""
        self.print_section("4. CHAT TEST WITH CORRECT MODEL")
        
        # First get available models
        try:
            models_response = requests.get(f"{self.base_urls['ollama']}/api/tags")
            if models_response.ok:
                models = models_response.json().get('models', [])
                if models:
                    # Use first available model
                    model_name = models[0]['name']
                    print(f"{Colors.YELLOW}Testing with model: {model_name}{Colors.ENDC}")
                    
                    # Test chat
                    chat_data = {
                        "model": model_name,
                        "message": "What is semantic search?",
                        "model_params": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40,
                            "repeat_penalty": 1.1
                        },
                        "fast_mode": True,
                        "conversation_id": "debug_test"
                    }
                    
                    response = requests.post(
                        f"{self.base_urls['flask']}/api/chat",
                        json=chat_data,
                        timeout=30
                    )
                    
                    if response.ok:
                        data = response.json()
                        print(f"{Colors.GREEN}✓ Chat successful{Colors.ENDC}")
                        print(f"  • Response length: {len(data.get('response', ''))} chars")
                        print(f"  • Chunks used: {data.get('metadata', {}).get('context_chunks_used', 0)}")
                        print(f"  • Confidence: {data.get('metadata', {}).get('confidence_score', 0)}/10")
                    else:
                        print(f"{Colors.RED}✗ Chat failed: {response.status_code}{Colors.ENDC}")
                        print(f"  Response: {response.text[:200]}")
                        
        except Exception as e:
            print(f"{Colors.RED}Error testing chat: {e}{Colors.ENDC}")
            
    def format_size(self, size_bytes):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
        
    def suggest_fixes(self):
        """Suggest fixes based on findings"""
        self.print_section("SUGGESTED FIXES")
        
        print(f"{Colors.YELLOW}1. Install correct Ollama model:{Colors.ENDC}")
        print("   docker exec ollama-server ollama pull llama2")
        print("   # or")
        print("   docker exec ollama-server ollama pull mistral")
        
        print(f"\n{Colors.YELLOW}2. Update your code to use correct model:{Colors.ENDC}")
        print("   Change 'llama3.2' to 'llama2' in:")
        print("   • flask-app/app.py")
        print("   • flask-app/static/js/main.js")
        print("   • Test scripts")
        
        print(f"\n{Colors.YELLOW}3. If vectors aren't storing:{Colors.ENDC}")
        print("   • Check Qdrant logs: docker logs vector-database")
        print("   • Restart services: docker-compose restart")
        print("   • Recreate collection:")
        print("     curl -X DELETE http://localhost:6333/collections/documents")
        print("     curl -X POST http://localhost:3000/collections/init")
        
        print(f"\n{Colors.YELLOW}4. For better vector search:{Colors.ENDC}")
        print("   • Upload more diverse documents")
        print("   • Adjust similarity threshold (currently 0.3)")
        print("   • Check chunk size settings")

if __name__ == "__main__":
    debugger = RAGDebugger()
    debugger.debug_all()
    debugger.suggest_fixes()
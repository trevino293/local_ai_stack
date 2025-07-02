#!/usr/bin/env python3
"""
Test and analyze the Local AI Stack vector RAG data flow
Tests each component in the pipeline from query to response
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from dataclasses import dataclass, asdict

# Configuration
CONFIG = {
    "flask_url": "http://localhost:5000",
    "mcp_url": "http://localhost:3000", 
    "embedding_proxy_url": "http://localhost:8080",
    "qdrant_url": "http://localhost:6333",
    "ollama_url": "http://localhost:11434"
}

@dataclass
class TestResult:
    component: str
    endpoint: str
    status: str
    response_time_ms: float
    data: Optional[Dict] = None
    error: Optional[str] = None

class VectorRAGFlowTester:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.results: List[TestResult] = []
        self.test_file = "test_document.txt"
        self.test_content = """
        Vector databases are essential for modern AI applications.
        They enable semantic search by converting text into high-dimensional vectors.
        Qdrant is a production-ready vector database with excellent performance.
        The Local AI Stack uses vector embeddings for enhanced RAG capabilities.
        """
        self.test_query = "What are vector databases used for?"
        
    def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Starting Vector RAG Flow Test\n")
        
        # Test each component
        self.test_health_checks()
        self.test_file_upload()
        self.test_vector_search()
        self.test_full_rag_pipeline()
        self.analyze_results()
        
    def test_health_checks(self):
        """Test all service health endpoints"""
        print("1️⃣ Testing Service Health Checks...")
        
        # Test Qdrant (correct endpoint)
        self._test_endpoint(
            "Qdrant", 
            f"{self.config['qdrant_url']}/",
            "GET"
        )
        
        # Test Embedding Proxy
        self._test_endpoint(
            "Embedding Proxy",
            f"{self.config['embedding_proxy_url']}/health", 
            "GET"
        )
        
        # Test MCP Server (correct endpoint)
        self._test_endpoint(
            "MCP Server",
            f"{self.config['mcp_url']}/status",
            "GET"
        )
        
        # Test Flask App
        self._test_endpoint(
            "Flask App",
            f"{self.config['flask_url']}/api/system/info",
            "GET"
        )
        
        # Test Ollama
        self._test_endpoint(
            "Ollama",
            f"{self.config['ollama_url']}/api/tags",
            "GET"
        )
        
    def test_file_upload(self):
        """Test file upload and vectorization"""
        print("\n2️⃣ Testing File Upload & Vectorization...")
        
        # First, ensure collection is initialized
        self._test_endpoint(
            "Collection Init",
            f"{self.config['mcp_url']}/collections/init",
            "POST",
            json={}
        )
        
        # Upload via Flask (which calls MCP)
        files = {'file': (self.test_file, self.test_content, 'text/plain')}
        
        result = self._test_endpoint(
            "File Upload",
            f"{self.config['flask_url']}/api/files",
            "POST",
            files=files
        )
        
        if result.status == "success" and result.data:
            vec_proc = result.data.get('vector_processing', {})
            print(f"   📦 Chunks created: {vec_proc.get('chunks_created', 0)}")
            print(f"   🔧 Processing method: {vec_proc.get('processing_method', 'unknown')}")
            print(f"   ⏱️ Processing time: {vec_proc.get('processing_time_ms', 0)}ms")
            
    def test_vector_search(self):
        """Test vector search at each layer"""
        print("\n3️⃣ Testing Vector Search Pipeline...")
        
        # Test direct Qdrant search (if accessible)
        self._test_vector_db_search()
        
        # Test via Embedding Proxy
        self._test_embedding_proxy_search()
        
        # Test via MCP Server
        self._test_mcp_search()
        
    def _test_vector_db_search(self):
        """Test direct Qdrant search"""
        try:
            # First check if Qdrant is running
            response = requests.get(f"{self.config['qdrant_url']}/", timeout=5)
            if not response.ok:
                print(f"   ⚠️  Qdrant not accessible")
                return
                
            # Check if collection exists
            response = requests.get(
                f"{self.config['qdrant_url']}/collections",
                timeout=5
            )
            
            if response.ok:
                collections = response.json().get('result', {}).get('collections', [])
                doc_collection = next((c for c in collections if c.get('name') == 'documents'), None)
                
                if not doc_collection:
                    print(f"   ⚠️  'documents' collection not found. Creating...")
                    # Create collection
                    create_response = requests.put(
                        f"{self.config['qdrant_url']}/collections/documents",
                        json={
                            "vectors": {
                                "size": 384,
                                "distance": "Cosine"
                            }
                        },
                        timeout=10
                    )
                    if create_response.ok:
                        print(f"   ✅ Created 'documents' collection")
                else:
                    # Get collection info
                    info_response = requests.get(
                        f"{self.config['qdrant_url']}/collections/documents",
                        timeout=5
                    )
                    if info_response.ok:
                        points_count = info_response.json().get('result', {}).get('points_count', 0)
                        print(f"   📊 Qdrant collection info: {points_count} vectors")
        except Exception as e:
            print(f"   ⚠️  Direct Qdrant access: {e}")
            
    def _test_embedding_proxy_search(self):
        """Test search via embedding proxy"""
        search_data = {
            "query": self.test_query,
            "topK": 3,
            "minSimilarity": 0.3
        }
        
        result = self._test_endpoint(
            "Embedding Proxy Search",
            f"{self.config['embedding_proxy_url']}/search",
            "POST",
            json=search_data
        )
        
        if result.status == "success" and result.data:
            results = result.data.get('results', [])
            print(f"   🔍 Proxy search returned {len(results)} results")
            for i, r in enumerate(results[:2]):
                print(f"      Result {i+1}: similarity={r.get('similarity', 0):.3f}, file={r.get('filename', 'unknown')}")
                
    def _test_mcp_search(self):
        """Test search via MCP server"""
        search_data = {
            "query": self.test_query,
            "topK": 3,
            "minSimilarity": 0.3
        }
        
        result = self._test_endpoint(
            "MCP Server Search",
            f"{self.config['mcp_url']}/search",
            "POST", 
            json=search_data
        )
        
        if result.status == "success" and result.data:
            stats = result.data.get('searchStats', {})
            print(f"   📡 MCP search stats: {stats}")
            
    def test_full_rag_pipeline(self):
        """Test complete RAG pipeline"""
        print("\n4️⃣ Testing Full RAG Pipeline...")
        
        # Test fast mode
        self._test_rag_mode("fast", fast_mode=True)
        
        # Test detailed mode
        self._test_rag_mode("detailed", fast_mode=False)
        
    def _test_rag_mode(self, mode: str, fast_mode: bool):
        """Test specific RAG mode"""
        chat_data = {
            "model": "llama3.2",
            "message": self.test_query,
            "model_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "fast_mode": fast_mode,
            "conversation_id": f"test_{mode}"
        }
        
        result = self._test_endpoint(
            f"RAG Pipeline ({mode})",
            f"{self.config['flask_url']}/api/chat",
            "POST",
            json=chat_data
        )
        
        if result.status == "success" and result.data:
            metadata = result.data.get('metadata', {})
            print(f"\n   🤖 {mode.upper()} Mode Results:")
            print(f"      Processing time: {metadata.get('processing_time_ms', 0)}ms")
            print(f"      Context chunks: {metadata.get('context_chunks_used', 0)}")
            print(f"      Confidence: {metadata.get('confidence_score', 0)}/10")
            print(f"      Vector DB: {metadata.get('vector_database_info', {}).get('type', 'unknown')}")
            
            # Show search performance
            search_perf = metadata.get('search_performance', {})
            if search_perf:
                print(f"      Search time: {search_perf.get('search_time_ms', 0)}ms")
                print(f"      Avg similarity: {search_perf.get('avg_similarity', 0):.3f}")
                
    def _test_endpoint(self, component: str, url: str, method: str, 
                      json: Optional[Dict] = None, files: Optional[Dict] = None) -> TestResult:
        """Test a single endpoint"""
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                if files:
                    response = requests.post(url, files=files, timeout=30)
                else:
                    response = requests.post(url, json=json, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response_time = (time.time() - start_time) * 1000
            
            if response.ok:
                result = TestResult(
                    component=component,
                    endpoint=url,
                    status="success",
                    response_time_ms=round(response_time, 2),
                    data=response.json() if response.content else None
                )
                print(f"   ✅ {component}: {result.response_time_ms}ms")
            else:
                result = TestResult(
                    component=component,
                    endpoint=url,
                    status="error",
                    response_time_ms=round(response_time, 2),
                    error=f"HTTP {response.status_code}: {response.text[:100]}"
                )
                print(f"   ❌ {component}: {result.error}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = TestResult(
                component=component,
                endpoint=url,
                status="error",
                response_time_ms=round(response_time, 2),
                error=str(e)
            )
            print(f"   ❌ {component}: {result.error}")
            
        self.results.append(result)
        return result
        
    def analyze_results(self):
        """Analyze test results"""
        print("\n📊 Test Analysis")
        print("=" * 60)
        
        # Success rate
        success_count = sum(1 for r in self.results if r.status == "success")
        total_count = len(self.results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        print(f"Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # Response times
        success_results = [r for r in self.results if r.status == "success"]
        if success_results:
            avg_time = sum(r.response_time_ms for r in success_results) / len(success_results)
            max_time = max(r.response_time_ms for r in success_results)
            min_time = min(r.response_time_ms for r in success_results)
            
            print(f"\nResponse Times:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
            
        # Failed components
        failed = [r for r in self.results if r.status == "error"]
        if failed:
            print(f"\n❌ Failed Components:")
            for r in failed:
                print(f"  - {r.component}: {r.error}")
                
        # Data flow validation
        print(f"\n🔄 Data Flow Validation:")
        self._validate_data_flow()
        
    def _validate_data_flow(self):
        """Validate the complete data flow"""
        validations = {
            "1. Flask → MCP": self._check_result("Flask App") and self._check_result("MCP Server"),
            "2. MCP → Embedding Proxy": self._check_result("MCP Server") and self._check_result("Embedding Proxy"),
            "3. Proxy → Vector DB": self._check_result("Embedding Proxy") and self._check_result("Qdrant"),
            "4. Vector Search Flow": self._check_result("MCP Server Search"),
            "5. RAG Pipeline": self._check_result("RAG Pipeline (fast)") or self._check_result("RAG Pipeline (detailed)"),
            "6. Ollama Integration": self._check_result("Ollama")
        }
        
        for flow, valid in validations.items():
            status = "✅" if valid else "❌"
            print(f"  {status} {flow}")
            
        # Overall health
        all_valid = all(validations.values())
        print(f"\n{'✅ System is fully operational!' if all_valid else '⚠️  Some components need attention'}")
        
    def _check_result(self, component: str) -> bool:
        """Check if a component test passed"""
        return any(r.component == component and r.status == "success" for r in self.results)
        
    def save_report(self, filename: str = "vector_rag_test_report.json"):
        """Save detailed test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "successful": sum(1 for r in self.results if r.status == "success"),
                "failed": sum(1 for r in self.results if r.status == "error"),
                "avg_response_time_ms": sum(r.response_time_ms for r in self.results) / len(self.results) if self.results else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: {filename}")

def main():
    """Run the vector RAG flow test"""
    tester = VectorRAGFlowTester(CONFIG)
    
    try:
        tester.run_full_test()
        tester.save_report()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
End-to-End RAG Pipeline Test Script
Tests multi-step reasoning and vector embeddings flow
Shows complete message processing pipeline with detailed logging
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import textwrap

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuration
CONFIG = {
    "flask_url": "http://localhost:5000",
    "mcp_url": "http://localhost:3000",
    "qdrant_url": "http://localhost:6333",
    "ollama_url": "http://localhost:11434"
}

class RAGPipelineDebugger:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.test_document = """
        Artificial Intelligence and Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables systems to learn 
        and improve from experience without being explicitly programmed. Deep learning, a 
        specialized form of machine learning, uses neural networks with multiple layers 
        to progressively extract higher-level features from raw input.
        
        Key concepts include:
        - Supervised Learning: Training with labeled data
        - Unsupervised Learning: Finding patterns in unlabeled data
        - Reinforcement Learning: Learning through interaction and feedback
        - Neural Networks: Computational models inspired by biological neurons
        - Natural Language Processing: Enabling computers to understand human language
        
        Vector databases play a crucial role in modern AI applications by enabling 
        semantic search through high-dimensional vector representations of data.
        """
        
        self.test_queries = [
            {
                "query": "What is machine learning?",
                "mode": "fast",
                "expected_context": ["machine learning", "artificial intelligence", "learn from experience"]
            },
            {
                "query": "Explain the different types of learning in AI",
                "mode": "detailed",
                "expected_context": ["supervised", "unsupervised", "reinforcement"]
            },
            {
                "query": "How do vector databases relate to AI?",
                "mode": "fast",
                "expected_context": ["vector databases", "semantic search", "AI applications"]
            }
        ]
        
    def print_header(self, text: str, color: str = Colors.HEADER):
        """Print colored header"""
        print(f"\n{color}{'='*80}")
        print(f"{text:^80}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
    def print_step(self, step: str, status: str = "INFO", color: str = Colors.BLUE):
        """Print step with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        status_color = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }.get(status, Colors.BLUE)
        
        print(f"{Colors.BOLD}[{timestamp}]{Colors.ENDC} {status_color}[{status}]{Colors.ENDC} {color}{step}{Colors.ENDC}")
        
    def print_json(self, data: Dict, indent: int = 2):
        """Pretty print JSON data"""
        print(json.dumps(data, indent=indent, default=str))
        
    def run_complete_test(self):
        """Run complete end-to-end test"""
        self.print_header("RAG PIPELINE END-TO-END TEST", Colors.CYAN)
        
        # 1. Test system health
        if not self.test_system_health():
            print(f"{Colors.RED}System health check failed. Exiting.{Colors.ENDC}")
            return
            
        # 2. Upload test document
        if not self.upload_test_document():
            print(f"{Colors.RED}Document upload failed. Exiting.{Colors.ENDC}")
            return
            
        # 3. Test each query
        for i, test_case in enumerate(self.test_queries, 1):
            self.test_query_flow(test_case, test_number=i)
            
        # 4. Show final analysis
        self.show_pipeline_analysis()
        
    def test_system_health(self) -> bool:
        """Test all system components"""
        self.print_header("STEP 1: SYSTEM HEALTH CHECK")
        
        components = [
            ("Flask App", f"{self.config['flask_url']}/api/system/info"),
            ("MCP Server", f"{self.config['mcp_url']}/status"),
            ("Qdrant", f"{self.config['qdrant_url']}/"),
            ("Ollama", f"{self.config['ollama_url']}/api/tags")
        ]
        
        all_healthy = True
        
        for name, url in components:
            try:
                response = requests.get(url, timeout=5)
                if response.ok:
                    self.print_step(f"{name}: Online ✓", "SUCCESS")
                    if name == "Flask App":
                        data = response.json()
                        print(f"  └─ RAG Pipeline: {data.get('rag_pipeline', {})}")
                else:
                    self.print_step(f"{name}: Error {response.status_code}", "ERROR")
                    all_healthy = False
            except Exception as e:
                self.print_step(f"{name}: {str(e)}", "ERROR")
                all_healthy = False
                
        return all_healthy
        
    def upload_test_document(self) -> bool:
        """Upload test document and show vectorization"""
        self.print_header("STEP 2: DOCUMENT UPLOAD & VECTORIZATION")
        
        try:
            # Create test file
            files = {
                'file': ('ai_fundamentals.txt', self.test_document, 'text/plain')
            }
            
            self.print_step("Uploading document...", "INFO")
            start_time = time.time()
            
            response = requests.post(
                f"{self.config['flask_url']}/api/files",
                files=files,
                timeout=30
            )
            
            upload_time = round((time.time() - start_time) * 1000)
            
            if response.ok:
                data = response.json()
                self.print_step(f"Upload successful in {upload_time}ms", "SUCCESS")
                
                print(f"\n{Colors.CYAN}Vectorization Details:{Colors.ENDC}")
                print(f"  • File: {data.get('filename')}")
                print(f"  • Size: {data.get('fileSize')} bytes")
                print(f"  • Chunks created: {data.get('chunksCreated')}")
                print(f"  • Processing method: {data.get('processingMethod')}")
                print(f"  • Vector DB: {data.get('vectorDatabase')}")
                print(f"  • Status: {data.get('vectorizationStatus')}")
                
                # Check vector stats
                self.check_vector_stats()
                return True
            else:
                self.print_step(f"Upload failed: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.print_step(f"Upload error: {str(e)}", "ERROR")
            return False
            
    def check_vector_stats(self):
        """Check vector database statistics"""
        try:
            response = requests.get(f"{self.config['mcp_url']}/vectors/stats", timeout=5)
            if response.ok:
                stats = response.json()
                print(f"\n{Colors.YELLOW}Vector Database Stats:{Colors.ENDC}")
                print(f"  • Collection: {stats.get('collection')}")
                print(f"  • Total vectors: {stats.get('vectorsCount', 0)}")
                print(f"  • Indexed files: {stats.get('indexedFilesCount', 0)}")
                print(f"  • Status: {stats.get('status')}")
        except:
            pass
            
    def test_query_flow(self, test_case: Dict, test_number: int):
        """Test complete query flow with detailed logging"""
        query = test_case['query']
        mode = test_case['mode']
        
        self.print_header(f"TEST {test_number}: {mode.upper()} MODE QUERY")
        print(f"{Colors.BOLD}Query:{Colors.ENDC} {query}\n")
        
        # Step 1: Show vector search
        self.show_vector_search(query, mode)
        
        # Step 2: Process chat request
        self.process_chat_request(query, mode)
        
        # Step 3: Analyze response
        self.analyze_response(test_case)
        
    def show_vector_search(self, query: str, mode: str):
        """Show vector search process"""
        self.print_step("VECTOR SEARCH PHASE", "INFO", Colors.YELLOW)
        
        try:
            # Perform search
            search_data = {
                "query": query,
                "topK": 3 if mode == "fast" else 5,
                "minSimilarity": 0.3
            }
            
            response = requests.post(
                f"{self.config['mcp_url']}/search",
                json=search_data,
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                results = data.get('results', [])
                
                print(f"\n{Colors.CYAN}Search Results ({len(results)} chunks found):{Colors.ENDC}")
                
                for i, result in enumerate(results, 1):
                    print(f"\n  [{i}] {Colors.GREEN}Similarity: {result.get('similarity', 0):.3f}{Colors.ENDC}")
                    print(f"      File: {result.get('filename')}")
                    print(f"      Chunk: {textwrap.fill(result.get('text', '')[:150] + '...', width=70, initial_indent='      ', subsequent_indent='      ')}")
                    
                print(f"\n  Search method: {data.get('searchStats', {}).get('method', 'unknown')}")
                
        except Exception as e:
            self.print_step(f"Search error: {str(e)}", "ERROR")
            
    def process_chat_request(self, query: str, mode: str):
        """Process chat request and show reasoning"""
        self.print_step("\nCHAT PROCESSING PHASE", "INFO", Colors.YELLOW)
        
        try:
            chat_data = {
                "model": "llama3.2",
                "message": query,
                "model_params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                },
                "fast_mode": mode == "fast",
                "conversation_id": f"test_{mode}_{int(time.time())}"
            }
            
            print(f"\n{Colors.CYAN}Request Configuration:{Colors.ENDC}")
            print(f"  • Model: {chat_data['model']}")
            print(f"  • Mode: {mode}")
            print(f"  • Temperature: {chat_data['model_params']['temperature']}")
            
            self.print_step("\nSending chat request...", "INFO")
            start_time = time.time()
            
            response = requests.post(
                f"{self.config['flask_url']}/api/chat",
                json=chat_data,
                timeout=60
            )
            
            processing_time = round((time.time() - start_time) * 1000)
            
            if response.ok:
                data = response.json()
                self.print_step(f"Response received in {processing_time}ms", "SUCCESS")
                
                # Show metadata
                metadata = data.get('metadata', {})
                print(f"\n{Colors.CYAN}Processing Metadata:{Colors.ENDC}")
                print(f"  • Processing mode: {metadata.get('processing_mode')}")
                print(f"  • Context chunks used: {metadata.get('context_chunks_used')}")
                print(f"  • Confidence score: {metadata.get('confidence_score')}/10")
                print(f"  • Processing time: {metadata.get('processing_time_ms')}ms")
                
                # Show response
                print(f"\n{Colors.GREEN}AI Response:{Colors.ENDC}")
                print(textwrap.fill(data.get('response', ''), width=80))
                
                # Show citations
                citations = data.get('citations', [])
                if citations:
                    print(f"\n{Colors.CYAN}Citations ({len(citations)}):{Colors.ENDC}")
                    for cite in citations:
                        print(f"  • {cite['file']} [{cite['type']}]")
                        
                # Store for analysis
                self.last_response = data
                
            else:
                self.print_step(f"Chat failed: {response.status_code}", "ERROR")
                print(response.text)
                
        except Exception as e:
            self.print_step(f"Chat error: {str(e)}", "ERROR")
            
    def analyze_response(self, test_case: Dict):
        """Analyze response quality"""
        self.print_step("\nRESPONSE ANALYSIS", "INFO", Colors.YELLOW)
        
        if hasattr(self, 'last_response'):
            response = self.last_response.get('response', '')
            expected = test_case.get('expected_context', [])
            
            print(f"\n{Colors.CYAN}Expected Context Coverage:{Colors.ENDC}")
            for term in expected:
                found = term.lower() in response.lower()
                status = f"{Colors.GREEN}✓ Found{Colors.ENDC}" if found else f"{Colors.RED}✗ Missing{Colors.ENDC}"
                print(f"  • '{term}': {status}")
                
    def show_pipeline_analysis(self):
        """Show overall pipeline analysis"""
        self.print_header("PIPELINE FLOW SUMMARY", Colors.CYAN)
        
        print(f"{Colors.YELLOW}Complete RAG Pipeline Flow:{Colors.ENDC}\n")
        
        flow_steps = [
            ("1. User Query", "Flask receives chat request"),
            ("2. Vector Search", "MCP server searches Qdrant for relevant chunks"),
            ("3. Context Assembly", "Top-K chunks assembled based on similarity"),
            ("4. Prompt Construction", "Query + context + conversation history"),
            ("5. LLM Generation", "Ollama processes prompt and generates response"),
            ("6. Post-Processing", "Citations extracted, confidence calculated"),
            ("7. Response Delivery", "Enhanced response with metadata returned")
        ]
        
        for step, description in flow_steps:
            print(f"  {Colors.BOLD}{step}{Colors.ENDC}")
            print(f"    └─ {description}")
            
        print(f"\n{Colors.GREEN}Key Integration Points:{Colors.ENDC}")
        print("  • Flask ↔ MCP Server: File operations and search")
        print("  • MCP Server ↔ Qdrant: Vector storage and retrieval")
        print("  • Flask ↔ Ollama: Text generation")
        print("  • All services: Health monitoring and status checks")
        
    def run_interactive_test(self):
        """Run interactive test mode"""
        self.print_header("INTERACTIVE RAG PIPELINE TEST", Colors.CYAN)
        
        while True:
            try:
                print(f"\n{Colors.YELLOW}Enter a query (or 'quit' to exit):{Colors.ENDC}")
                query = input("> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                    
                print(f"\n{Colors.YELLOW}Select mode:{Colors.ENDC}")
                print("  1. Fast mode (3 chunks)")
                print("  2. Detailed mode (5+ chunks)")
                mode_input = input("Choice (1/2): ").strip()
                
                mode = "fast" if mode_input == "1" else "detailed"
                
                test_case = {
                    "query": query,
                    "mode": mode,
                    "expected_context": []
                }
                
                self.test_query_flow(test_case, test_number="Interactive")
                
            except KeyboardInterrupt:
                print("\n\nTest interrupted.")
                break
                
def main():
    """Main function"""
    debugger = RAGPipelineDebugger(CONFIG)
    
    print(f"{Colors.BOLD}RAG Pipeline End-to-End Debugger{Colors.ENDC}")
    print("This tool shows the complete flow of multi-step reasoning and vector embeddings\n")
    
    print("Select test mode:")
    print("1. Run complete automated test")
    print("2. Interactive query mode")
    
    choice = input("\nChoice (1/2): ").strip()
    
    if choice == "1":
        debugger.run_complete_test()
    elif choice == "2":
        debugger.run_interactive_test()
    else:
        print("Invalid choice. Exiting.")
        
if __name__ == "__main__":
    main()
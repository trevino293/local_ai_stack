# flask-app/app.py - Simplified RAG with proper model handling
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import json
from datetime import datetime
import logging
import time

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:3000')
VECTOR_DB_URL = os.getenv('VECTOR_DB_URL', 'http://localhost:6333')

# Default to a model that's more likely to exist
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'llama3.2:3b')

# System files that should always be included as context
SYSTEM_FILES = ['admin', 'system', 'default', 'config']

# In-memory storage
chat_history = []
saved_model_params = {}
saved_configurations = []
config_id_counter = 1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemContextManager:
    """Manages system context injection for all queries"""
    
    def __init__(self, mcp_server_url):
        self.mcp_server_url = mcp_server_url
        self.system_context_cache = None
        self.last_cache_update = 0
        self.cache_duration = 300  # 5 minutes
        self.system_keywords = ['system', 'admin', 'config', 'context']
        
    def get_system_context(self):
        """Get cached system context or refresh if needed"""
        current_time = time.time()
        
        if (self.system_context_cache is None or 
            current_time - self.last_cache_update > self.cache_duration):
            self.system_context_cache = self._load_system_context()
            self.last_cache_update = current_time
            
        return self.system_context_cache
    
    def _load_system_context(self):
        """Load essential system context from system files"""
        try:
            # Search for system configuration files
            response = requests.post(
                f"{self.mcp_server_url}/search",
                json={
                    'query': 'system configuration features architecture api vectorization rag',
                    'topK': 10,
                    'minSimilarity': 0.1
                },
                timeout=10
            )
            
            if response.ok:
                results = response.json().get('results', [])
                
                # Extract system-specific information
                system_info = {
                    'system_name': 'Local AI Stack (OL-MCP)',
                    'version': '3.0.0',
                    'core_features': [],
                    'architecture': [],
                    'capabilities': [],
                    'api_endpoints': [],
                    'models': []
                }
                
                for result in results:
                    filename = result.get('filename', '').lower()
                    text = result.get('text', '')
                    
                    # Parse system.json content
                    if 'system.json' in filename:
                        self._parse_system_json(text, system_info)
                    
                    # Parse config.json content
                    elif 'config.json' in filename:
                        self._parse_config_json(text, system_info)
                    
                    # Parse any other system files
                    elif any(kw in filename for kw in self.system_keywords):
                        self._parse_generic_system_file(text, system_info)
                
                # Build comprehensive system context
                return self._build_system_context(system_info)
                
        except Exception as e:
            logger.error(f"Failed to load system context: {e}")
            
        # Fallback system context
        return self._get_fallback_context()
    
    def _parse_system_json(self, text, system_info):
        """Extract information from system.json content"""
        text_lower = text.lower()
        
        # Features
        if 'vectorized rag' in text_lower or 'vectorization' in text_lower:
            system_info['core_features'].append('Vectorized RAG with semantic search')
        if 'multi-step reasoning' in text_lower or 'six-stage' in text_lower:
            system_info['core_features'].append('Six-stage deliberation system')
        if 'confidence scoring' in text_lower:
            system_info['core_features'].append('AI confidence scoring (1-10)')
        if 'citation tracking' in text_lower:
            system_info['core_features'].append('Automatic citation tracking')
        
        # Architecture components
        if 'qdrant' in text_lower:
            system_info['architecture'].append('Qdrant vector database')
        if 'flask' in text_lower:
            system_info['architecture'].append('Flask web interface (port 5000)')
        if 'ollama' in text_lower:
            system_info['architecture'].append('Ollama LLM server (port 11434)')
        if 'mcp' in text_lower or 'filesystem' in text_lower:
            system_info['architecture'].append('MCP filesystem server (port 3000)')
        
        # API endpoints
        if '/api/chat' in text:
            system_info['api_endpoints'].append('POST /api/chat - Enhanced chat')
        if '/api/files' in text:
            system_info['api_endpoints'].append('GET/POST /api/files - File management')
        if '/search' in text:
            system_info['api_endpoints'].append('POST /search - Semantic search')
        
        # Capabilities
        if 'semantic search' in text_lower:
            system_info['capabilities'].append('Semantic document search')
        if 'document processing' in text_lower or 'vectorization' in text_lower:
            system_info['capabilities'].append('Automatic document vectorization')
        if 'conversation history' in text_lower:
            system_info['capabilities'].append('Persistent conversation history')
    
    def _parse_config_json(self, text, system_info):
        """Extract information from config.json content"""
        if 'response_framework' in text:
            system_info['capabilities'].append('Enterprise response framework')
        if 'gcp_specific' in text:
            system_info['capabilities'].append('GCP integration guidance')
        if 'risk_assessment' in text:
            system_info['capabilities'].append('Risk assessment framework')
    
    def _parse_generic_system_file(self, text, system_info):
        """Parse any other system-related files"""
        text_lower = text.lower()
        
        # Look for model references
        if 'llama' in text_lower:
            system_info['models'].append('Llama models')
        if 'mistral' in text_lower:
            system_info['models'].append('Mistral')
        if 'codellama' in text_lower:
            system_info['models'].append('CodeLlama')
    
    def _build_system_context(self, system_info):
        """Build a comprehensive system context string"""
        context_parts = [
            f"=== {system_info['system_name']} v{system_info['version']} ===",
            "",
            "CORE FEATURES:",
            "• " + "\n• ".join(system_info['core_features'][:5]) if system_info['core_features'] 
                else "• Vectorized RAG pipeline\n• Multi-step reasoning\n• Confidence scoring",
            "",
            "ARCHITECTURE:",
            "• " + "\n• ".join(system_info['architecture'][:4]) if system_info['architecture']
                else "• Flask app\n• MCP server\n• Qdrant vector DB\n• Ollama LLM",
            "",
            "KEY CAPABILITIES:",
            "• " + "\n• ".join(system_info['capabilities'][:4]) if system_info['capabilities']
                else "• Semantic search\n• Document vectorization\n• Chat history\n• Citation tracking",
            ""
        ]
        
        # Add API info if available
        if system_info['api_endpoints']:
            context_parts.extend([
                "MAIN APIS:",
                "• " + "\n• ".join(system_info['api_endpoints'][:3]),
                ""
            ])
        
        # Add model info if available
        if system_info['models']:
            context_parts.extend([
                f"SUPPORTED MODELS: {', '.join(set(system_info['models']))}",
                ""
            ])
        
        return "\n".join(context_parts)
    
    def _get_fallback_context(self):
        """Return fallback context when system files can't be loaded"""
        return """=== Local AI Stack (OL-MCP) v3.0.0 ===

CORE FEATURES:
• Vectorized RAG pipeline with semantic search
• Six-stage multi-step reasoning system
• AI confidence scoring (1-10)
• Automatic citation tracking
• Fast and detailed processing modes

ARCHITECTURE:
• Flask web interface (port 5000)
• MCP filesystem server (port 3000)
• Qdrant vector database (port 6333)
• Ollama LLM server (port 11434)

KEY CAPABILITIES:
• Semantic document search using embeddings
• Automatic document vectorization on upload
• Persistent conversation history
• System file prioritization
• Model parameter configuration

MAIN APIS:
• POST /api/chat - Enhanced chat with RAG
• GET/POST /api/files - File management
• POST /search - Semantic search

This is a self-hosted AI system with complete data privacy."""

# Initialize system context manager
system_context_mgr = SystemContextManager(MCP_SERVER_URL)


class SimpleRAGPipeline:
    """Simplified RAG pipeline focused on vector search"""
    
    def __init__(self, ollama_host, mcp_server_url):
        self.ollama_host = ollama_host
        self.mcp_server_url = mcp_server_url
        self.available_models = []
        
        logger.info(f"Initialized Simple RAG Pipeline")
        logger.info(f"  Ollama: {self.ollama_host}")
        logger.info(f"  MCP Server: {self.mcp_server_url}")
        
        # Check available models on startup
        self._check_available_models()
        
    def _check_available_models(self):
        """Check which models are available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.ok:
                data = response.json()
                self.available_models = [m['name'] for m in data.get('models', [])]
                logger.info(f"Available Ollama models: {self.available_models}")
            else:
                logger.warning("Could not fetch Ollama models")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")
            
    def _get_valid_model(self, requested_model):
        """Get a valid model name, falling back if needed"""
        if requested_model in self.available_models:
            return requested_model
        
        # Try without version suffix
        base_model = requested_model.split(':')[0]
        for model in self.available_models:
            if model.startswith(base_model):
                logger.info(f"Using {model} instead of {requested_model}")
                return model
        
        # Fallback to first available model
        if self.available_models:
            logger.warning(f"Model {requested_model} not found, using {self.available_models[0]}")
            return self.available_models[0]
        
        # Last resort
        return DEFAULT_MODEL
        
    def process_query(self, model, user_message, model_params, conversation_history=None, fast_mode=True):
        """Process query with simplified vector search"""
        
        start_time = time.time()
        
        try:
            # Validate model
            valid_model = self._get_valid_model(model)
            
            logger.info(f"Processing query: '{user_message[:50]}...' with model {valid_model}")
            
            # 1. Search for relevant chunks
            search_options = {
                'topK': 3 if fast_mode else 5,
                'minSimilarity': 0.1  # Lower threshold for better recall
            }
            
            search_results = self._search_vectors(user_message, **search_options)
            
            # 2. Build context from search results
            context_content = self._build_context(search_results)
            
            # 3. Create prompt
            prompt = self._create_prompt(user_message, context_content, conversation_history, fast_mode)
            
            # 4. Generate response
            response = self._call_ollama(valid_model, prompt, model_params)
            
            # 5. Extract metadata
            citations = self._extract_citations(search_results)
            confidence = self._calculate_confidence(search_results, response)
            
            processing_time = round((time.time() - start_time) * 1000)
            
            return {
                'response': response,
                'citations': citations,
                'search_results': search_results,
                'metadata': {
                    'model_used': valid_model,
                    'processing_mode': 'fast' if fast_mode else 'detailed',
                    'context_chunks_used': len(search_results),
                    'confidence_score': confidence,
                    'processing_time_ms': processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            processing_time = round((time.time() - start_time) * 1000)
            
            return {
                'response': f"I encountered an error: {str(e)}. Please check that the model '{model}' is installed.",
                'citations': [],
                'search_results': [],
                'metadata': {
                    'processing_mode': 'error',
                    'error': str(e),
                    'processing_time_ms': processing_time
                }
            }
    
    def _search_vectors(self, query, topK=5, minSimilarity=0.1):
        """Search for relevant chunks via MCP server"""
        try:
            response = requests.post(
                f"{self.mcp_server_url}/search",
                json={
                    'query': query,
                    'topK': topK,
                    'minSimilarity': minSimilarity
                },
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"Vector search returned {len(results)} results")
                return results
            else:
                logger.error(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    # Add these methods to SimpleRAGPipeline class in flask-app/app.py

    def _build_context(self, search_results):
        """Build context with system file prioritization and relevance scoring"""
        if not search_results:
            return "No relevant context found in the knowledge base."
    
        # Separate system files from regular files
        system_results = []
        regular_results = []
    
        for result in search_results:
            filename = result.get('filename', '').lower()
            if any(kw in filename for kw in ['system', 'admin', 'config', 'context']):
                system_results.append(result)
            else:
                regular_results.append(result)
    
        context_parts = []
    
        # Always include system files first with clear marking
        if system_results:
            context_parts.append("=== SYSTEM CONTEXT ===")
            for i, result in enumerate(system_results, 1):
                filename = result.get('filename', 'unknown')
                text = result.get('text', '')
                similarity = result.get('similarity', 0)
            
                # For system.json, parse and include key sections
                if 'system.json' in filename and '{' in text:
                    try:
                        # Extract key system information
                        context_parts.append(f"[SYSTEM CONFIG - {filename}]")
                        context_parts.append(self._extract_system_context(text))
                    except:
                        context_parts.append(f"[SYSTEM FILE {i}: {filename} (relevance: {similarity:.2f})]")
                        context_parts.append(text)
                else:
                    context_parts.append(f"[SYSTEM FILE {i}: {filename} (relevance: {similarity:.2f})]")
                    context_parts.append(text)
                context_parts.append("")
    
        # Then include regular context
        if regular_results:
            context_parts.append("=== DOCUMENT CONTEXT ===")
            for i, result in enumerate(regular_results, 1):
                filename = result.get('filename', 'unknown')
                text = result.get('text', '')
                similarity = result.get('similarity', 0)
            
                context_parts.append(f"[Document {i}: {filename} (relevance: {similarity:.2f})]")
                context_parts.append(text)
                context_parts.append("")
    
        return "\n".join(context_parts)

    def _extract_system_context(self, json_text):
        """Extract key information from system.json content"""
        try:
            # Try to parse relevant parts
            key_sections = []
        
            # Look for specific patterns
            if 'key_features' in json_text:
                key_sections.append("Key System Features:")
                # Extract features section
                features_start = json_text.find('"key_features"')
                if features_start > -1:
                    # Simple extraction of the features section
                    features_text = json_text[features_start:features_start+1000]
                    key_sections.append(features_text[:500] + "...")
        
            if 'system_architecture' in json_text:
                key_sections.append("\nSystem Architecture:")
                arch_start = json_text.find('"system_architecture"')
                if arch_start > -1:
                    arch_text = json_text[arch_start:arch_start+800]
                    key_sections.append(arch_text[:400] + "...")
        
            if 'api_endpoints' in json_text:
                key_sections.append("\nAPI Information:")
                api_start = json_text.find('"api_endpoints"')
                if api_start > -1:
                    api_text = json_text[api_start:api_start+600]
                    key_sections.append(api_text[:300] + "...")
        
            return "\n".join(key_sections) if key_sections else json_text[:1000]
        except:
            return json_text[:1000]

    def _create_prompt(self, query, context, history, fast_mode):
        """Create enhanced prompt with automatic system context injection"""
        prompt_parts = []
    
        # ALWAYS inject system context first
        system_context = system_context_mgr.get_system_context()
        prompt_parts.append(system_context)
        prompt_parts.append("")
    
        # System instruction
        prompt_parts.append("You are the AI assistant for the Local AI Stack system described above.")
        prompt_parts.append("Use your knowledge of the system architecture and features when answering questions.")
        prompt_parts.append("")
    
        # Add conversation history if available
        if history and len(history) > 0:
            prompt_parts.append("=== CONVERSATION HISTORY ===")
            for h in history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {h['message'][:150]}...")
                prompt_parts.append(f"Assistant: {h['response'][:150]}...")
            prompt_parts.append("")
    
        # Add retrieved context with clear separation
        if context and context != "No relevant context found in the knowledge base.":
            prompt_parts.append("=== KNOWLEDGE BASE CONTEXT ===")
            prompt_parts.append(context)
            prompt_parts.append("")
    
        # Add the query with clear marking
        prompt_parts.append("=== USER QUESTION ===")
        prompt_parts.append(f"{query}")
        prompt_parts.append("")
    
        # Mode-specific instructions
        prompt_parts.append("=== INSTRUCTIONS ===")
        if fast_mode:
            prompt_parts.append("Provide a clear, concise answer based on the system information and context provided.")
            prompt_parts.append("Focus on the most relevant information.")
        else:
            prompt_parts.append("Provide a comprehensive answer using the system information and context provided.")
            prompt_parts.append("1. Consider the system's architecture and capabilities")
            prompt_parts.append("2. Include relevant details from the knowledge base")
            prompt_parts.append("3. Explain your reasoning and cite specific sources when applicable")
            prompt_parts.append("4. If the context doesn't contain enough information, acknowledge this")
    
        return "\n".join(prompt_parts)

    def _calculate_confidence(self, search_results, response):
        """Enhanced confidence calculation based on context quality"""
        if not search_results:
            return 3  # Low confidence without context
    
        # Check for system file presence
        has_system_context = any(
            any(kw in r.get('filename', '').lower() for kw in ['system', 'admin', 'config'])
            for r in search_results
        )
    
        # Average similarity of results
        avg_similarity = sum(r.get('similarity', 0) for r in search_results) / len(search_results)
    
        # Count high-quality results
        high_quality_results = sum(1 for r in search_results if r.get('similarity', 0) > 0.5)
    
        # Base confidence calculation
        if avg_similarity > 0.7 and high_quality_results >= 2:
            base_confidence = 9
        elif avg_similarity > 0.5:
            base_confidence = 7
        elif avg_similarity > 0.3:
            base_confidence = 6
        else:
            base_confidence = 4
    
        # Boost for system context
        if has_system_context:
            base_confidence = min(10, base_confidence + 1)
    
        # Adjust based on response quality
        if len(response) > 200 and response not in ["No response generated", "Error"]:
            # Check if response seems to use context
            context_terms = set()
            for result in search_results:
                text = result.get('text', '').lower()
                context_terms.update(text.split()[:20])  # Key terms from context
        
            response_lower = response.lower()
            context_usage = sum(1 for term in context_terms if term in response_lower)
        
            if context_usage > 10:
                base_confidence = min(10, base_confidence + 1)
    
        return base_confidence
    
    def _call_ollama(self, model, prompt, params):
        """Call Ollama API"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get('temperature', 0.7),
                "top_p": params.get('top_p', 0.9),
                "top_k": params.get('top_k', 40),
                "repeat_penalty": params.get('repeat_penalty', 1.1)
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.ok:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                error_msg = f"Ollama API error {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', response.text)}"
                except:
                    error_msg += f": {response.text}"
                return error_msg
                
        except requests.exceptions.Timeout:
            return "Response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error calling model: {str(e)}"
    
    def _extract_citations(self, search_results):
        """Extract citations from search results"""
        citations = []
        seen_files = set()
        
        for result in search_results:
            filename = result.get('filename', '')
            if filename and filename not in seen_files:
                seen_files.add(filename)
                citations.append({
                    'file': filename,
                    'type': 'SYSTEM' if any(kw in filename.lower() for kw in SYSTEM_FILES) else 'USER',
                    'relevance': result.get('similarity', 0)
                })
        
        return citations
    

# Initialize RAG pipeline
rag_pipeline = SimpleRAGPipeline(OLLAMA_HOST, MCP_SERVER_URL)

# Modify the process_query method to inject system context
# Add this at the beginning of _create_prompt method:

def _create_prompt(self, query, context, history, fast_mode):
    """Create enhanced prompt with system context injection"""
    prompt_parts = []
    
    # ALWAYS inject system context first
    system_context = system_context_mgr.get_system_context()
    prompt_parts.append("=== SYSTEM INFORMATION ===")
    prompt_parts.append(system_context)
    prompt_parts.append("")
    
    # System instruction
    prompt_parts.append("You are the AI assistant for the Local AI Stack system described above.")
    prompt_parts.append("Use your knowledge of the system architecture and features when answering questions.")
    prompt_parts.append("")
    
    # Rest of the prompt building continues as before...
    # [Previous prompt building code]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if response.ok:
            data = response.json()
            models = data.get('models', [])
            
            # Update pipeline's model list
            rag_pipeline.available_models = [m['name'] for m in models]
            
            return jsonify({"models": models})
        else:
            return jsonify({"error": f"Ollama API error: {response.status_code}"}), 500
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return jsonify({"error": str(e), "models": []}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint"""
    try:
        data = request.json
        model = data.get('model', DEFAULT_MODEL)
        message = data.get('message', '')
        model_params = data.get('model_params', {})
        conversation_id = data.get('conversation_id', 'default')
        fast_mode = data.get('fast_mode', True)
        
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get conversation history
        conversation_history = [
            entry for entry in chat_history 
            if entry.get('conversation_id') == conversation_id
        ][-5:]
        
        # Process with RAG pipeline
        result = rag_pipeline.process_query(
            model=model,
            user_message=message,
            model_params=model_params,
            conversation_history=conversation_history,
            fast_mode=fast_mode
        )
        
        # Store in history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "model": result['metadata'].get('model_used', model),
            "message": message,
            "response": result['response'],
            "citations": result['citations'],
            "metadata": result['metadata']
        }
        
        chat_history.append(chat_entry)
        
        return jsonify({
            'response': result['response'],
            'citations': result['citations'],
            'conversation_id': conversation_id,
            'metadata': result['metadata']
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": str(e),
            "response": "I encountered an error processing your request.",
            "metadata": {"error": True}
        }), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List files from MCP server"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/files", timeout=10)
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"MCP server error: {response.status_code}"}), 500
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files', methods=['POST'])
def upload_file():
    """Upload file to MCP server"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{MCP_SERVER_URL}/files", files=files, timeout=60)
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Upload failed: {response.text}"}), 500
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete file"""
    if any(keyword in filename.lower() for keyword in SYSTEM_FILES):
        return jsonify({"error": "Cannot delete system files"}), 403
    
    try:
        response = requests.delete(f"{MCP_SERVER_URL}/files/{filename}", timeout=30)
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Delete failed: {response.status_code}"}), 500
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp/status', methods=['GET'])
def mcp_status():
    """Get MCP server status"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/status", timeout=5)
        return jsonify(response.json() if response.ok else {"status": "offline"})
    except:
        return jsonify({"status": "offline"}), 503

@app.route('/api/embedding/health', methods=['GET'])
def embedding_health():
    """Check vector health via MCP stats"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/vectors/stats", timeout=5)
        if response.ok:
            stats = response.json()
            return jsonify({
                "status": "online",
                "vector_database": "qdrant",
                "points_count": stats.get('vectorsCount', 0),
                "indexed_files": stats.get('indexedFilesCount', 0),
                "embedding_method": "simple-embedding"
            })
        else:
            return jsonify({"status": "offline"}), 503
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e)}), 503

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        "system": {
            "name": "Local AI Stack",
            "version": "3.1.0",
            "timestamp": datetime.now().isoformat()
        },
        "rag_pipeline": {
            "vectorized_search_enabled": True,
            "vector_database": "qdrant",
            "embedding_method": "simple-embedding",
            "available_models": rag_pipeline.available_models,
            "default_model": DEFAULT_MODEL
        },
        "services": {
            "ollama_host": OLLAMA_HOST,
            "mcp_server_url": MCP_SERVER_URL,
            "vector_db_url": VECTOR_DB_URL
        }
    })

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history"""
    conversation_id = request.args.get('conversation_id', 'default')
    filtered = [h for h in chat_history if h.get('conversation_id') == conversation_id]
    return jsonify({"history": filtered})

@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared"})

# Model parameters routes
@app.route('/api/model-params', methods=['POST'])
def save_model_params():
    """Save model parameters"""
    try:
        data = request.json
        model = data.get('model', 'default')
        saved_model_params[model] = data.get('params', {})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-configs', methods=['GET'])
def get_saved_configs():
    """Get saved configurations"""
    return jsonify({"configurations": saved_configurations})

@app.route('/api/saved-configs', methods=['POST'])
def create_saved_config():
    """Create saved configuration"""
    global config_id_counter
    
    try:
        data = request.json
        config = {
            "id": config_id_counter,
            "name": data.get('name'),
            "params": data.get('params'),
            "created_at": datetime.now().isoformat()
        }
        saved_configurations.append(config)
        config_id_counter += 1
        return jsonify({"status": "success", "configuration": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-configs/<int:config_id>', methods=['DELETE'])
def delete_saved_config(config_id):
    """Delete saved configuration"""
    global saved_configurations
    saved_configurations = [c for c in saved_configurations if c['id'] != config_id]
    return jsonify({"status": "success"})

if __name__ == '__main__':
    logger.info("Starting Simplified Local AI Stack")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Ollama: {OLLAMA_HOST}")
    logger.info(f"MCP Server: {MCP_SERVER_URL}")
    logger.info(f"Vector DB: {VECTOR_DB_URL}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
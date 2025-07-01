from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import json
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:3000')
EMBEDDING_SERVICE_URL = os.getenv('EMBEDDING_SERVICE_URL', 'http://localhost:8080')

# System files that should always be included as context
SYSTEM_FILES = [
    'admin', 'system', 'default', 'config'
]

# In-memory storage (in production, use a database)
chat_history = []
saved_model_params = {}
saved_configurations = []
config_id_counter = 1

# Default parameter presets
DEFAULT_PRESETS = {
    "creative": {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "repeat_penalty": 1.0,
        "seed": -1,
        "num_predict": -1
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": -1,
        "num_predict": -1
    },
    "precise": {
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 20,
        "repeat_penalty": 1.2,
        "seed": -1,
        "num_predict": -1
    }
}

class VectorizedRAGPipeline:
    """Complete RAG pipeline with local embedding support"""
    
    def __init__(self, ollama_host, mcp_server_url, embedding_service_url):
        self.ollama_host = ollama_host
        self.mcp_server_url = mcp_server_url
        self.embedding_service_url = embedding_service_url
        
    def process_query(self, model, user_message, model_params, conversation_history=None, fast_mode=True):
        """Main entry point for processing queries with vectorized RAG"""
        
        try:
            if fast_mode:
                return self._fast_response(model, user_message, model_params, conversation_history)
            else:
                return self._detailed_response(model, user_message, model_params, conversation_history)
        except Exception as e:
            logging.error(f"RAG pipeline error: {e}")
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'citations': [],
                'context_chunks_used': 0,
                'search_results': [],
                'processing_mode': 'error',
                'confidence_score': 1,
                'error': str(e)
            }
        
    def _fast_response(self, model, user_message, model_params, conversation_history=None):
        """Fast mode: single pass with semantic search"""
        
        # Get relevant context via semantic search
        relevant_chunks = self._semantic_search(user_message, top_k=3)
        context_content = self._format_search_results(relevant_chunks)
        
        # Format conversation history
        conversation_context = ""
        if conversation_history:
            conversation_context = self._format_conversation_history(conversation_history[-3:])
        
        # Single optimized prompt
        prompt = f"""Answer the user's question using the provided context and conversation history.

{conversation_context}

USER QUESTION: {user_message}

RELEVANT CONTEXT:
{context_content}

INSTRUCTIONS:
1. Provide a clear, direct answer
2. Cite sources using [Source: filename] format
3. If context is insufficient, say so clearly
4. Maintain conversation continuity

Generate a helpful response:"""

        response = self._call_ollama(model, prompt, model_params)
        citations = self._extract_citations_from_search(response, relevant_chunks)
        
        return {
            'response': response,
            'citations': citations,
            'context_chunks_used': len(relevant_chunks),
            'search_results': relevant_chunks,
            'processing_mode': 'fast',
            'confidence_score': self._estimate_confidence(relevant_chunks, user_message),
            'metadata': {
                'processing_mode': 'fast',
                'context_chunks_used': len(relevant_chunks),
                'confidence_score': self._estimate_confidence(relevant_chunks, user_message),
                'search_results': relevant_chunks,
                'reasoning_pattern': 'Local-RAG'
            }
        }
    
    def _detailed_response(self, model, user_message, model_params, conversation_history=None):
        """Detailed mode: analysis + response with enhanced reasoning"""
        
        # Step 1: Analyze query and search
        analysis = self._analyze_query(model, user_message, model_params)
        relevant_chunks = self._semantic_search(user_message, top_k=5)
        
        # Step 2: Generate response with analysis
        context_content = self._format_search_results(relevant_chunks)
        conversation_context = ""
        if conversation_history:
            conversation_context = self._format_conversation_history(conversation_history[-5:])
        
        # Enhanced detailed prompt
        prompt = f"""Provide a comprehensive response based on analysis and available context.

QUERY ANALYSIS: {analysis}

{conversation_context}

USER QUESTION: {user_message}

RELEVANT CONTEXT:
{context_content}

RESPONSE REQUIREMENTS:
1. Address the query comprehensively
2. Reference specific context chunks using [Source: filename]
3. Explain your reasoning process
4. Assess confidence in your answer (1-10)
5. Note any limitations or gaps

Generate detailed response:"""

        response = self._call_ollama(model, prompt, model_params)
        citations = self._extract_citations_from_search(response, relevant_chunks)
        confidence = self._extract_confidence_from_response(response)
        
        # Generate enhanced reasoning chain
        reasoning_chain = self._generate_reasoning_chain(user_message, relevant_chunks, analysis)
        
        return {
            'response': response,
            'citations': citations,
            'analysis': analysis,
            'context_chunks_used': len(relevant_chunks),
            'search_results': relevant_chunks,
            'processing_mode': 'detailed',
            'confidence_score': confidence,
            'metadata': {
                'processing_mode': 'detailed',
                'context_chunks_used': len(relevant_chunks),
                'confidence_score': confidence,
                'search_results': relevant_chunks,
                'reasoning_pattern': 'Enhanced-Local-RAG',
                'reasoning_chain': reasoning_chain,
                'confidence_breakdown': self._generate_confidence_breakdown(confidence, relevant_chunks)
            }
        }
    
    def _semantic_search(self, query, top_k=3):
        """Enhanced semantic search with local embedding validation"""
        try:
            # First, check if embedding service is using local embeddings
            health_response = requests.get(f"{self.embedding_service_url}/health", timeout=5)
            embedding_info = {}
            if health_response.ok:
                health_data = health_response.json()
                embedding_info = {
                    'model': health_data.get('model', 'unknown'),
                    'model_type': health_data.get('model_type', 'unknown'),
                    'is_local': health_data.get('model_type') == 'local'
                }
                logging.info(f"Using embedding model: {health_data.get('model', 'unknown')} "
                           f"({health_data.get('model_type', 'unknown')})")
            
            # Perform search via MCP server
            response = requests.post(f"{self.mcp_server_url}/search", 
                json={
                    'query': query,
                    'topK': top_k,
                    'minSimilarity': 0.2  # Lower threshold for local embeddings
                },
                timeout=15  # Increased timeout for local processing
            )
            
            if response.ok:
                data = response.json()
                results = data.get('results', [])
                search_stats = data.get('searchStats', {})
                
                # Log search performance
                logging.info(f"Search completed: {len(results)} results, "
                           f"avg similarity: {search_stats.get('averageSimilarity', 0):.3f}, "
                           f"time: {search_stats.get('totalTimeMs', 0)}ms")
                
                # Enhance results with local embedding metadata
                for result in results:
                    result['embedding_type'] = 'local' if embedding_info.get('is_local') else 'external'
                    result['model_type'] = embedding_info.get('model_type', 'unknown')
                    result['embedding_model'] = embedding_info.get('model', 'unknown')
                
                return results
            else:
                logging.error(f"Search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logging.error(f"Semantic search error: {e}")
            return []
    
    def _estimate_confidence(self, search_results, query):
        """Enhanced confidence estimation for local embeddings"""
        if not search_results:
            return 3
        
        avg_similarity = sum(r.get('similarity', 0) for r in search_results) / len(search_results)
        result_count = len(search_results)
        
        # Adjusted thresholds for local embeddings (typically lower than transformer models)
        base_confidence = 5
        
        # Similarity contribution (adjusted for local embeddings)
        if avg_similarity > 0.6:
            similarity_bonus = 3
        elif avg_similarity > 0.4:
            similarity_bonus = 2
        elif avg_similarity > 0.25:
            similarity_bonus = 1
        else:
            similarity_bonus = 0
        
        # Result count contribution
        if result_count >= 5:
            count_bonus = 1
        elif result_count >= 3:
            count_bonus = 0.5
        else:
            count_bonus = 0
        
        # Query complexity penalty (longer queries might be harder to match)
        query_length = len(query.split())
        if query_length > 15:
            complexity_penalty = -0.5
        elif query_length > 25:
            complexity_penalty = -1
        else:
            complexity_penalty = 0
        
        final_confidence = base_confidence + similarity_bonus + count_bonus + complexity_penalty
        
        return max(1, min(10, round(final_confidence, 1)))
    
    def _format_search_results(self, search_results):
        """Enhanced formatting with local embedding metadata"""
        if not search_results:
            return "No relevant context found in local knowledge base."
        
        context = ""
        total_similarity = sum(r.get('similarity', 0) for r in search_results)
        avg_similarity = total_similarity / len(search_results) if search_results else 0
        
        # Check if using local embeddings
        is_local = search_results[0].get('embedding_type') == 'local' if search_results else False
        embedding_type = "Local Embeddings" if is_local else "Standard Embeddings"
        
        context += f"\n--- VECTORIZED SEARCH RESULTS ({embedding_type}) ---\n"
        context += f"Found {len(search_results)} relevant chunks with average similarity: {avg_similarity:.3f}\n"
        if is_local:
            context += "🏠 Using local embeddings - complete privacy maintained\n"
        context += "\n"
        
        for i, result in enumerate(search_results, 1):
            similarity = result.get('similarity', 0)
            filename = result.get('filename', 'unknown')
            chunk = result.get('chunk', '')
            
            # Quality indicator based on similarity
            if similarity > 0.5:
                quality = "HIGH"
            elif similarity > 0.3:
                quality = "MEDIUM"
            else:
                quality = "LOW"
            
            context += f"--- [CHUNK {i}] {filename} | Similarity: {similarity:.3f} | Quality: {quality} ---\n"
            context += chunk + "\n\n"
        
        context += f"--- END SEARCH RESULTS ---\n"
        return context
    
    def _analyze_query(self, model, user_message, model_params):
        """Quick analysis of user query"""
        analysis_prompt = f"""Analyze this query briefly:

QUERY: {user_message}

Provide a 2-3 sentence analysis covering:
- Query type (factual, analytical, creative, etc.)
- Key concepts to search for
- Expected answer complexity

Analysis:"""
        
        # Use lower temperature for analysis
        analysis_params = model_params.copy()
        analysis_params['temperature'] = 0.3
        
        return self._call_ollama(model, analysis_prompt, analysis_params)
    
    def _extract_confidence_from_response(self, response):
        """Extract confidence score from detailed response"""
        import re
        confidence_match = re.search(r'confidence[:\s]+(\d+)', response.lower())
        if confidence_match:
            return int(confidence_match.group(1))
        return 7
    
    def _extract_citations_from_search(self, response, search_results):
        """Extract citations based on search results"""
        citations = []
        filenames = set(result.get('filename', '') for result in search_results)
        
        for filename in filenames:
            if filename and (f"[Source: {filename}]" in response or filename in response):
                citations.append({
                    'file': filename,
                    'type': 'SYSTEM' if self._is_system_file(filename) else 'USER',
                    'relevance': self._get_file_relevance(filename, search_results)
                })
        
        return citations
    
    def _get_file_relevance(self, filename, search_results):
        """Get average relevance score for a file"""
        file_results = [r for r in search_results if r.get('filename') == filename]
        if file_results:
            avg_similarity = sum(r.get('similarity', 0) for r in file_results) / len(file_results)
            return f"{avg_similarity:.3f}"
        return "0.000"
    
    def _format_conversation_history(self, history):
        """Format conversation history efficiently"""
        if not history:
            return ""
        
        formatted = "\n--- RECENT CONVERSATION ---\n"
        for entry in history:
            formatted += f"USER: {entry['message']}\n"
            formatted += f"ASSISTANT: {entry['response'][:200]}...\n\n"
        
        return formatted + "--- END CONVERSATION ---\n"
    
    def _generate_reasoning_chain(self, query, search_results, analysis):
        """Generate a reasoning chain for detailed mode"""
        return [
            {
                'stage': 'decomposition',
                'query_type': 'analytical',
                'complexity': 'moderate',
                'components': len(query.split())
            },
            {
                'stage': 'evidence_gathering',
                'ranked_sources': len(search_results),
                'overall_evidence_quality': 'good' if search_results else 'limited'
            },
            {
                'stage': 'pattern_identification',
                'pattern_type': 'local-rag',
                'reasoning_steps': 3
            },
            {
                'stage': 'hypothesis_formation',
                'candidate_approaches': 1,
                'primary_approach': 'semantic_search_synthesis'
            },
            {
                'stage': 'verification',
                'logical_consistency': {'score': 8},
                'completeness_assessment': {'information_sufficiency': 'adequate'}
            },
            {
                'stage': 'synthesis',
                'strategy': 'comprehensive',
                'citation_targets': len(search_results)
            }
        ]
    
    def _generate_confidence_breakdown(self, confidence, search_results):
        """Generate confidence factor breakdown"""
        return {
            'context_quality': min(confidence * 0.8, 10),
            'information_completeness': min(confidence * 0.9, 10),
            'reasoning_validation': min(confidence * 0.85, 10),
            'source_reliability': min(len(search_results) * 2, 10)
        }
    
    def _call_ollama(self, model, prompt, model_params):
        """Make optimized API call to Ollama"""
        options = self._prepare_model_options(model_params)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", 
                                   json=payload, timeout=60)
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return "Error: Unable to generate response"
    
    def _prepare_model_options(self, model_params):
        """Convert model parameters to Ollama options"""
        options = {}
        
        if 'temperature' in model_params:
            options['temperature'] = float(model_params['temperature'])
        if 'top_p' in model_params:
            options['top_p'] = float(model_params['top_p'])
        if 'top_k' in model_params:
            options['top_k'] = int(model_params['top_k'])
        if 'repeat_penalty' in model_params:
            options['repeat_penalty'] = float(model_params['repeat_penalty'])
        if 'seed' in model_params and model_params['seed'] != -1:
            options['seed'] = int(model_params['seed'])
        if 'num_predict' in model_params and model_params['num_predict'] != -1:
            options['num_predict'] = int(model_params['num_predict'])
            
        return options
    
    def _is_system_file(self, filename):
        """Check if file is a system file"""
        system_keywords = ['admin', 'system', 'default', 'config']
        return any(keyword in filename.lower() for keyword in system_keywords)

# Initialize enhanced RAG pipeline
rag_pipeline = VectorizedRAGPipeline(OLLAMA_HOST, MCP_SERVER_URL, EMBEDDING_SERVICE_URL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-params', methods=['GET'])
def get_model_params():
    """Get saved model parameters for the current session"""
    model = request.args.get('model', 'default')
    
    if model in saved_model_params:
        return jsonify({
            "status": "success",
            "params": saved_model_params[model],
            "timestamp": saved_model_params[model].get('saved_at')
        })
    else:
        # Return default balanced preset
        return jsonify({
            "status": "default",
            "params": DEFAULT_PRESETS["balanced"],
            "timestamp": None
        })

@app.route('/api/model-params', methods=['POST'])
def save_model_params():
    """Save model parameters for a specific model"""
    data = request.json
    model = data.get('model', 'default')
    params = data.get('params', {})
    
    # Validate parameters
    required_params = ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'num_predict']
    if not all(param in params for param in required_params):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Save parameters with timestamp
    saved_model_params[model] = {
        **params,
        'saved_at': datetime.now().isoformat(),
        'model': model
    }
    
    return jsonify({
        "status": "success",
        "message": f"Parameters saved for model {model}",
        "timestamp": saved_model_params[model]['saved_at']
    })

@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get available parameter presets"""
    return jsonify({
        "presets": DEFAULT_PRESETS,
        "default": "balanced"
    })

@app.route('/api/presets/<preset_name>', methods=['GET'])
def get_preset(preset_name):
    """Get specific preset parameters"""
    if preset_name in DEFAULT_PRESETS:
        return jsonify({
            "status": "success",
            "preset": preset_name,
            "params": DEFAULT_PRESETS[preset_name]
        })
    else:
        return jsonify({"error": "Preset not found"}), 404

def is_system_file(filename):
    """Check if a file should be treated as a system file"""
    filename_lower = filename.lower()
    return any(keyword in filename_lower for keyword in SYSTEM_FILES)

def get_all_context_files():
    """Get all available context files with system file identification"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/files")
        if response.ok:
            files = response.json()
            return files
        return []
    except:
        return []

@app.route('/api/chat', methods=['POST'])
def vectorized_chat():
    """Enhanced chat endpoint with local embedding support"""
    data = request.json
    model = data.get('model', 'llama2')
    message = data.get('message', '')
    model_params = data.get('model_params', {})
    conversation_id = data.get('conversation_id', 'default')
    fast_mode = data.get('fast_mode', True)  # Default to fast mode
    
    try:
        # Get conversation history
        conversation_messages = [
            entry for entry in chat_history 
            if entry.get('conversation_id', 'default') == conversation_id
        ]
        conversation_history = conversation_messages[-5:]  # Last 5 for context
        
        # Use vectorized RAG pipeline
        result = rag_pipeline.process_query(
            model, 
            message, 
            model_params,
            conversation_history=conversation_history,
            fast_mode=fast_mode
        )
        
        # Store chat history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "model": model,
            "message": message,
            "response": result['response'],
            "citations": result['citations'],
            "context_chunks_used": result['context_chunks_used'],
            "processing_mode": result['processing_mode'],
            "confidence_score": result['confidence_score'],
            "model_params": model_params,
            "metadata": result.get('metadata', {})
        }
        
        chat_history.append(chat_entry)
        
        return jsonify({
            'response': result['response'],
            'citations': result['citations'],
            'metadata': result.get('metadata', {
                'processing_mode': result['processing_mode'],
                'context_chunks_used': result['context_chunks_used'],
                'confidence_score': result['confidence_score'],
                'search_results': result.get('search_results', [])
            }),
            'conversation_id': conversation_id,
            **chat_entry
        })
        
    except Exception as e:
        logging.error(f"Vectorized chat error: {e}")
        return jsonify({"error": f"Chat processing error: {str(e)}"}), 500
    
@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history with optional filtering"""
    model_filter = request.args.get('model')
    conversation_filter = request.args.get('conversation_id')
    limit = request.args.get('limit', type=int)
    
    filtered_history = chat_history
    
    if model_filter:
        filtered_history = [h for h in filtered_history if h.get('model') == model_filter]
    
    if conversation_filter:
        filtered_history = [h for h in filtered_history if h.get('conversation_id', 'default') == conversation_filter]
    
    if limit:
        filtered_history = filtered_history[-limit:]
    
    return jsonify(filtered_history)

@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared successfully"})

@app.route('/api/chat/conversations', methods=['GET'])
def get_conversations():
    """Get list of conversation sessions"""
    conversations = {}
    for entry in chat_history:
        conv_id = entry.get('conversation_id', 'default')
        if conv_id not in conversations:
            conversations[conv_id] = {
                'id': conv_id,
                'last_activity': entry['timestamp'],
                'message_count': 0,
                'first_message': entry['message'][:50] + '...'
            }
        conversations[conv_id]['message_count'] += 1
        if entry['timestamp'] > conversations[conv_id]['last_activity']:
            conversations[conv_id]['last_activity'] = entry['timestamp']
    
    return jsonify(list(conversations.values()))

@app.route('/api/chat/conversations/<conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get history for a specific conversation"""
    conversation_messages = [
        entry for entry in chat_history 
        if entry.get('conversation_id', 'default') == conversation_id
    ]
    return jsonify(conversation_messages)

@app.route('/api/chat/conversations/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """Clear specific conversation history"""
    global chat_history
    chat_history = [
        entry for entry in chat_history 
        if entry.get('conversation_id', 'default') != conversation_id
    ]
    return jsonify({"message": f"Conversation {conversation_id} cleared successfully"})

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        response = requests.get(f"{MCP_SERVER_URL}/files")
        files = response.json()
        
        # Add metadata about system files
        if isinstance(files, list):
            file_info = []
            for file in files:
                file_info.append({
                    "filename": file,
                    "is_system": is_system_file(file)
                })
            return jsonify(files)  # Keep backward compatibility
        
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/files', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Forward file to MCP server
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{MCP_SERVER_URL}/files", files=files)
        
        if response.ok:
            result_data = response.json()
            is_system = is_system_file(file.filename)
            
            # Enhance response with local embedding info
            enhanced_response = {
                "message": "File uploaded and vectorized successfully", 
                "filename": file.filename,
                "is_system": is_system,
                **result_data  # Include all MCP server response data
            }
            
            return jsonify(enhanced_response)
        else:
            return jsonify({"error": "Failed to upload to MCP server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    # Prevent deletion of system files
    if is_system_file(filename):
        return jsonify({"error": "Cannot delete system files"}), 403
    
    try:
        response = requests.delete(f"{MCP_SERVER_URL}/files/{filename}")
        if response.ok:
            return jsonify({"message": "File deleted successfully"})
        else:
            return jsonify({"error": "Failed to delete file"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp/status', methods=['GET'])
def mcp_status():
    try:
        response = requests.get(f"{MCP_SERVER_URL}/status")
        return jsonify(response.json())
    except:
        return jsonify({"status": "offline"}), 503

# New endpoint for embedding service monitoring
@app.route('/api/embedding/health', methods=['GET'])
def embedding_service_health():
    """Get embedding service health and info"""
    try:
        response = requests.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=5)
        
        if response.ok:
            data = response.json()
            return jsonify({
                "status": "online",
                "embedding_service": data,
                "local_embedding": data.get('model_type') == 'local',
                "fully_local": 'none' in data.get('dependencies', '').lower()
            })
        else:
            return jsonify({"status": "offline", "error": f"HTTP {response.status_code}"}), 503
            
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e)}), 503

@app.route('/api/embedding/stats', methods=['GET'])
def embedding_service_stats():
    """Get detailed embedding service statistics"""
    try:
        response = requests.get(f"{EMBEDDING_SERVICE_URL}/stats", timeout=5)
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"HTTP {response.status_code}"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 503

@app.route('/api/vectors/stats', methods=['GET'])
def vector_statistics():
    """Get vector store statistics from MCP server"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/vectors/stats", timeout=5)
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"HTTP {response.status_code}"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 503

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Enhanced system information including RAG pipeline status"""
    try:
        all_files = get_all_context_files()
        system_files = [f for f in all_files if is_system_file(f)]
        user_files = [f for f in all_files if not is_system_file(f)]
        
        return jsonify({
            "total_files": len(all_files),
            "system_files": system_files,
            "user_files": user_files,
            "system_file_keywords": SYSTEM_FILES,
            "saved_param_configs": len(saved_model_params),
            "available_presets": list(DEFAULT_PRESETS.keys()),
            "rag_pipeline": {
                "vectorized_search_enabled": True,
                "local_embedding_support": True,
                "two_stage_processing": True,
                "citation_tracking": True,
                "confidence_scoring": True,
                "conversation_context": True
            },
            "embedding_service_url": EMBEDDING_SERVICE_URL
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-configs', methods=['GET'])
def get_saved_configurations():
    """Retrieve all saved parameter configurations"""
    try:
        # Sort configurations by creation time (most recent first)
        sorted_configs = sorted(saved_configurations, 
                               key=lambda x: x.get('created_at', ''), 
                               reverse=True)
        
        return jsonify({
            "status": "success",
            "configurations": sorted_configs,
            "total_count": len(sorted_configs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-configs', methods=['POST'])
def create_saved_configuration():
    """Create a new named parameter configuration"""
    global config_id_counter
    
    data = request.json
    name = data.get('name', '').strip()
    params = data.get('params', {})
    model = data.get('model', 'default')
    
    # Validate input parameters
    if not name:
        return jsonify({"error": "Configuration name is required"}), 400
    
    if len(name) > 50:
        return jsonify({"error": "Configuration name must be 50 characters or less"}), 400
    
    # Check for duplicate names
    if any(config['name'].lower() == name.lower() for config in saved_configurations):
        return jsonify({"error": "A configuration with this name already exists"}), 400
    
    # Validate required parameters
    required_params = ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'num_predict']
    if not all(param in params for param in required_params):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Create new configuration entry
    new_config = {
        "id": config_id_counter,
        "name": name,
        "params": params,
        "model": model,
        "created_at": datetime.now().isoformat(),
        "last_used": None,
        "usage_count": 0
    }
    
    saved_configurations.append(new_config)
    config_id_counter += 1
    
    return jsonify({
        "status": "success",
        "message": f"Configuration '{name}' saved successfully",
        "configuration": new_config
    })

@app.route('/api/saved-configs/<int:config_id>', methods=['DELETE'])
def delete_saved_configuration(config_id):
    """Delete a specific saved configuration"""
    global saved_configurations
    
    # Find configuration by ID
    config_to_delete = None
    config_index = None
    
    for index, config in enumerate(saved_configurations):
        if config['id'] == config_id:
            config_to_delete = config
            config_index = index
            break
    
    if not config_to_delete:
        return jsonify({"error": "Configuration not found"}), 404
    
    # Remove configuration from storage
    saved_configurations.pop(config_index)
    
    return jsonify({
        "status": "success",
        "message": f"Configuration '{config_to_delete['name']}' deleted successfully"
    })

@app.route('/api/saved-configs/<int:config_id>/apply', methods=['POST'])
def apply_saved_configuration(config_id):
    """Apply a saved configuration and update usage statistics"""
    # Find configuration by ID
    config_to_apply = None
    
    for config in saved_configurations:
        if config['id'] == config_id:
            config_to_apply = config
            break
    
    if not config_to_apply:
        return jsonify({"error": "Configuration not found"}), 404
    
    # Update usage statistics
    config_to_apply['last_used'] = datetime.now().isoformat()
    config_to_apply['usage_count'] = config_to_apply.get('usage_count', 0) + 1
    
    return jsonify({
        "status": "success",
        "message": f"Configuration '{config_to_apply['name']}' applied successfully",
        "configuration": config_to_apply
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
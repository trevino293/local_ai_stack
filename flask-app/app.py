# flask-app/app.py - Fixed version without embedding proxy
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

# System files that should always be included as context
SYSTEM_FILES = [
    'admin', 'system', 'default', 'config', 'haag'
]

# In-memory storage
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedVectorRAGPipeline:
    """Simplified RAG pipeline using Qdrant's built-in embeddings via MCP server"""
    
    def __init__(self, ollama_host, mcp_server_url, vector_db_url):
        self.ollama_host = ollama_host
        self.mcp_server_url = mcp_server_url
        self.vector_db_url = vector_db_url
        
        logger.info(f"Initialized Simplified RAG Pipeline:")
        logger.info(f"  Ollama: {self.ollama_host}")
        logger.info(f"  MCP Server: {self.mcp_server_url}")
        logger.info(f"  Vector DB: {self.vector_db_url}")
        
    def process_query(self, model, user_message, model_params, conversation_history=None, fast_mode=True):
        """Process queries using MCP server's search functionality"""
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{user_message[:50]}...' (fast_mode={fast_mode})")
            
            # Search for relevant chunks via MCP server
            search_options = {
                'topK': 3 if fast_mode else 5,
                'minSimilarity': 0.3
            }
            
            relevant_chunks = self._semantic_search(user_message, **search_options)
            
            # Format context and generate response
            context_content = self._format_search_results(relevant_chunks, mode='fast' if fast_mode else 'detailed')
            
            # Format conversation history
            conversation_context = ""
            if conversation_history:
                conversation_context = self._format_conversation_history(conversation_history[-3:] if fast_mode else conversation_history[-5:])
            
            # Generate prompt
            prompt = self._create_prompt(user_message, context_content, conversation_context, fast_mode)
            
            # Call Ollama
            response = self._call_ollama(model, prompt, model_params)
            
            # Extract citations and calculate confidence
            citations = self._extract_citations_from_search(response, relevant_chunks)
            confidence = self._estimate_confidence(relevant_chunks, user_message, 'fast' if fast_mode else 'detailed')
            
            processing_time = round((time.time() - start_time) * 1000)
            
            return {
                'response': response,
                'citations': citations,
                'context_chunks_used': len(relevant_chunks),
                'search_results': relevant_chunks,
                'processing_mode': 'fast' if fast_mode else 'detailed',
                'confidence_score': confidence,
                'processing_time_ms': processing_time,
                'metadata': {
                    'processing_mode': 'fast' if fast_mode else 'detailed',
                    'context_chunks_used': len(relevant_chunks),
                    'confidence_score': confidence,
                    'processing_time_ms': processing_time,
                    'vector_database_info': {
                        'type': 'qdrant',
                        'embedding_method': 'built-in'
                    }
                }
            }
            
        except Exception as e:
            processing_time = round((time.time() - start_time) * 1000)
            logger.error(f"RAG pipeline error after {processing_time}ms: {e}")
            
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'citations': [],
                'context_chunks_used': 0,
                'search_results': [],
                'processing_mode': 'error',
                'confidence_score': 1,
                'processing_time_ms': processing_time,
                'error': str(e),
                'metadata': {
                    'processing_mode': 'error',
                    'error_type': type(e).__name__,
                    'processing_time_ms': processing_time
                }
            }
    
    def _semantic_search(self, query, topK=3, minSimilarity=0.3):
        """Search via MCP server"""
        try:
            response = requests.post(f"{self.mcp_server_url}/search", 
                json={
                    'query': query,
                    'topK': topK,
                    'minSimilarity': minSimilarity
                },
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                return data.get('results', [])
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _create_prompt(self, user_message, context_content, conversation_context, fast_mode):
        """Create appropriate prompt based on mode"""
        if fast_mode:
            return f"""Answer the user's question using the provided context.

{conversation_context}

USER QUESTION: {user_message}

RELEVANT CONTEXT:
{context_content}

Instructions: Provide a clear, direct answer based on the context. Cite sources when referencing specific information.

Response:"""
        else:
            return f"""Provide a comprehensive response based on the context.

{conversation_context}

USER QUESTION: {user_message}

DETAILED CONTEXT:
{context_content}

Instructions: 
1. Address the query comprehensively
2. Reference specific sources
3. Explain your reasoning
4. Note any limitations

Detailed Response:"""
    
    def _format_search_results(self, search_results, mode='detailed'):
        """Format search results for context"""
        if not search_results:
            return "No relevant context found."
        
        context = ""
        for i, result in enumerate(search_results, 1):
            similarity = result.get('similarity', 0)
            filename = result.get('filename', 'unknown')
            chunk = result.get('text', result.get('chunk', ''))
            
            context += f"[{i}] {filename} (similarity: {similarity:.3f})\n{chunk}\n\n"
        
        return context
    
    def _format_conversation_history(self, history):
        """Format conversation history"""
        if not history:
            return ""
        
        formatted = "RECENT CONVERSATION:\n"
        for entry in history:
            formatted += f"USER: {entry['message'][:100]}...\n"
            formatted += f"ASSISTANT: {entry['response'][:150]}...\n\n"
        
        return formatted
    
    def _call_ollama(self, model, prompt, model_params):
        """Call Ollama API"""
        options = {
            'temperature': model_params.get('temperature', 0.7),
            'top_p': model_params.get('top_p', 0.9),
            'top_k': model_params.get('top_k', 40),
            'repeat_penalty': model_params.get('repeat_penalty', 1.1)
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", 
                                   json=payload, 
                                   timeout=120)
            
            if response.ok:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: Ollama API returned {response.status_code}"
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: Unable to generate response - {str(e)}"
    
    def _extract_citations_from_search(self, response, search_results):
        """Extract citations from response"""
        citations = []
        filenames = set(result.get('filename', '') for result in search_results)
        
        for filename in filenames:
            if filename and filename.lower() in response.lower():
                citations.append({
                    'file': filename,
                    'type': 'SYSTEM' if self._is_system_file(filename) else 'USER'
                })
        
        return citations
    
    def _estimate_confidence(self, search_results, query, mode='detailed'):
        """Estimate confidence score"""
        if not search_results:
            return 3
        
        avg_similarity = sum(r.get('similarity', 0) for r in search_results) / len(search_results)
        base_confidence = 6 if mode == 'detailed' else 5
        
        if avg_similarity > 0.7:
            similarity_bonus = 3
        elif avg_similarity > 0.5:
            similarity_bonus = 2
        elif avg_similarity > 0.3:
            similarity_bonus = 1
        else:
            similarity_bonus = 0
        
        return min(10, base_confidence + similarity_bonus)
    
    def _is_system_file(self, filename):
        """Check if file is a system file"""
        if not filename:
            return False
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in SYSTEM_FILES)

# Initialize RAG pipeline
rag_pipeline = SimplifiedVectorRAGPipeline(OLLAMA_HOST, MCP_SERVER_URL, VECTOR_DB_URL)

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
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Ollama API error: {response.status_code}"}), 500
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.json
        model = data.get('model', 'llama2')
        message = data.get('message', '')
        model_params = data.get('model_params', {})
        conversation_id = data.get('conversation_id', 'default')
        fast_mode = data.get('fast_mode', True)
        
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get conversation history
        conversation_history = [
            entry for entry in chat_history 
            if entry.get('conversation_id', 'default') == conversation_id
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
            "model": model,
            "message": message,
            "response": result['response'],
            "citations": result['citations'],
            "confidence_score": result['confidence_score'],
            "metadata": result.get('metadata', {})
        }
        
        chat_history.append(chat_entry)
        
        return jsonify({
            'response': result['response'],
            'citations': result['citations'],
            'conversation_id': conversation_id,
            'metadata': result.get('metadata', {})
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

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
        response = requests.post(f"{MCP_SERVER_URL}/files", files=files, timeout=120)
        
        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Failed to upload: {response.status_code}"}), 500
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
            return jsonify({"error": f"Failed to delete: {response.status_code}"}), 500
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
    """Check vector database health"""
    try:
        # Check Qdrant directly
        response = requests.get(f"{VECTOR_DB_URL}/collections/documents", timeout=5)
        
        if response.ok:
            data = response.json()
            return jsonify({
                "status": "online",
                "vector_database": "qdrant",
                "embedding_method": "built-in",
                "points_count": data.get('result', {}).get('points_count', 0),
                "status": data.get('result', {}).get('status', 'unknown')
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
            "version": "3.0.0",
            "timestamp": datetime.now().isoformat()
        },
        "rag_pipeline": {
            "vectorized_search_enabled": True,
            "vector_database": "qdrant",
            "embedding_method": "built-in"
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
    filtered = [h for h in chat_history if h.get('conversation_id', 'default') == conversation_id]
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
    logger.info("Starting Local AI Stack Flask Application")
    logger.info(f"Ollama: {OLLAMA_HOST}")
    logger.info(f"MCP Server: {MCP_SERVER_URL}")
    logger.info(f"Vector DB: {VECTOR_DB_URL}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
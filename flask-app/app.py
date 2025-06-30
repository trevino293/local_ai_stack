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

# System files that should always be included as context
SYSTEM_FILES = [
    'admin', 'system', 'default', 'config', 'haag', 'response-instructions'
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

# Enhanced RAG Pipeline with Deliberation
class EnhancedRAGPipeline:
    def __init__(self, ollama_host, mcp_server_url):
        self.ollama_host = ollama_host
        self.mcp_server_url = mcp_server_url
        
    def deliberate_and_respond(self, model, user_message, context_files, model_params):
        """Two-stage RAG: deliberation then concrete response"""
        
        # Stage 1: Deliberation - analyze context and plan response
        deliberation_result = self._deliberation_stage(model, user_message, context_files, model_params)
        
        # Stage 2: Concrete Response - generate final answer
        final_response = self._response_stage(model, user_message, context_files, deliberation_result, model_params)
        
        return {
            'deliberation': deliberation_result,
            'response': final_response['content'],
            'metadata': {
                'context_files_used': final_response['context_files_used'],
                'reasoning_trace': deliberation_result.get('reasoning_trace', ''),
                'confidence_score': deliberation_result.get('confidence_score', 7),
                'source_citations': final_response['citations']
            }
        }
    
    def _deliberation_stage(self, model, user_message, context_files, model_params):
        """Stage 1: Analyze context and plan response approach"""
        
        context_content = self._load_context_files(context_files)
        
        deliberation_prompt = f"""
You are in DELIBERATION MODE. Analyze the user's question and available context to plan your response.

USER QUESTION: {user_message}

AVAILABLE CONTEXT:
{context_content}

DELIBERATION TASKS:
1. RELEVANCE ANALYSIS: Which context files are most relevant? Why?
2. INFORMATION GAPS: What information is missing to fully answer the question?
3. RESPONSE STRATEGY: What approach will best address the user's needs?
4. CONFIDENCE ASSESSMENT: How confident can you be in your answer (1-10)?
5. CITATION PLAN: Which specific sections should be cited?

Provide a structured analysis in JSON format:
{{
    "relevant_files": ["file1.txt", "file2.md"],
    "key_insights": ["insight1", "insight2"],
    "information_gaps": ["gap1", "gap2"],
    "response_strategy": "detailed explanation of approach",
    "confidence_score": 8,
    "reasoning_trace": "step-by-step thought process",
    "citation_targets": [
        {{"file": "file1.txt", "section": "relevant section", "reason": "supports main claim"}}
    ]
}}
"""

        # Use lower temperature for deliberation (more analytical)
        deliberation_params = model_params.copy()
        deliberation_params['temperature'] = max(0.3, model_params.get('temperature', 0.7) - 0.4)
        
        deliberation_response = self._call_ollama(model, deliberation_prompt, deliberation_params)
        
        try:
            return json.loads(deliberation_response)
        except:
            # Fallback if JSON parsing fails
            return {
                "relevant_files": context_files,
                "key_insights": ["Analysis pending"],
                "information_gaps": [],
                "response_strategy": "Direct response approach",
                "confidence_score": 7,
                "reasoning_trace": deliberation_response,
                "citation_targets": []
            }
    
    def _response_stage(self, model, user_message, context_files, deliberation, model_params):
        """Stage 2: Generate concrete, well-structured response"""
        
        # Load only the most relevant context files identified in deliberation
        relevant_files = deliberation.get('relevant_files', context_files)
        focused_context = self._load_context_files(relevant_files)
        
        response_prompt = f"""
You are now in RESPONSE MODE. Provide a clear, concrete answer to the user's question.

USER QUESTION: {user_message}

DELIBERATION INSIGHTS:
- Response Strategy: {deliberation.get('response_strategy', '')}
- Key Insights: {', '.join(deliberation.get('key_insights', []))}
- Information Gaps: {', '.join(deliberation.get('information_gaps', []))}

RELEVANT CONTEXT:
{focused_context}

RESPONSE REQUIREMENTS:
1. Start with a direct answer to the user's question
2. Provide supporting details from the context
3. Cite specific sources using [File: filename] format
4. Address any information gaps noted in deliberation
5. Be concrete and actionable where possible
6. End with next steps or recommendations if appropriate

Generate a well-structured response that directly addresses the user's needs.
"""

        final_response = self._call_ollama(model, response_prompt, model_params)
        
        # Extract citations and metadata
        citations = self._extract_citations(final_response, relevant_files)
        
        return {
            'content': final_response,
            'context_files_used': relevant_files,
            'citations': citations
        }
    
    def _load_context_files(self, context_files):
        """Load and format context from files"""
        context_content = ""
        
        for filename in context_files:
            try:
                response = requests.get(f"{self.mcp_server_url}/files/{filename}")
                if response.ok:
                    file_type = "SYSTEM" if self._is_system_file(filename) else "USER"
                    context_content += f"\n--- [{file_type} FILE: {filename}] ---\n{response.text}\n"
            except Exception as e:
                logging.error(f"Error loading file {filename}: {e}")
                
        return context_content
    
    def _call_ollama(self, model, prompt, model_params):
        """Make API call to Ollama"""
        options = self._prepare_model_options(model_params)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload)
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
    
    def _extract_citations(self, response_text, context_files):
        """Extract citation information from response"""
        citations = []
        for filename in context_files:
            if f"[File: {filename}]" in response_text or filename in response_text:
                citations.append({
                    'file': filename,
                    'type': 'SYSTEM' if self._is_system_file(filename) else 'USER'
                })
        return citations
    
    def _is_system_file(self, filename):
        """Check if file is a system file"""
        return any(keyword in filename.lower() for keyword in SYSTEM_FILES)

# Initialize enhanced RAG pipeline
rag_pipeline = EnhancedRAGPipeline(OLLAMA_HOST, MCP_SERVER_URL)

@app.route('/')
def index():
    return render_template('index.html')

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
def enhanced_chat():
    """Enhanced chat endpoint with deliberation RAG pipeline"""
    data = request.json
    model = data.get('model', 'llama2')
    message = data.get('message', '')
    context_files = data.get('context_files', [])
    model_params = data.get('model_params', {})
    
    # Get all available files
    all_files = get_all_context_files()
    
    # Always include system files
    system_files = [f for f in all_files if is_system_file(f)]
    
    # Combine user-selected files with system files, removing duplicates
    all_context_files = list(set(context_files + system_files))
    
    try:
        # Use enhanced RAG pipeline
        result = rag_pipeline.deliberate_and_respond(model, message, all_context_files, model_params)
        
        # Store enhanced chat history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "message": message,
            "response": result['response'],
            "deliberation": result['deliberation'],
            "metadata": result['metadata'],
            "context_files": all_context_files,
            "user_selected_files": context_files,
            "system_files": system_files,
            "model_params": model_params,
            "context_file_count": len(all_context_files)
        }
        
        chat_history.append(chat_entry)
        
        return jsonify({
            'response': result['response'],
            'deliberation_summary': {
                'confidence': result['deliberation'].get('confidence_score', 7),
                'strategy': result['deliberation'].get('response_strategy', 'Standard approach'),
                'files_used': result['metadata']['context_files_used']
            },
            'citations': result['metadata']['source_citations'],
            'reasoning_available': True,
            **chat_entry
        })
        
    except Exception as e:
        logging.error(f"Enhanced chat error: {e}")
        return jsonify({"error": f"Enhanced RAG error: {str(e)}"}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history with optional filtering"""
    model_filter = request.args.get('model')
    limit = request.args.get('limit', type=int)
    
    filtered_history = chat_history
    
    if model_filter:
        filtered_history = [h for h in chat_history if h.get('model') == model_filter]
    
    if limit:
        filtered_history = filtered_history[-limit:]
    
    return jsonify(filtered_history)

@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared successfully"})

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
            is_system = is_system_file(file.filename)
            return jsonify({
                "message": "File uploaded successfully", 
                "filename": file.filename,
                "is_system": is_system
            })
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
                "deliberation_enabled": True,
                "two_stage_processing": True,
                "citation_tracking": True,
                "confidence_scoring": True
            }
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
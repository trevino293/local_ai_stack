from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os
import json
from datetime import datetime

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

def prepare_model_options(model_params):
    """Prepare Ollama model options from parameters"""
    options = {}
    
    # Map frontend parameters to Ollama options
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

@app.route('/api/chat', methods=['POST'])
def chat():
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
    
    # Build context from files
    context = ""
    context_file_count = 0
    
    if all_context_files:
        for file in all_context_files:
            try:
                file_response = requests.get(f"{MCP_SERVER_URL}/files/{file}")
                if file_response.ok:
                    file_type = "SYSTEM" if is_system_file(file) else "USER"
                    context += f"[{file_type} FILE: {file}]\n{file_response.text}\n\n"
                    context_file_count += 1
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
    
    # Prepare model options
    options = prepare_model_options(model_params)
    
    try:
        # Use system/instruction format for better context awareness
        if context:
            system_prompt = f"""You are a helpful AI assistant with access to context files. Use the provided file contents to answer questions accurately and comprehensively.

Context Information ({context_file_count} files loaded):
{context}

Instructions:
- Reference specific files when relevant to your response
- Distinguish between system files (administrative/configuration) and user files (project-specific)
- Provide detailed, contextual answers based on the available information
- If asked about something not covered in the context files, clearly state that limitation"""
            
            # Use the chat endpoint with proper formatting and options
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": False
            }
            
            # Add options if provided
            if options:
                payload["options"] = options
                
            response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            
        else:
            # Regular generate endpoint for no context
            payload = {
                "model": model,
                "prompt": message,
                "stream": False
            }
            
            # Add options if provided
            if options:
                payload["options"] = options
                
            response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        
        result = response.json()
        
        # Extract response based on endpoint used
        if 'message' in result:
            response_text = result['message']['content']
        else:
            response_text = result.get('response', '')
        
        # Store in chat history with enhanced metadata
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "message": message,
            "response": response_text,
            "context_files": all_context_files,
            "user_selected_files": context_files,
            "system_files": system_files,
            "model_params": model_params,
            "context_file_count": context_file_count,
            "ollama_options": options
        }
        chat_history.append(chat_entry)
        
        return jsonify(chat_entry)
        
    except Exception as e:
        error_message = f"Error communicating with Ollama: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

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
    """Provide information about system configuration"""
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
            "available_presets": list(DEFAULT_PRESETS.keys())
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
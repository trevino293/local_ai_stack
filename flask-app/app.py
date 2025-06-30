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

# In-memory chat history (in production, use a database)
chat_history = []

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
            "context_file_count": context_file_count
        }
        chat_history.append(chat_entry)
        
        return jsonify(chat_entry)
        
    except Exception as e:
        error_message = f"Error communicating with Ollama: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

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
            "system_file_keywords": SYSTEM_FILES
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
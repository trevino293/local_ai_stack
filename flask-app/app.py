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

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model', 'llama2')
    message = data.get('message', '')
    context_files = data.get('context_files', [])
    
    # Build context from files
    context = ""
    if context_files:
        for file in context_files:
            try:
                file_response = requests.get(f"{MCP_SERVER_URL}/files/{file}")
                if file_response.ok:
                    context += f"File '{file}':\n{file_response.text}\n\n"
            except:
                pass
    
    # Use system/instruction format for better context awareness
    if context:
        # Different format for better context utilization
        system_prompt = f"You are a helpful assistant. Use the following file contents to answer questions:\n\n{context}"
        
        # Try using the chat endpoint with proper formatting
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": False
            }
        )
    else:
        # Regular generate endpoint for no context
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": message,
                "stream": False
            }
        )
    
    # Call Ollama
    try:
        result = response.json()
        
        # Extract response based on endpoint used
        if 'message' in result:
            response_text = result['message']['content']
        else:
            response_text = result.get('response', '')
        
        # Store in chat history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "message": message,
            "response": response_text,
            "context_files": context_files
        }
        chat_history.append(chat_entry)
        
        return jsonify(chat_entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        response = requests.get(f"{MCP_SERVER_URL}/files")
        return jsonify(response.json())
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
            return jsonify({"message": "File uploaded successfully", "filename": file.filename})
        else:
            return jsonify({"error": "Failed to upload to MCP server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        response = requests.delete(f"{MCP_SERVER_URL}/files/{filename}")
        return jsonify({"message": "File deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mcp/status', methods=['GET'])
def mcp_status():
    try:
        response = requests.get(f"{MCP_SERVER_URL}/status")
        return jsonify(response.json())
    except:
        return jsonify({"status": "offline"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
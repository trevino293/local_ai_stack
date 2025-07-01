# simple_embedding_service.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load lightweight model (22MB)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route('/embed', methods=['POST'])
def embed_text():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    try:
        embedding = model.encode(text).tolist()
        return jsonify({
            'embedding': embedding,
            'model': 'all-MiniLM-L6-v2',
            'dimensions': len(embedding)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'all-MiniLM-L6-v2'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
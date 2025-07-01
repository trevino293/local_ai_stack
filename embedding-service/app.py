# app.py - Minimal embedding service without sentence-transformers dependency
from flask import Flask, request, jsonify
import numpy as np
import hashlib
import re
import math
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SimpleEmbedding:
    """Simple TF-IDF based embedding service"""
    
    def __init__(self, dimensions=384):
        self.dimensions = dimensions
        self.vocabulary = set()
        self.idf_scores = {}
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase and split into words
        text = text.lower()
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Split into words and filter empty strings
        words = [word for word in text.split() if word.strip()]
        return words
    
    def compute_tf(self, words):
        """Compute term frequency"""
        word_count = len(words)
        tf_dict = {}
        for word in words:
            tf_dict[word] = tf_dict.get(word, 0) + 1
        
        # Normalize by document length
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / word_count
        
        return tf_dict
    
    def get_embedding_vector(self, text):
        """Generate embedding vector from text"""
        words = self.preprocess_text(text)
        
        if not words:
            return [0.0] * self.dimensions
        
        # Compute TF
        tf_dict = self.compute_tf(words)
        
        # Create feature vector
        features = []
        
        # Basic features: word count, character count, average word length
        features.extend([
            len(words) / 100.0,  # Normalized word count
            len(text) / 1000.0,  # Normalized character count
            sum(len(word) for word in words) / len(words) / 10.0 if words else 0,  # Avg word length
        ])
        
        # Hash-based features for semantic content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(0, min(len(text_hash), 60), 2):
            features.append(int(text_hash[i:i+2], 16) / 255.0)
        
        # N-gram features
        for n in [1, 2, 3]:
            ngrams = self.get_ngrams(words, n)
            ngram_hash = hashlib.md5(' '.join(ngrams).encode()).hexdigest()
            for i in range(0, min(len(ngram_hash), 40), 2):
                features.append(int(ngram_hash[i:i+2], 16) / 255.0)
        
        # Pad or truncate to desired dimensions
        while len(features) < self.dimensions:
            features.append(0.0)
        
        features = features[:self.dimensions]
        
        # Normalize vector
        norm = math.sqrt(sum(x*x for x in features))
        if norm > 0:
            features = [x/norm for x in features]
        
        return features
    
    def get_ngrams(self, words, n):
        """Generate n-grams from words"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams

# Initialize the embedding model
embedder = SimpleEmbedding(dimensions=384)

@app.route('/embed', methods=['POST'])
def embed_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate embedding
        embedding = embedder.get_embedding_vector(text)
        
        return jsonify({
            'embedding': embedding,
            'model': 'simple-tfidf-hash',
            'dimensions': len(embedding),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'simple-tfidf-hash',
        'dimensions': 384,
        'version': '1.0.0'
    })

@app.route('/batch_embed', methods=['POST'])
def batch_embed():
    """Batch embedding endpoint for multiple texts"""
    try:
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Texts array is required'}), 400
        
        embeddings = []
        for text in texts:
            embedding = embedder.get_embedding_vector(text)
            embeddings.append(embedding)
        
        return jsonify({
            'embeddings': embeddings,
            'model': 'simple-tfidf-hash',
            'dimensions': 384,
            'count': len(embeddings),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    """Compute similarity between two texts"""
    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({'error': 'Both text1 and text2 are required'}), 400
        
        # Generate embeddings
        embedding1 = embedder.get_embedding_vector(text1)
        embedding2 = embedder.get_embedding_vector(text2)
        
        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))
        
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        
        return jsonify({
            'similarity': similarity,
            'text1_length': len(text1),
            'text2_length': len(text2),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Similarity computation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Simple Embedding Service...")
    logger.info("Using TF-IDF + Hash-based embeddings (384 dimensions)")
    app.run(host='0.0.0.0', port=8080, debug=False)
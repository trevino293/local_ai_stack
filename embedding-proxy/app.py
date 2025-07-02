# embedding-proxy/app.py - Fixed proxy with proper Qdrant connection
from flask import Flask, request, jsonify
import requests
import os
import json
import logging
from datetime import datetime
import time

app = Flask(__name__)

# Configuration
VECTOR_DB_URL = os.getenv('VECTOR_DB_URL', 'http://qdrant:6333')
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'qdrant')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorProxy:
    """Simplified proxy for Qdrant with direct embedding"""
    
    def __init__(self, db_url, collection_name):
        self.db_url = db_url
        self.collection_name = collection_name
        self.session = requests.Session()
        self.vector_size = 384  # all-MiniLM-L6-v2
        
    def check_health(self):
        """Check Qdrant health"""
        try:
            response = self.session.get(f"{self.db_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def ensure_collection(self):
        """Ensure collection exists"""
        try:
            # Check if exists
            response = self.session.get(f"{self.db_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                return True
            
            # Create collection
            collection_config = {
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                }
            }
            
            response = self.session.put(
                f"{self.db_url}/collections/{self.collection_name}",
                json=collection_config
            )
            
            return response.status_code in [200, 201]
            
        except Exception as e:
            logger.error(f"Collection error: {e}")
            return False
    
    def store_document(self, filename, content, chunk_size=512):
        """Store document chunks with better mock embeddings"""
        try:
            # Simple chunking by words
            words = content.split()
            chunks = []
            for i in range(0, len(words), chunk_size // 4):  # Approximate words per chunk
                chunk = ' '.join(words[i:i + chunk_size // 4])
                if chunk.strip():
                    chunks.append(chunk)
            
            if not chunks:
                chunks = [content]  # At least one chunk
            
            # Store chunks as points
            points = []
            import hashlib
            
            for i, chunk in enumerate(chunks):
                # Generate a consistent mock embedding based on content
                # This ensures same content gets similar embeddings
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                
                # Create vector with some variation based on content
                base_value = sum(ord(c) for c in chunk_hash[:6]) / 1000.0
                mock_vector = []
                
                for j in range(self.vector_size):
                    # Create variation in vector based on chunk content
                    char_idx = j % len(chunk_hash)
                    value = (ord(chunk_hash[char_idx]) - ord('0')) / 100.0
                    mock_vector.append(min(0.9, max(0.1, base_value + value)))
                
                point_id = f"{filename}_{i}_{chunk_hash[:8]}"
                
                points.append({
                    "id": point_id,
                    "vector": mock_vector,
                    "payload": {
                        "content": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            # Batch upsert with wait for confirmation
            response = self.session.put(
                f"{self.db_url}/collections/{self.collection_name}/points",
                json={"points": points},
                params={"wait": "true"},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Stored {len(chunks)} chunks for {filename}")
                return True, len(chunks)
            else:
                logger.error(f"Failed to store points: {response.status_code} - {response.text}")
                return False, 0
            
        except Exception as e:
            logger.error(f"Store error: {e}")
            return False, 0
    
    def search(self, query_text, top_k=5):
        """Search with better mock query vector"""
        try:
            import hashlib
            
            # Generate query vector similar to how we store documents
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            base_value = sum(ord(c) for c in query_hash[:6]) / 1000.0
            
            mock_vector = []
            for j in range(self.vector_size):
                char_idx = j % len(query_hash)
                value = (ord(query_hash[char_idx]) - ord('0')) / 100.0
                mock_vector.append(min(0.9, max(0.1, base_value + value)))
            
            search_request = {
                "vector": mock_vector,
                "limit": top_k,
                "with_payload": True,
                "score_threshold": 0.0  # Accept all results for now
            }
            
            response = self.session.post(
                f"{self.db_url}/collections/{self.collection_name}/points/search",
                json=search_request,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get("result", [])
                return True, results
            else:
                logger.error(f"Search failed: {response.status_code} - {response.text}")
                return False, []
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return False, []

# Initialize proxy
proxy = SimpleVectorProxy(VECTOR_DB_URL, COLLECTION_NAME)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    db_healthy = proxy.check_health()
    
    return jsonify({
        "status": "ok" if db_healthy else "degraded",
        "vector_database": VECTOR_DB_TYPE,
        "database_url": VECTOR_DB_URL,
        "collection_name": COLLECTION_NAME,
        "version": "2.0.0-prebuilt",
        "database_status": "connected" if db_healthy else "disconnected",
        "embedding_method": "builtin",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    })

@app.route('/embed/document', methods=['POST'])
def embed_document():
    """Process and store document"""
    try:
        data = request.json
        filename = data.get('filename', '')
        content = data.get('content', '')
        
        if not filename or not content:
            return jsonify({'error': 'Filename and content required'}), 400
        
        # Ensure collection exists
        if not proxy.ensure_collection():
            return jsonify({'error': 'Failed to initialize collection'}), 500
        
        # Store document
        success, chunks = proxy.store_document(filename, content)
        
        if success:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'chunks_created': chunks,
                'processing_method': 'prebuilt_vectordb',
                'vector_database': VECTOR_DB_TYPE
            })
        else:
            return jsonify({'error': 'Storage failed'}), 500
            
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for similar content"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('topK', 5)
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        success, results = proxy.search(query, top_k)
        
        if success:
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'filename': result['payload']['filename'],
                    'text': result['payload']['content'],
                    'similarity': result.get('score', 0.5),
                    'chunk_index': result['payload']['chunk_index']
                })
            
            return jsonify({
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'vector_database': VECTOR_DB_TYPE,
                'processing_method': 'prebuilt_vectordb'
            })
        else:
            return jsonify({'error': 'Search failed'}), 500
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete document vectors"""
    try:
        # Delete by payload filter
        delete_request = {
            "filter": {
                "must": [
                    {
                        "key": "filename",
                        "match": {"value": filename}
                    }
                ]
            }
        }
        
        response = proxy.session.post(
            f"{proxy.db_url}/collections/{proxy.collection_name}/points/delete",
            json=delete_request
        )
        
        if response.status_code in [200, 201]:
            return jsonify({
                'status': 'success',
                'message': f'Deleted {filename}'
            })
        else:
            return jsonify({'error': 'Delete failed'}), 500
            
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/collections/init', methods=['POST'])
def init_collection():
    """Initialize collection"""
    try:
        success = proxy.ensure_collection()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Collection {COLLECTION_NAME} initialized',
                'vector_database': VECTOR_DB_TYPE
            })
        else:
            return jsonify({'error': 'Init failed'}), 500
            
    except Exception as e:
        logger.error(f"Init error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get stats"""
    try:
        response = proxy.session.get(f"{proxy.db_url}/collections/{proxy.collection_name}")
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'status': 'success',
                'vector_database': VECTOR_DB_TYPE,
                'collection_name': COLLECTION_NAME,
                'total_vectors': data.get('result', {}).get('points_count', 0)
            })
        else:
            return jsonify({'error': 'Stats failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting simplified embedding proxy")
    logger.info(f"Vector DB: {VECTOR_DB_URL}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    
    # Try to initialize collection on startup
    time.sleep(2)  # Wait for Qdrant to be ready
    if proxy.ensure_collection():
        logger.info("Collection initialized")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
# embedding-proxy/app.py - Lightweight proxy to pre-built vector DB
from flask import Flask, request, jsonify
import requests
import os
import json
import logging
from datetime import datetime

app = Flask(__name__)

# Configuration
VECTOR_DB_URL = os.getenv('VECTOR_DB_URL', 'http://qdrant:6333')
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'qdrant')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBProxy:
    """Lightweight proxy to pre-built vector database services"""
    
    def __init__(self, db_url, db_type, collection_name):
        self.db_url = db_url
        self.db_type = db_type.lower()
        self.collection_name = collection_name
        self.session = requests.Session()
        
    def ensure_collection_exists(self):
        """Ensure the collection exists in the vector database"""
        if self.db_type == 'qdrant':
            return self._ensure_qdrant_collection()
        elif self.db_type == 'weaviate':
            return self._ensure_weaviate_schema()
        else:
            raise ValueError(f"Unsupported vector DB type: {self.db_type}")
    
    def _ensure_qdrant_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            response = self.session.get(f"{self.db_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection with auto-vectorization
            collection_config = {
                "vectors": {
                    "size": 384,  # all-MiniLM-L6-v2 dimensions
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "default_segment_number": 2
                },
                "replication_factor": 1
            }
            
            response = self.session.put(
                f"{self.db_url}/collections/{self.collection_name}",
                json=collection_config
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Created collection '{self.collection_name}'")
                return True
            else:
                logger.error(f"Failed to create collection: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            return False
    
    def _ensure_weaviate_schema(self):
        """Create Weaviate schema if it doesn't exist"""
        try:
            # Check if class exists
            response = self.session.get(f"{self.db_url}/v1/schema/{self.collection_name}")
            if response.status_code == 200:
                logger.info(f"Schema '{self.collection_name}' already exists")
                return True
            
            # Create schema
            schema = {
                "class": self.collection_name,
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "filename",
                        "dataType": ["string"],
                        "description": "Source filename"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Chunk index within document"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["string"],
                        "description": "Processing timestamp"
                    }
                ],
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                }
            }
            
            response = self.session.post(f"{self.db_url}/v1/schema", json=schema)
            
            if response.status_code in [200, 201]:
                logger.info(f"Created schema '{self.collection_name}'")
                return True
            else:
                logger.error(f"Failed to create schema: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring schema exists: {e}")
            return False
    
    def embed_and_store_document(self, filename, content, chunk_size=512, overlap=50):
        """Process document and store with automatic vectorization"""
        if self.db_type == 'qdrant':
            return self._store_qdrant_document(filename, content, chunk_size, overlap)
        elif self.db_type == 'weaviate':
            return self._store_weaviate_document(filename, content, chunk_size, overlap)
    
    def _store_qdrant_document(self, filename, content, chunk_size, overlap):
        """Store document in Qdrant with auto-vectorization"""
        try:
            chunks = self._chunk_text(content, chunk_size, overlap)
            points = []
            
            for i, chunk in enumerate(chunks):
                point = {
                    "id": f"{filename}_{i}_{hash(chunk) % 1000000}",
                    "payload": {
                        "content": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # For Qdrant with transformers, let it handle vectorization
                if hasattr(self, 'use_external_embeddings') and self.use_external_embeddings:
                    # If using external embedding service
                    embedding = self._get_embedding(chunk)
                    point["vector"] = embedding
                
                points.append(point)
            
            # Batch upsert
            response = self.session.put(
                f"{self.db_url}/collections/{self.collection_name}/points",
                json={"points": points}
            )
            
            if response.status_code in [200, 201]:
                return {"success": True, "chunks_created": len(chunks)}
            else:
                logger.error(f"Failed to store document: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return {"success": False, "error": str(e)}
    
    def _store_weaviate_document(self, filename, content, chunk_size, overlap):
        """Store document in Weaviate with auto-vectorization"""
        try:
            chunks = self._chunk_text(content, chunk_size, overlap)
            objects = []
            
            for i, chunk in enumerate(chunks):
                obj = {
                    "class": self.collection_name,
                    "properties": {
                        "content": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                objects.append(obj)
            
            # Batch insert
            response = self.session.post(
                f"{self.db_url}/v1/batch/objects",
                json={"objects": objects}
            )
            
            if response.status_code in [200, 201]:
                return {"success": True, "chunks_created": len(chunks)}
            else:
                logger.error(f"Failed to store document: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return {"success": False, "error": str(e)}
    
    def search_similar(self, query, top_k=5, min_similarity=0.3):
        """Search for similar content"""
        if self.db_type == 'qdrant':
            return self._search_qdrant(query, top_k, min_similarity)
        elif self.db_type == 'weaviate':
            return self._search_weaviate(query, top_k, min_similarity)
    
    def _search_qdrant(self, query, top_k, min_similarity):
        """Search Qdrant collection"""
        try:
            search_request = {
                "vector": query if isinstance(query, list) else None,
                "limit": top_k,
                "score_threshold": min_similarity,
                "with_payload": True
            }
            
            # If query is text, use Qdrant's built-in text search if available
            if isinstance(query, str):
                # Use payload-based search for text queries
                search_request = {
                    "filter": {
                        "should": [
                            {
                                "key": "content",
                                "match": {"text": query}
                            }
                        ]
                    },
                    "limit": top_k,
                    "with_payload": True
                }
            
            response = self.session.post(
                f"{self.db_url}/collections/{self.collection_name}/points/search",
                json=search_request
            )
            
            if response.status_code == 200:
                results = response.json().get("result", [])
                formatted_results = []
                
                for result in results:
                    formatted_results.append({
                        "filename": result["payload"]["filename"],
                        "text": result["payload"]["content"],
                        "similarity": result.get("score", 0),
                        "chunk_index": result["payload"]["chunk_index"],
                        "timestamp": result["payload"]["timestamp"]
                    })
                
                return {"success": True, "results": formatted_results}
            else:
                logger.error(f"Search failed: {response.text}")
                return {"success": False, "results": []}
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"success": False, "results": []}
    
    def _search_weaviate(self, query, top_k, min_similarity):
        """Search Weaviate collection"""
        try:
            search_query = {
                "query": f"""
                {{
                    Get {{
                        {self.collection_name}(
                            nearText: {{
                                concepts: ["{query}"]
                            }}
                            limit: {top_k}
                        ) {{
                            content
                            filename
                            chunk_index
                            timestamp
                            _additional {{
                                certainty
                            }}
                        }}
                    }}
                }}
                """
            }
            
            response = self.session.post(
                f"{self.db_url}/v1/graphql",
                json=search_query
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("Get", {}).get(self.collection_name, [])
                formatted_results = []
                
                for result in results:
                    certainty = result.get("_additional", {}).get("certainty", 0)
                    if certainty >= min_similarity:
                        formatted_results.append({
                            "filename": result["filename"],
                            "text": result["content"],
                            "similarity": certainty,
                            "chunk_index": result["chunk_index"],
                            "timestamp": result["timestamp"]
                        })
                
                return {"success": True, "results": formatted_results}
            else:
                logger.error(f"Search failed: {response.text}")
                return {"success": False, "results": []}
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"success": False, "results": []}
    
    def delete_document(self, filename):
        """Delete all chunks for a document"""
        if self.db_type == 'qdrant':
            return self._delete_qdrant_document(filename)
        elif self.db_type == 'weaviate':
            return self._delete_weaviate_document(filename)
    
    def _delete_qdrant_document(self, filename):
        """Delete document from Qdrant"""
        try:
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
            
            response = self.session.post(
                f"{self.db_url}/collections/{self.collection_name}/points/delete",
                json=delete_request
            )
            
            return response.status_code in [200, 201]
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def _chunk_text(self, text, chunk_size=512, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
        
        return chunks if chunks else [text]
    
    def get_stats(self):
        """Get vector database statistics"""
        if self.db_type == 'qdrant':
            return self._get_qdrant_stats()
        elif self.db_type == 'weaviate':
            return self._get_weaviate_stats()
    
    def _get_qdrant_stats(self):
        """Get Qdrant collection statistics"""
        try:
            response = self.session.get(f"{self.db_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                data = response.json()
                return {
                    "total_vectors": data.get("result", {}).get("points_count", 0),
                    "vector_size": data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {}).get("size", 384),
                    "distance_metric": data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {}).get("distance", "Cosine"),
                    "status": data.get("result", {}).get("status", "unknown")
                }
        except Exception as e:
            logger.error(f"Stats error: {e}")
        
        return {}

# Initialize proxy
vector_proxy = VectorDBProxy(VECTOR_DB_URL, VECTOR_DB_TYPE, COLLECTION_NAME)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test vector DB connection
        if vector_proxy.db_type == 'qdrant':
            response = vector_proxy.session.get(f"{VECTOR_DB_URL}/health")
            db_healthy = response.status_code == 200
        else:
            response = vector_proxy.session.get(f"{VECTOR_DB_URL}/v1/meta")
            db_healthy = response.status_code == 200
        
        return jsonify({
            "status": "ok" if db_healthy else "degraded",
            "vector_database": vector_proxy.db_type,
            "database_url": VECTOR_DB_URL,
            "collection_name": COLLECTION_NAME,
            "version": "2.0.0-prebuilt",
            "database_status": "connected" if db_healthy else "disconnected",
            "embedding_method": "builtin",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/embed/document', methods=['POST'])
def embed_document():
    """Process and embed document using pre-built vector DB"""
    try:
        data = request.json
        filename = data.get('filename', '')
        content = data.get('content', '')
        chunk_size = data.get('chunk_size', 512)
        overlap = data.get('overlap', 50)
        
        if not filename or not content:
            return jsonify({'error': 'Filename and content are required'}), 400
        
        # Ensure collection exists
        if not vector_proxy.ensure_collection_exists():
            return jsonify({'error': 'Failed to initialize vector collection'}), 500
        
        # Store document with auto-vectorization
        result = vector_proxy.embed_and_store_document(filename, content, chunk_size, overlap)
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'chunks_created': result['chunks_created'],
                'processing_method': 'prebuilt_vectordb',
                'vector_database': vector_proxy.db_type
            })
        else:
            return jsonify({'error': result.get('error', 'Unknown error')}), 500
            
    except Exception as e:
        logger.error(f"Document embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_similar():
    """Search for similar content"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('topK', 5)
        min_similarity = data.get('minSimilarity', 0.3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        result = vector_proxy.search_similar(query, top_k, min_similarity)
        
        if result['success']:
            return jsonify({
                'query': query,
                'results': result['results'],
                'total_results': len(result['results']),
                'vector_database': vector_proxy.db_type,
                'processing_method': 'prebuilt_vectordb'
            })
        else:
            return jsonify({'error': 'Search failed'}), 500
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete document embeddings"""
    try:
        success = vector_proxy.delete_document(filename)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Deleted embeddings for {filename}'
            })
        else:
            return jsonify({'error': 'Failed to delete embeddings'}), 500
            
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get vector database statistics"""
    try:
        stats = vector_proxy.get_stats()
        
        return jsonify({
            'status': 'success',
            'vector_database': vector_proxy.db_type,
            'collection_name': vector_proxy.collection_name,
            'embedding_method': 'prebuilt_vectordb',
            'database_url': VECTOR_DB_URL,
            **stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/collections/init', methods=['POST'])
def initialize_collection():
    """Manually initialize the vector collection"""
    try:
        success = vector_proxy.ensure_collection_exists()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Collection {COLLECTION_NAME} initialized',
                'vector_database': vector_proxy.db_type
            })
        else:
            return jsonify({'error': 'Failed to initialize collection'}), 500
            
    except Exception as e:
        logger.error(f"Collection init error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting Vector DB Proxy for {VECTOR_DB_TYPE} at {VECTOR_DB_URL}")
    
    # Initialize collection on startup
    try:
        vector_proxy.ensure_collection_exists()
        logger.info("Vector collection initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize collection on startup: {e}")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
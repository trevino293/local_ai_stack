# embedding-service/app_vectordb.py - Enhanced embedding service with vector database
from flask import Flask, request, jsonify
import numpy as np
import hashlib
import re
import math
import logging
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import requests
from collections import Counter

# Vector Database imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http.models import CollectionStatus
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Qdrant client not available. Install with: pip install qdrant-client")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class VectorDatabaseManager:
    """Unified interface for vector database operations"""
    
    def __init__(self, db_type: str = "qdrant"):
        self.db_type = db_type.lower()
        self.client = None
        self.collection_name = os.getenv('COLLECTION_NAME', 'document_embeddings')
        self.vector_size = 384  # MiniLM dimensions
        
        if self.db_type == "qdrant":
            self._init_qdrant()
        elif self.db_type == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client and collection"""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")
        
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available")
        
        try:
            # For production, use HTTP client
            chroma_host = os.getenv('CHROMA_HOST', 'localhost')
            chroma_port = int(os.getenv('CHROMA_PORT', 8000))
            
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document embeddings for Local AI Stack"}
                )
                logger.info(f"Created ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings in the vector database"""
        try:
            if self.db_type == "qdrant":
                return self._store_qdrant(embeddings_data)
            elif self.db_type == "chroma":
                return self._store_chroma(embeddings_data)
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return False
    
    def _store_qdrant(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings in Qdrant"""
        points = []
        
        for data in embeddings_data:
            point = PointStruct(
                id=data['id'],
                vector=data['embedding'],
                payload={
                    'filename': data['filename'],
                    'chunk_index': data['chunk_index'],
                    'text': data['text'],
                    'timestamp': data['timestamp'],
                    'file_hash': data.get('file_hash', ''),
                    'metadata': data.get('metadata', {})
                }
            )
            points.append(point)
        
        # Batch insert
        batch_size = int(os.getenv('BATCH_SIZE', 32))
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Stored {len(embeddings_data)} embeddings in Qdrant")
        return True
    
    def _store_chroma(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings in ChromaDB"""
        ids = [data['id'] for data in embeddings_data]
        embeddings = [data['embedding'] for data in embeddings_data]
        metadatas = []
        documents = []
        
        for data in embeddings_data:
            metadatas.append({
                'filename': data['filename'],
                'chunk_index': data['chunk_index'],
                'timestamp': data['timestamp'],
                'file_hash': data.get('file_hash', ''),
                **data.get('metadata', {})
            })
            documents.append(data['text'])
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Stored {len(embeddings_data)} embeddings in ChromaDB")
        return True
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      min_similarity: float = 0.3, filters: Dict = None) -> List[Dict]:
        """Search for similar vectors"""
        try:
            if self.db_type == "qdrant":
                return self._search_qdrant(query_embedding, top_k, min_similarity, filters)
            elif self.db_type == "chroma":
                return self._search_chroma(query_embedding, top_k, min_similarity, filters)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_qdrant(self, query_embedding: List[float], top_k: int, 
                      min_similarity: float, filters: Dict) -> List[Dict]:
        """Search similar vectors in Qdrant"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=min_similarity,
            query_filter=filters
        )
        
        results = []
        for hit in search_result:
            results.append({
                'id': hit.id,
                'similarity': hit.score,
                'filename': hit.payload['filename'],
                'chunk_index': hit.payload['chunk_index'],
                'text': hit.payload['text'],
                'timestamp': hit.payload['timestamp'],
                'metadata': hit.payload.get('metadata', {})
            })
        
        return results
    
    def _search_chroma(self, query_embedding: List[float], top_k: int,
                      min_similarity: float, filters: Dict) -> List[Dict]:
        """Search similar vectors in ChromaDB"""
        search_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        results = []
        if search_result['ids'] and search_result['ids'][0]:
            for i, doc_id in enumerate(search_result['ids'][0]):
                distance = search_result['distances'][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    metadata = search_result['metadatas'][0][i]
                    results.append({
                        'id': doc_id,
                        'similarity': similarity,
                        'filename': metadata['filename'],
                        'chunk_index': metadata['chunk_index'],
                        'text': search_result['documents'][0][i],
                        'timestamp': metadata['timestamp'],
                        'metadata': {k: v for k, v in metadata.items() 
                                   if k not in ['filename', 'chunk_index', 'timestamp']}
                    })
        
        return results
    
    def delete_by_filename(self, filename: str) -> bool:
        """Delete all embeddings for a specific file"""
        try:
            if self.db_type == "qdrant":
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                    )
                )
            elif self.db_type == "chroma":
                self.collection.delete(where={"filename": filename})
            
            logger.info(f"Deleted embeddings for file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete embeddings for {filename}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if self.db_type == "qdrant":
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    'total_vectors': collection_info.points_count,
                    'vector_size': collection_info.config.params.vectors.size,
                    'distance_metric': collection_info.config.params.vectors.distance.value,
                    'status': collection_info.status.value
                }
            elif self.db_type == "chroma":
                count = self.collection.count()
                return {
                    'total_vectors': count,
                    'vector_size': self.vector_size,
                    'distance_metric': 'cosine',
                    'status': 'ready'
                }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

class EnhancedEmbeddingService:
    """Enhanced embedding service with proper sentence transformers"""
    
    def __init__(self, dimensions=384):
        self.dimensions = dimensions
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize vector database
        db_type = os.getenv('VECTOR_DB_TYPE', 'qdrant')
        self.vector_db = VectorDatabaseManager(db_type)
        
        # Try to use real sentence transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.use_real_embeddings = True
            logger.info(f"Using real sentence transformers: {self.model_name}")
        except ImportError:
            logger.warning("Sentence transformers not available, using fallback TF-IDF method")
            self.use_real_embeddings = False
            self._init_fallback_embedder()
    
    def _init_fallback_embedder(self):
        """Initialize fallback TF-IDF based embedder"""
        self.vocabulary = set()
        self.idf_scores = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.use_real_embeddings:
            return self._generate_real_embedding(text)
        else:
            return self._generate_fallback_embedding(text)
    
    def _generate_real_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformers"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate embedding using TF-IDF fallback"""
        words = self._preprocess_text(text)
        
        if not words:
            return [0.0] * self.dimensions
        
        # Compute TF
        tf_dict = self._compute_tf(words)
        
        # Create feature vector
        features = []
        
        # Basic features
        features.extend([
            len(words) / 100.0,
            len(text) / 1000.0,
            sum(len(word) for word in words) / len(words) / 10.0 if words else 0,
        ])
        
        # Hash-based features
        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(0, min(len(text_hash), 60), 2):
            features.append(int(text_hash[i:i+2], 16) / 255.0)
        
        # N-gram features
        for n in [1, 2, 3]:
            ngrams = self._get_ngrams(words, n)
            ngram_hash = hashlib.md5(' '.join(ngrams).encode()).hexdigest()
            for i in range(0, min(len(ngram_hash), 40), 2):
                features.append(int(ngram_hash[i:i+2], 16) / 255.0)
        
        # Pad or truncate
        while len(features) < self.dimensions:
            features.append(0.0)
        features = features[:self.dimensions]
        
        # Normalize
        norm = math.sqrt(sum(x*x for x in features))
        if norm > 0:
            features = [x/norm for x in features]
        
        return features
    
    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = [word for word in text.split() if word.strip()]
        return words
    
    def _compute_tf(self, words):
        """Compute term frequency"""
        word_count = len(words)
        tf_dict = {}
        for word in words:
            tf_dict[word] = tf_dict.get(word, 0) + 1
        
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / word_count
        
        return tf_dict
    
    def _get_ngrams(self, words, n):
        """Generate n-grams from words"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def process_document(self, filename: str, content: str, chunk_size: int = 512, 
                        overlap: int = 50) -> List[Dict[str, Any]]:
        """Process document into chunks and generate embeddings"""
        # Create chunks
        chunks = self._chunk_text(content, chunk_size, overlap)
        
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        embeddings_data = []
        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk['text'])
            
            chunk_data = {
                'id': f"{filename}_{i}_{file_hash[:8]}",
                'filename': filename,
                'chunk_index': i,
                'text': chunk['text'],
                'embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'file_hash': file_hash,
                'metadata': {
                    'start_index': chunk['start_index'],
                    'end_index': chunk['end_index'],
                    'chunk_size': len(chunk['text'])
                }
            }
            embeddings_data.append(chunk_data)
        
        return embeddings_data
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append({
                    'text': ' '.join(chunk_words),
                    'start_index': i,
                    'end_index': min(i + chunk_size, len(words))
                })
        
        return chunks

# Initialize embedding service
embedding_service = EnhancedEmbeddingService()

@app.route('/embed', methods=['POST'])
def embed_text():
    """Generate embedding for text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        embedding = embedding_service.generate_embedding(text)
        
        return jsonify({
            'embedding': embedding,
            'model': embedding_service.model_name,
            'dimensions': len(embedding),
            'status': 'success',
            'method': 'real' if embedding_service.use_real_embeddings else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed/document', methods=['POST'])
def embed_document():
    """Process and embed entire document"""
    try:
        data = request.json
        filename = data.get('filename', '')
        content = data.get('content', '')
        chunk_size = data.get('chunk_size', 512)
        overlap = data.get('overlap', 50)
        
        if not filename or not content:
            return jsonify({'error': 'Filename and content are required'}), 400
        
        # Delete existing embeddings for this file
        embedding_service.vector_db.delete_by_filename(filename)
        
        # Process document
        embeddings_data = embedding_service.process_document(filename, content, chunk_size, overlap)
        
        # Store in vector database
        success = embedding_service.vector_db.store_embeddings(embeddings_data)
        
        if success:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'chunks_created': len(embeddings_data),
                'total_tokens': len(content.split()),
                'processing_method': 'real' if embedding_service.use_real_embeddings else 'fallback'
            })
        else:
            return jsonify({'error': 'Failed to store embeddings'}), 500
        
    except Exception as e:
        logger.error(f"Document embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_similar():
    """Search for similar embeddings"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('topK', 5)
        min_similarity = data.get('minSimilarity', 0.3)
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)
        
        # Search vector database
        results = embedding_service.vector_db.search_similar(
            query_embedding, top_k, min_similarity, filters
        )
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': len(results),
            'search_method': 'vector_db',
            'processing_method': 'real' if embedding_service.use_real_embeddings else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_embeddings(filename):
    """Delete all embeddings for a file"""
    try:
        success = embedding_service.vector_db.delete_by_filename(filename)
        
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
        stats = embedding_service.vector_db.get_collection_stats()
        
        return jsonify({
            'status': 'success',
            'vector_database': embedding_service.vector_db.db_type,
            'collection_name': embedding_service.vector_db.collection_name,
            'embedding_method': 'real' if embedding_service.use_real_embeddings else 'fallback',
            'model_name': embedding_service.model_name,
            **stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test vector database connection
        stats = embedding_service.vector_db.get_collection_stats()
        
        return jsonify({
            'status': 'ok',
            'vector_database': embedding_service.vector_db.db_type,
            'embedding_method': 'real' if embedding_service.use_real_embeddings else 'fallback',
            'model': embedding_service.model_name,
            'dimensions': embedding_service.dimensions,
            'version': '2.0.0',
            'database_status': 'connected' if stats else 'disconnected'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced Embedding Service with Vector Database...")
    logger.info(f"Vector Database: {embedding_service.vector_db.db_type}")
    logger.info(f"Embedding Method: {'Real' if embedding_service.use_real_embeddings else 'Fallback'}")
    app.run(host='0.0.0.0', port=8080, debug=False)
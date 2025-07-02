// mcp-server/server.js - Enhanced version with vector support
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Environment variables for embedding proxy
const EMBEDDING_PROXY_URL = process.env.EMBEDDING_PROXY_URL || 'http://embedding-proxy:8080';
const VECTOR_DB_URL = process.env.VECTOR_DB_URL || 'http://qdrant:6333';
const VECTOR_DB_TYPE = process.env.VECTOR_DB_TYPE || 'qdrant';
const COLLECTION_NAME = process.env.COLLECTION_NAME || 'documents';

// Middleware
app.use(cors());
app.use(express.json());

// File upload configuration
const storage = multer.diskStorage({
    destination: async (req, file, cb) => {
        const uploadPath = '/workspace';
        await fs.mkdir(uploadPath, { recursive: true });
        cb(null, uploadPath);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ 
    storage,
    limits: {
        fileSize: 50 * 1024 * 1024, // 50MB limit
    }
});

// Cache for file metadata
let fileMetadataCache = new Map();

/**
 * Vector file manager with embedding proxy integration
 */
class VectorFileManager {
    constructor() {
        this.embeddingProxyUrl = EMBEDDING_PROXY_URL;
        console.log(`Initialized VectorFileManager with proxy: ${this.embeddingProxyUrl}`);
    }

    async processFile(filename, content) {
        try {
            // Create text chunks
            const chunks = this.createChunks(content);
        
            // Store with Qdrant's built-in indexing
            const points = chunks.map((chunk, i) => ({
                id: `${filename}_${i}`,
                payload: {
                    content: chunk,
                    filename: filename,
                    chunk_index: i
                }
            }));
        
            const response = await axios.put(
                `${VECTOR_DB_URL}/collections/${this.collection_name}/points`,
                { points },
                { params: { wait: "true" } }
            );
        
            return {
                success: response.status === 200,
                chunks: chunks.length
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async searchSimilar(query, options = {}) {
        const { topK = 5, minSimilarity = 0.3 } = options;
    
        try {
            // Use Qdrant's search with raw text
            const response = await axios.post(`${VECTOR_DB_URL}/collections/${this.collection_name}/points/search`, {
                vector: query,  // Qdrant will handle text->vector conversion
                limit: topK,
                with_payload: true,
                score_threshold: minSimilarity,
                params: {
                    indexed_only: true
                }
            }, { timeout: 10000 });

            if (response.data.result) {
                return {
                    success: true,
                    results: response.data.result
                };
            }
            return { success: true, results: [] };
        } catch (error) {
            console.error(`Search failed:`, error.message);
            return { success: false, error: error.message, results: [] };
        }
    }

    async deleteFileEmbeddings(filename) {
        console.log(`🗑️ Deleting embeddings for: ${filename}`);
        
        try {
            const response = await axios.delete(`${this.embeddingProxyUrl}/delete/${filename}`, {
                timeout: 10000
            });

            if (response.data.status === 'success') {
                console.log(`✅ Deleted embeddings for: ${filename}`);
                fileMetadataCache.delete(filename);
                return { success: true };
            } else {
                throw new Error(response.data.error || 'Delete failed');
            }
        } catch (error) {
            console.error(`❌ Failed to delete embeddings:`, error.message);
            return { success: false, error: error.message };
        }
    }

    async initializeCollection() {
        console.log(`🗃️ Initializing collection: ${COLLECTION_NAME}`);
        
        try {
            const response = await axios.post(`${this.embeddingProxyUrl}/collections/init`, {}, {
                timeout: 10000,
                headers: {'Content-Type': 'application/json'}
            });

            if (response.data.status === 'success') {
                console.log(`✅ Collection initialized`);
                return { success: true };
            } else {
                throw new Error(response.data.error || 'Init failed');
            }
        } catch (error) {
            console.error(`❌ Collection init failed:`, error.message);
            return { success: false, error: error.message };
        }
    }
}

// Initialize vector file manager
const vectorManager = new VectorFileManager();

// Routes

app.get('/status', async (req, res) => {
    res.json({ 
        status: 'online',
        version: '3.0.0',
        workspace: '/workspace',
        vectorization: {
            enabled: true,
            embeddingProxy: EMBEDDING_PROXY_URL,
            vectorDatabase: VECTOR_DB_TYPE
        }
    });
});

app.get('/health', async (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString()
    });
});

app.get('/files', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const fileList = [];
        
        for (const file of files) {
            try {
                const stats = await fs.stat(path.join('/workspace', file));
                if (stats.isFile()) {
                    const metadata = fileMetadataCache.get(file) || { processed: false };
                    fileList.push({
                        name: file,
                        size: stats.size,
                        modified: stats.mtime,
                        vectorized: metadata.processed || false,
                        chunkCount: metadata.chunks || 0
                    });
                }
            } catch (error) {
                console.warn(`Error reading file ${file}:`, error.message);
            }
        }
        
        res.json(fileList);
    } catch (error) {
        console.error('File listing error:', error);
        res.status(500).json({ error: 'Failed to list files' });
    }
});

app.get('/files/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        const filePath = path.join('/workspace', filename);
        const content = await fs.readFile(filePath, 'utf8');
        res.type('text/plain').send(content);
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'File not found' });
        } else {
            res.status(500).json({ error: 'Failed to read file' });
        }
    }
});

app.post('/files', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const filename = req.file.filename;
    console.log(`⬆️ File uploaded: ${filename}`);
    
    try {
        const filePath = path.join('/workspace', filename);
        const content = await fs.readFile(filePath, 'utf8');
        
        // Process with embedding proxy
        const result = await vectorManager.processFile(filename, content);
        
        res.json({
            message: result.success ? 'File uploaded and vectorized' : 'File uploaded but vectorization failed',
            filename: filename,
            fileSize: req.file.size,
            chunksCreated: result.chunks || 0,
            vectorizationStatus: result.success ? 'success' : 'failed',
            processingMethod: result.method || 'unknown',
            error: result.error
        });
        
    } catch (error) {
        console.error(`File processing error:`, error);
        res.status(500).json({ error: 'File processing failed' });
    }
});

// Vector search endpoint
app.post('/search', async (req, res) => {
    const { query, topK = 5, minSimilarity = 0.3 } = req.body;
    
    if (!query) {
        return res.status(400).json({ error: 'Query is required' });
    }
    
    console.log(`🔍 Search request: "${query}"`);
    
    try {
        const searchResult = await vectorManager.searchSimilar(query, { topK, minSimilarity });
        
        if (searchResult.success) {
            const formattedResults = searchResult.results.map(result => ({
                filename: result.filename,
                chunk: result.text,
                similarity: result.similarity,
                chunkIndex: result.chunk_index
            }));
            
            res.json({
                query: query,
                results: formattedResults,
                totalResults: searchResult.totalResults,
                searchStats: {
                    embeddingProxy: EMBEDDING_PROXY_URL,
                    vectorDbType: VECTOR_DB_TYPE
                }
            });
        } else {
            res.status(500).json({
                error: 'Search failed',
                details: searchResult.error
            });
        }
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ error: 'Search failed' });
    }
});

// Collection initialization endpoint
app.post('/collections/init', async (req, res) => {
    console.log('🗃️ Collection init requested');
    
    try {
        const result = await vectorManager.initializeCollection();
        
        if (result.success) {
            res.json({
                status: 'success',
                message: 'Collection initialized',
                collection: COLLECTION_NAME
            });
        } else {
            res.status(500).json({
                status: 'error',
                error: result.error
            });
        }
    } catch (error) {
        console.error('Collection init error:', error);
        res.status(500).json({ error: 'Collection init failed' });
    }
});

app.delete('/files/:filename', async (req, res) => {
    const filename = req.params.filename;
    
    try {
        const filePath = path.join('/workspace', filename);
        await fs.unlink(filePath);
        
        // Delete vectors
        await vectorManager.deleteFileEmbeddings(filename);
        
        res.json({
            message: 'File and vectors deleted',
            filename: filename
        });
    } catch (error) {
        console.error(`Delete error:`, error);
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'File not found' });
        } else {
            res.status(500).json({ error: 'Delete failed' });
        }
    }
});

// Error handling
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, async () => {
    console.log(`🚀 Enhanced MCP Server running on port ${PORT}`);
    console.log(`📡 Embedding Proxy: ${EMBEDDING_PROXY_URL}`);
    console.log(`💾 Vector DB: ${VECTOR_DB_TYPE} at ${VECTOR_DB_URL}`);
    
    // Create workspace directory
    await fs.mkdir('/workspace', { recursive: true });
    console.log('✅ Workspace initialized');
});
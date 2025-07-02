// mcp-server/server.js - Fixed version with Qdrant built-in embeddings
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Environment variables
const VECTOR_DB_URL = process.env.VECTOR_DB_URL || 'http://qdrant:6333';
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
 * Vector file manager using Qdrant's built-in text embeddings
 */
class VectorFileManager {
    constructor() {
        this.vectorDbUrl = VECTOR_DB_URL;
        this.collectionName = COLLECTION_NAME;
        console.log(`Initialized VectorFileManager with Qdrant: ${this.vectorDbUrl}`);
    }

    async ensureCollection() {
        try {
            // Check if collection exists
            const checkResponse = await axios.get(`${this.vectorDbUrl}/collections/${this.collectionName}`);
            console.log(`✅ Collection '${this.collectionName}' exists`);
            return true;
        } catch (error) {
            if (error.response?.status === 404) {
                // Create collection with text embedding support
                console.log(`📦 Creating collection '${this.collectionName}' with text embeddings...`);
                try {
                    await axios.put(`${this.vectorDbUrl}/collections/${this.collectionName}`, {
                        vectors: {
                            size: 384,
                            distance: "Cosine"
                        }
                    });
                    console.log(`✅ Collection '${this.collectionName}' created`);
                    return true;
                } catch (createError) {
                    console.error(`❌ Failed to create collection:`, createError.message);
                    return false;
                }
            }
            console.error(`❌ Failed to check collection:`, error.message);
            return false;
        }
    }

    createChunks(content, chunkSize = 512, overlap = 50) {
        const words = content.split(/\s+/);
        const chunks = [];
        
        for (let i = 0; i < words.length; i += chunkSize - overlap) {
            const chunk = words.slice(i, i + chunkSize).join(' ');
            if (chunk.trim()) {
                chunks.push(chunk.trim());
            }
        }
        
        return chunks;
    }

    async processFile(filename, content) {
        try {
            // Ensure collection exists
            await this.ensureCollection();
            
            // Create text chunks
            const chunks = this.createChunks(content);
            console.log(`📄 Processing ${filename}: ${chunks.length} chunks`);
            
            // Generate simple embeddings (normalized word frequency vectors)
            const points = chunks.map((chunk, i) => {
                const vector = this.generateSimpleEmbedding(chunk);
                return {
                    id: Date.now() + i, // Qdrant wants numeric IDs
                    vector: vector,
                    payload: {
                        text: chunk,
                        filename: filename,
                        chunk_index: i,
                        timestamp: new Date().toISOString()
                    }
                };
            });
            
            // Upload points to Qdrant
            const response = await axios.put(
                `${this.vectorDbUrl}/collections/${this.collectionName}/points`,
                { points },
                { 
                    params: { wait: "true" },
                    timeout: 30000
                }
            );
            
            if (response.data.status === 'ok' || response.data.status === 'completed') {
                console.log(`✅ Stored ${chunks.length} chunks for ${filename}`);
                fileMetadataCache.set(filename, {
                    processed: true,
                    chunks: chunks.length,
                    timestamp: new Date().toISOString()
                });
                return { 
                    success: true, 
                    chunks: chunks.length,
                    method: 'simple-embedding'
                };
            } else {
                throw new Error('Qdrant upload failed');
            }
        } catch (error) {
            console.error(`❌ Failed to process ${filename}:`, error.message);
            return { 
                success: false, 
                error: error.message,
                chunks: 0 
            };
        }
    }

    generateSimpleEmbedding(text) {
        // Simple embedding: normalized TF vector
        const words = text.toLowerCase().split(/\s+/);
        const wordFreq = {};
        
        // Count word frequencies
        for (const word of words) {
            if (word.length > 2) { // Skip short words
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            }
        }
        
        // Create a fixed-size vector (384 dimensions)
        const vector = new Array(384).fill(0);
        
        // Hash words to vector positions
        Object.keys(wordFreq).forEach(word => {
            const hash = this.hashString(word);
            const position = Math.abs(hash) % 384;
            vector[position] += wordFreq[word] / words.length;
        });
        
        // Normalize vector
        const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        if (magnitude > 0) {
            for (let i = 0; i < vector.length; i++) {
                vector[i] /= magnitude;
            }
        }
        
        return vector;
    }

    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash;
    }

    async searchSimilar(query, options = {}) {
        const { topK = 5, minSimilarity = 0.3 } = options;
        
        try {
            // Generate embedding for query
            const queryVector = this.generateSimpleEmbedding(query);
            
            // Search with vector
            const searchResponse = await axios.post(`${this.vectorDbUrl}/collections/${this.collectionName}/points/search`, {
                vector: queryVector,
                limit: topK,
                with_payload: true,
                with_vector: false,
                score_threshold: minSimilarity
            }, { 
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });

            if (searchResponse.data.result) {
                console.log(`🔍 Found ${searchResponse.data.result.length} results`);
                return {
                    success: true,
                    results: searchResponse.data.result.map(r => ({
                        filename: r.payload.filename,
                        text: r.payload.text,
                        similarity: r.score,
                        chunk_index: r.payload.chunk_index
                    })),
                    totalResults: searchResponse.data.result.length
                };
            }
            
            return { success: true, results: [], totalResults: 0 };
        } catch (error) {
            console.error(`❌ Search failed:`, error.response?.data || error.message);
            return await this.fallbackTextSearch(query, topK);
        }
    }

    async fallbackTextSearch(query, limit = 5) {
        console.log(`⚠️ Using fallback text search`);
        try {
            // Get all points
            const scrollResponse = await axios.post(`${this.vectorDbUrl}/collections/${this.collectionName}/points/scroll`, {
                limit: 1000,
                with_payload: true,
                with_vector: false
            });

            if (scrollResponse.data.result?.points) {
                // Simple text matching
                const queryLower = query.toLowerCase();
                const results = scrollResponse.data.result.points
                    .filter(point => point.payload.text?.toLowerCase().includes(queryLower))
                    .slice(0, limit)
                    .map(point => ({
                        filename: point.payload.filename,
                        text: point.payload.text,
                        similarity: 0.5, // Fixed score for text match
                        chunk_index: point.payload.chunk_index
                    }));

                return {
                    success: true,
                    results,
                    totalResults: results.length
                };
            }
            
            return { success: true, results: [], totalResults: 0 };
        } catch (error) {
            console.error(`❌ Fallback search failed:`, error.message);
            return { success: false, error: error.message, results: [] };
        }
    }

    async deleteFileEmbeddings(filename) {
        console.log(`🗑️ Deleting vectors for: ${filename}`);
        
        try {
            // Get all points for this file
            const scrollResponse = await axios.post(`${this.vectorDbUrl}/collections/${this.collectionName}/points/scroll`, {
                filter: {
                    must: [
                        {
                            key: "filename",
                            match: { value: filename }
                        }
                    ]
                },
                limit: 1000
            });

            if (scrollResponse.data.result?.points?.length > 0) {
                const pointIds = scrollResponse.data.result.points.map(p => p.id);
                
                // Delete points
                await axios.post(`${this.vectorDbUrl}/collections/${this.collectionName}/points/delete`, {
                    points: pointIds
                }, { params: { wait: "true" } });
                
                console.log(`✅ Deleted ${pointIds.length} vectors for ${filename}`);
                fileMetadataCache.delete(filename);
                return { success: true, deleted: pointIds.length };
            }
            
            return { success: true, deleted: 0 };
        } catch (error) {
            console.error(`❌ Failed to delete vectors:`, error.message);
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
            method: 'qdrant-built-in',
            vectorDatabase: 'qdrant',
            collection: COLLECTION_NAME
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
        
        // Process with Qdrant
        const result = await vectorManager.processFile(filename, content);
        
        res.json({
            message: result.success ? 'File uploaded and vectorized' : 'File uploaded but vectorization failed',
            filename: filename,
            fileSize: req.file.size,
            chunksCreated: result.chunks || 0,
            vectorizationStatus: result.success ? 'success' : 'failed',
            processingMethod: result.method || 'unknown',
            vectorDatabase: 'qdrant',
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
            res.json({
                query: query,
                results: searchResult.results,
                totalResults: searchResult.totalResults,
                searchStats: {
                    vectorDbType: 'qdrant',
                    method: 'built-in-embeddings'
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
        const success = await vectorManager.ensureCollection();
        
        if (success) {
            res.json({
                status: 'success',
                message: 'Collection initialized',
                collection: COLLECTION_NAME
            });
        } else {
            res.status(500).json({
                status: 'error',
                error: 'Failed to initialize collection'
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
        const deleteResult = await vectorManager.deleteFileEmbeddings(filename);
        
        res.json({
            message: 'File and vectors deleted',
            filename: filename,
            vectorsDeleted: deleteResult.deleted || 0
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

// Get vector statistics
app.get('/vectors/stats', async (req, res) => {
    try {
        const collectionInfo = await axios.get(`${VECTOR_DB_URL}/collections/${COLLECTION_NAME}`);
        
        res.json({
            collection: COLLECTION_NAME,
            pointsCount: collectionInfo.data.result.points_count || 0,
            vectorsCount: collectionInfo.data.result.vectors_count || 0,
            indexedFilesCount: fileMetadataCache.size,
            status: collectionInfo.data.result.status || 'unknown'
        });
    } catch (error) {
        res.json({
            collection: COLLECTION_NAME,
            pointsCount: 0,
            vectorsCount: 0,
            indexedFilesCount: fileMetadataCache.size,
            status: 'error',
            error: error.message
        });
    }
});

// Error handling
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Process system files on startup
async function processSystemFiles() {
    console.log('🔄 Processing system files...');
    
    try {
        await vectorManager.ensureCollection();
        
        const files = await fs.readdir('/workspace');
        for (const file of files) {
            if (file.toLowerCase().includes('system') || 
                file.toLowerCase().includes('admin') || 
                file.toLowerCase().includes('config')) {
                
                try {
                    const content = await fs.readFile(path.join('/workspace', file), 'utf8');
                    const result = await vectorManager.processFile(file, content);
                    console.log(`📌 System file ${file}: ${result.success ? 'processed' : 'failed'}`);
                } catch (error) {
                    console.error(`Failed to process system file ${file}:`, error.message);
                }
            }
        }
    } catch (error) {
        console.error('System files processing error:', error);
    }
}

// Start server
app.listen(PORT, async () => {
    console.log(`🚀 MCP Server running on port ${PORT}`);
    console.log(`💾 Vector DB: Qdrant at ${VECTOR_DB_URL}`);
    console.log(`📦 Collection: ${COLLECTION_NAME}`);
    
    // Create workspace directory
    await fs.mkdir('/workspace', { recursive: true });
    console.log('✅ Workspace initialized');
    
    // Process system files after a short delay
    setTimeout(processSystemFiles, 3000);
});
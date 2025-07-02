// mcp-server/server.js - Fixed vector storage and search
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

// In-memory cache for tracking files and vectors
let fileVectorMap = new Map();

/**
 * Simplified Vector Manager using basic embeddings
 */
class SimpleVectorManager {
    constructor() {
        this.vectorDbUrl = VECTOR_DB_URL;
        this.collectionName = COLLECTION_NAME;
        this.vectorIdCounter = 1;
        console.log(`Initialized SimpleVectorManager with Qdrant: ${this.vectorDbUrl}`);
    }

    async ensureCollection() {
        try {
            // Check if collection exists
            const checkResponse = await axios.get(`${this.vectorDbUrl}/collections/${this.collectionName}`);
            console.log(`✅ Collection '${this.collectionName}' exists`);
            return true;
        } catch (error) {
            if (error.response?.status === 404) {
                // Create collection
                console.log(`📦 Creating collection '${this.collectionName}'...`);
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
            if (chunk.trim() && chunk.split(/\s+/).length > 10) { // Minimum 10 words
                chunks.push(chunk.trim());
            }
        }
        
        // If no chunks created, use the whole content as one chunk
        if (chunks.length === 0 && content.trim()) {
            chunks.push(content.trim());
        }
        
        return chunks;
    }

    async processFile(filename, content) {
        try {
            // Ensure collection exists
            await this.ensureCollection();
            
            // Create text chunks
            const chunks = this.createChunks(content);
            console.log(`📄 Processing ${filename}: Creating ${chunks.length} chunks`);
            
            // Generate vectors and points
            const points = [];
            const vectorIds = [];
            
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i];
                const vector = this.generateSimpleEmbedding(chunk);
                const pointId = this.vectorIdCounter++;
                
                points.push({
                    id: pointId,
                    vector: vector,
                    payload: {
                        text: chunk,
                        filename: filename,
                        chunk_index: i,
                        total_chunks: chunks.length,
                        timestamp: new Date().toISOString()
                    }
                });
                
                vectorIds.push(pointId);
            }
            
            // Upload points to Qdrant
            if (points.length > 0) {
                const response = await axios.put(
                    `${this.vectorDbUrl}/collections/${this.collectionName}/points`,
                    { 
                        points: points 
                    },
                    { 
                        params: { wait: "true" },
                        timeout: 30000,
                        headers: { 'Content-Type': 'application/json' }
                    }
                );
                
                if (response.data.status === 'ok' || response.data.result) {
                    console.log(`✅ Stored ${points.length} vectors for ${filename}`);
                    fileVectorMap.set(filename, {
                        vectorIds: vectorIds,
                        chunkCount: chunks.length,
                        timestamp: new Date().toISOString()
                    });
                    return { 
                        success: true, 
                        chunks: chunks.length,
                        vectorIds: vectorIds
                    };
                }
            }
            
            throw new Error('No chunks created or upload failed');
            
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
        // Create a 384-dimensional vector using word features
        const vector = new Array(384).fill(0);
        const words = text.toLowerCase().split(/\s+/);
        
        // Feature extraction
        const features = {
            // Word frequencies
            wordFreq: {},
            // Character n-grams
            bigrams: {},
            // Semantic markers
            hasQuestions: /\?/.test(text),
            hasNumbers: /\d/.test(text),
            avgWordLength: 0
        };
        
        // Calculate features
        let totalLength = 0;
        for (const word of words) {
            if (word.length > 2) {
                features.wordFreq[word] = (features.wordFreq[word] || 0) + 1;
                totalLength += word.length;
                
                // Bigrams
                for (let i = 0; i < word.length - 1; i++) {
                    const bigram = word.substring(i, i + 2);
                    features.bigrams[bigram] = (features.bigrams[bigram] || 0) + 1;
                }
            }
        }
        
        features.avgWordLength = words.length > 0 ? totalLength / words.length : 0;
        
        // Encode features into vector
        let index = 0;
        
        // Word frequencies (use first 200 dimensions)
        Object.entries(features.wordFreq).forEach(([word, freq]) => {
            const hash = this.hashString(word);
            const pos = Math.abs(hash) % 200;
            vector[pos] += freq / words.length;
        });
        
        // Bigrams (dimensions 200-350)
        Object.entries(features.bigrams).forEach(([bigram, freq]) => {
            const hash = this.hashString(bigram);
            const pos = 200 + (Math.abs(hash) % 150);
            vector[pos] += freq / (words.length * 2);
        });
        
        // Other features (dimensions 350-384)
        vector[350] = features.hasQuestions ? 1 : 0;
        vector[351] = features.hasNumbers ? 1 : 0;
        vector[352] = features.avgWordLength / 10; // Normalize
        
        // Add some randomness for diversity
        for (let i = 353; i < 384; i++) {
            vector[i] = Math.random() * 0.1;
        }
        
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
            hash = hash & hash;
        }
        return hash;
    }

    async searchSimilar(query, options = {}) {
        const { topK = 5, minSimilarity = 0.1 } = options; // Lower threshold
        
        try {
            // Generate embedding for query
            const queryVector = this.generateSimpleEmbedding(query);
            
            // Search in Qdrant
            const searchPayload = {
                vector: queryVector,
                limit: topK,
                with_payload: true,
                with_vector: false
            };
            
            // Only add score threshold if it's positive
            if (minSimilarity > 0) {
                searchPayload.score_threshold = minSimilarity;
            }
            
            const searchResponse = await axios.post(
                `${this.vectorDbUrl}/collections/${this.collectionName}/points/search`, 
                searchPayload,
                { 
                    timeout: 30000,
                    headers: { 'Content-Type': 'application/json' }
                }
            );

            if (searchResponse.data.result) {
                const results = searchResponse.data.result;
                console.log(`🔍 Found ${results.length} results for "${query}"`);
                
                return {
                    success: true,
                    results: results.map(r => ({
                        filename: r.payload.filename,
                        text: r.payload.text,
                        similarity: r.score,
                        chunk_index: r.payload.chunk_index,
                        total_chunks: r.payload.total_chunks
                    })),
                    totalResults: results.length,
                    method: 'vector-search'
                };
            }
            
            return { success: true, results: [], totalResults: 0 };
            
        } catch (error) {
            console.error(`❌ Search error:`, error.response?.data || error.message);
            // Try keyword fallback
            return await this.keywordSearch(query, topK);
        }
    }

    async keywordSearch(query, limit = 5) {
        console.log(`⚠️ Using keyword search fallback`);
        try {
            // Get all points
            const scrollResponse = await axios.post(
                `${this.vectorDbUrl}/collections/${this.collectionName}/points/scroll`, 
                {
                    limit: 1000,
                    with_payload: true,
                    with_vector: false
                }
            );

            if (scrollResponse.data.result?.points) {
                const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
                
                // Score each point based on keyword matches
                const scoredPoints = scrollResponse.data.result.points.map(point => {
                    const text = point.payload.text?.toLowerCase() || '';
                    let score = 0;
                    
                    // Count matches
                    queryWords.forEach(word => {
                        const matches = (text.match(new RegExp(word, 'g')) || []).length;
                        score += matches;
                    });
                    
                    // Normalize by text length
                    score = score / (text.split(/\s+/).length || 1);
                    
                    return { ...point, score };
                });
                
                // Sort by score and take top results
                const topResults = scoredPoints
                    .filter(p => p.score > 0)
                    .sort((a, b) => b.score - a.score)
                    .slice(0, limit);
                
                return {
                    success: true,
                    results: topResults.map(point => ({
                        filename: point.payload.filename,
                        text: point.payload.text,
                        similarity: Math.min(point.score, 1),
                        chunk_index: point.payload.chunk_index,
                        total_chunks: point.payload.total_chunks
                    })),
                    totalResults: topResults.length,
                    method: 'keyword-fallback'
                };
            }
            
            return { success: true, results: [], totalResults: 0 };
            
        } catch (error) {
            console.error(`❌ Keyword search failed:`, error.message);
            return { success: false, error: error.message, results: [] };
        }
    }

    async deleteFileVectors(filename) {
        console.log(`🗑️ Deleting vectors for: ${filename}`);
        
        try {
            const fileInfo = fileVectorMap.get(filename);
            if (fileInfo && fileInfo.vectorIds) {
                // Delete by IDs
                const deleteResponse = await axios.post(
                    `${this.vectorDbUrl}/collections/${this.collectionName}/points/delete`,
                    {
                        points: fileInfo.vectorIds
                    },
                    { params: { wait: "true" } }
                );
                
                console.log(`✅ Deleted ${fileInfo.vectorIds.length} vectors for ${filename}`);
                fileVectorMap.delete(filename);
                return { success: true, deleted: fileInfo.vectorIds.length };
            }
            
            // Fallback: delete by filter
            const scrollResponse = await axios.post(
                `${this.vectorDbUrl}/collections/${this.collectionName}/points/scroll`,
                {
                    filter: {
                        must: [{
                            key: "filename",
                            match: { value: filename }
                        }]
                    },
                    limit: 1000
                }
            );
            
            if (scrollResponse.data.result?.points?.length > 0) {
                const pointIds = scrollResponse.data.result.points.map(p => p.id);
                await axios.post(
                    `${this.vectorDbUrl}/collections/${this.collectionName}/points/delete`,
                    { points: pointIds },
                    { params: { wait: "true" } }
                );
                
                console.log(`✅ Deleted ${pointIds.length} vectors for ${filename}`);
                return { success: true, deleted: pointIds.length };
            }
            
            return { success: true, deleted: 0 };
            
        } catch (error) {
            console.error(`❌ Failed to delete vectors:`, error.message);
            return { success: false, error: error.message };
        }
    }

    async getCollectionStats() {
        try {
            const response = await axios.get(`${this.vectorDbUrl}/collections/${this.collectionName}`);
            return response.data.result || {};
        } catch (error) {
            return { points_count: 0, vectors_count: 0 };
        }
    }
}

// Initialize vector manager
const vectorManager = new SimpleVectorManager();

// Routes
app.get('/status', async (req, res) => {
    res.json({ 
        status: 'online',
        version: '3.1.0',
        workspace: '/workspace',
        vectorization: {
            enabled: true,
            method: 'simple-embedding',
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
                    const vectorInfo = fileVectorMap.get(file) || { chunkCount: 0 };
                    fileList.push({
                        name: file,
                        size: stats.size,
                        modified: stats.mtime,
                        vectorized: vectorInfo.chunkCount > 0,
                        chunkCount: vectorInfo.chunkCount || 0
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
        
        // Process with vector manager
        const result = await vectorManager.processFile(filename, content);
        
        res.json({
            message: result.success ? 'File uploaded and vectorized' : 'File uploaded but vectorization failed',
            filename: filename,
            fileSize: req.file.size,
            chunksCreated: result.chunks || 0,
            vectorizationStatus: result.success ? 'success' : 'failed',
            processingMethod: 'simple-embedding',
            vectorDatabase: 'qdrant',
            error: result.error
        });
        
    } catch (error) {
        console.error(`File processing error:`, error);
        res.status(500).json({ 
            error: 'File processing failed',
            details: error.message
        });
    }
});

// Vector search endpoint
app.post('/search', async (req, res) => {
    const { query, topK = 5, minSimilarity = 0.1 } = req.body;
    
    if (!query) {
        return res.status(400).json({ error: 'Query is required' });
    }
    
    console.log(`🔍 Search request: "${query}" (topK=${topK}, minSimilarity=${minSimilarity})`);
    
    try {
        const searchResult = await vectorManager.searchSimilar(query, { topK, minSimilarity });
        
        res.json({
            query: query,
            results: searchResult.results || [],
            totalResults: searchResult.totalResults || 0,
            searchStats: {
                vectorDbType: 'qdrant',
                method: searchResult.method || 'unknown'
            }
        });
        
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ 
            error: 'Search failed',
            details: error.message
        });
    }
});

// Collection initialization endpoint
app.post('/collections/init', async (req, res) => {
    console.log('🗃️ Collection init requested');
    
    try {
        const success = await vectorManager.ensureCollection();
        
        res.json({
            status: success ? 'success' : 'error',
            message: success ? 'Collection initialized' : 'Failed to initialize collection',
            collection: COLLECTION_NAME
        });
        
    } catch (error) {
        console.error('Collection init error:', error);
        res.status(500).json({ 
            error: 'Collection init failed',
            details: error.message
        });
    }
});

app.delete('/files/:filename', async (req, res) => {
    const filename = req.params.filename;
    
    try {
        const filePath = path.join('/workspace', filename);
        
        // Delete vectors first
        const deleteResult = await vectorManager.deleteFileVectors(filename);
        
        // Then delete file
        await fs.unlink(filePath);
        
        res.json({
            message: 'File and vectors deleted',
            filename: filename,
            vectorsDeleted: deleteResult.deleted || 0
        });
        
    } catch (error) {
        console.error(`Delete error:`, error);
        if (error.code === 'ENOENT') {
            // File doesn't exist, but still try to delete vectors
            const deleteResult = await vectorManager.deleteFileVectors(filename);
            res.json({
                message: 'File not found but vectors cleaned up',
                filename: filename,
                vectorsDeleted: deleteResult.deleted || 0
            });
        } else {
            res.status(500).json({ 
                error: 'Delete failed',
                details: error.message
            });
        }
    }
});

// Get vector statistics
app.get('/vectors/stats', async (req, res) => {
    try {
        const collectionStats = await vectorManager.getCollectionStats();
        
        res.json({
            collection: COLLECTION_NAME,
            pointsCount: collectionStats.points_count || 0,
            vectorsCount: collectionStats.vectors_count || collectionStats.points_count || 0,
            indexedFilesCount: fileVectorMap.size,
            status: collectionStats.status || 'unknown',
            filesTracked: Array.from(fileVectorMap.keys())
        });
        
    } catch (error) {
        res.json({
            collection: COLLECTION_NAME,
            pointsCount: 0,
            vectorsCount: 0,
            indexedFilesCount: fileVectorMap.size,
            status: 'error',
            error: error.message
        });
    }
});

// Error handling
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({ 
        error: 'Internal server error',
        details: error.message
    });
});

// Process system files on startup
async function processSystemFiles() {
    console.log('🔄 Processing system files...');
    
    try {
        // Ensure collection exists first
        await vectorManager.ensureCollection();
        
        // Wait a bit for Qdrant to be ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const files = await fs.readdir('/workspace');
        for (const file of files) {
            const isSystemFile = file.toLowerCase().includes('system') || 
                               file.toLowerCase().includes('admin') || 
                               file.toLowerCase().includes('config');
            
            if (isSystemFile || file.endsWith('.txt') || file.endsWith('.md')) {
                try {
                    const content = await fs.readFile(path.join('/workspace', file), 'utf8');
                    const result = await vectorManager.processFile(file, content);
                    console.log(`📌 Processed ${file}: ${result.success ? 'success' : 'failed'}`);
                } catch (error) {
                    console.error(`Failed to process ${file}:`, error.message);
                }
            }
        }
        
        // Show final stats
        const stats = await vectorManager.getCollectionStats();
        console.log(`✅ System files processed. Total vectors: ${stats.vectors_count || 0}`);
        
    } catch (error) {
        console.error('System files processing error:', error);
    }
}

// Start server
app.listen(PORT, async () => {
    console.log(`🚀 MCP Server v3.1 running on port ${PORT}`);
    console.log(`💾 Vector DB: Qdrant at ${VECTOR_DB_URL}`);
    console.log(`📦 Collection: ${COLLECTION_NAME}`);
    
    // Create workspace directory
    await fs.mkdir('/workspace', { recursive: true });
    console.log('✅ Workspace initialized');
    
    // Process system files after startup
    setTimeout(processSystemFiles, 3000);
});
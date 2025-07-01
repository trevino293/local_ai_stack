const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const EMBEDDING_SERVICE = process.env.EMBEDDING_SERVICE || 'http://embedding-service:8080';

// Middleware
app.use(cors());
app.use(express.json());

// In-memory vector storage (use a proper vector DB in production)
let vectorStore = [];
let documentChunks = [];

// Configure multer for file uploads
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

const upload = multer({ storage });

// Utility functions
function chunkText(text, chunkSize = 512, overlap = 50) {
    const words = text.split(' ');
    const chunks = [];
    
    for (let i = 0; i < words.length; i += chunkSize - overlap) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        if (chunk.trim().length > 0) {
            chunks.push({
                text: chunk,
                startIndex: i,
                endIndex: Math.min(i + chunkSize, words.length)
            });
        }
    }
    
    return chunks;
}

async function generateEmbedding(text) {
    try {
        const response = await axios.post(`${EMBEDDING_SERVICE}/embed`, {
            text: text
        }, { timeout: 30000 });
        
        return response.data.embedding;
    } catch (error) {
        console.error('Embedding generation failed:', error.message);
        return null;
    }
}

function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function processAndVectorizeFile(filename, content) {
    console.log(`Processing file: ${filename}`);
    
    // Chunk the content
    const chunks = chunkText(content, 512, 50);
    console.log(`Created ${chunks.length} chunks for ${filename}`);
    
    // Generate embeddings for each chunk
    const processedChunks = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const embedding = await generateEmbedding(chunk.text);
        
        if (embedding) {
            const chunkData = {
                filename: filename,
                chunkIndex: i,
                text: chunk.text,
                embedding: embedding,
                startIndex: chunk.startIndex,
                endIndex: chunk.endIndex,
                timestamp: new Date().toISOString()
            };
            
            processedChunks.push(chunkData);
            vectorStore.push(chunkData);
        } else {
            console.warn(`Failed to generate embedding for chunk ${i} of ${filename}`);
        }
    }
    
    console.log(`Successfully vectorized ${processedChunks.length} chunks for ${filename}`);
    return processedChunks;
}

// Routes

// Status endpoint with vectorization info
app.get('/status', (req, res) => {
    res.json({ 
        status: 'online',
        version: '2.0.0',
        workspace: '/workspace',
        vectorization: {
            enabled: true,
            totalVectors: vectorStore.length,
            embeddingService: EMBEDDING_SERVICE
        }
    });
});

// Health check for embedding service
app.get('/embedding/health', async (req, res) => {
    try {
        const response = await axios.get(`${EMBEDDING_SERVICE}/health`, { timeout: 5000 });
        res.json({ 
            embeddingService: 'online',
            modelInfo: response.data 
        });
    } catch (error) {
        res.status(503).json({ 
            embeddingService: 'offline',
            error: error.message 
        });
    }
});

// List files
app.get('/files', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const fileList = [];
        
        for (const file of files) {
            const stats = await fs.stat(path.join('/workspace', file));
            if (stats.isFile()) {
                const vectorCount = vectorStore.filter(v => v.filename === file).length;
                fileList.push({
                    name: file,
                    size: stats.size,
                    modified: stats.mtime,
                    vectorized: vectorCount > 0,
                    chunkCount: vectorCount
                });
            }
        }
        
        res.json(fileList);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get file content
app.get('/files/:filename', async (req, res) => {
    try {
        const filePath = path.join('/workspace', req.params.filename);
        const content = await fs.readFile(filePath, 'utf8');
        res.type('text/plain').send(content);
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

// Upload file with automatic vectorization
app.post('/files', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    try {
        // Read and process the uploaded file
        const filePath = path.join('/workspace', req.file.filename);
        const content = await fs.readFile(filePath, 'utf8');
        
        // Remove existing vectors for this file
        vectorStore = vectorStore.filter(v => v.filename !== req.file.filename);
        
        // Process and vectorize the file
        const chunks = await processAndVectorizeFile(req.file.filename, content);
        
        res.json({ 
            message: 'File uploaded and vectorized successfully',
            filename: req.file.filename,
            chunksCreated: chunks.length,
            vectorizationStatus: chunks.length > 0 ? 'success' : 'failed'
        });
    } catch (error) {
        console.error('File processing error:', error);
        res.status(500).json({ 
            error: 'File upload succeeded but vectorization failed',
            details: error.message 
        });
    }
});

// Semantic search endpoint
app.post('/search', async (req, res) => {
    const { query, topK = 5, minSimilarity = 0.3 } = req.body;
    
    if (!query) {
        return res.status(400).json({ error: 'Query is required' });
    }
    
    try {
        // Generate embedding for the query
        const queryEmbedding = await generateEmbedding(query);
        
        if (!queryEmbedding) {
            return res.status(500).json({ error: 'Failed to generate query embedding' });
        }
        
        // Calculate similarities
        const similarities = vectorStore.map(chunk => ({
            ...chunk,
            similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
        }))
        .filter(chunk => chunk.similarity >= minSimilarity)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK);
        
        // Format results
        const results = similarities.map(chunk => ({
            filename: chunk.filename,
            chunk: chunk.text,
            similarity: chunk.similarity,
            chunkIndex: chunk.chunkIndex,
            timestamp: chunk.timestamp
        }));
        
        res.json({
            query: query,
            results: results,
            totalResults: similarities.length,
            searchStats: {
                totalVectors: vectorStore.length,
                queryTime: Date.now()
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

// Batch vectorization endpoint for existing files
app.post('/vectorize/batch', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const results = [];
        
        for (const filename of files) {
            const stats = await fs.stat(path.join('/workspace', filename));
            if (stats.isFile()) {
                try {
                    const content = await fs.readFile(path.join('/workspace', filename), 'utf8');
                    
                    // Remove existing vectors for this file
                    vectorStore = vectorStore.filter(v => v.filename !== filename);
                    
                    // Process and vectorize
                    const chunks = await processAndVectorizeFile(filename, content);
                    
                    results.push({
                        filename: filename,
                        status: 'success',
                        chunksCreated: chunks.length
                    });
                } catch (error) {
                    results.push({
                        filename: filename,
                        status: 'error',
                        error: error.message
                    });
                }
            }
        }
        
        res.json({
            message: 'Batch vectorization completed',
            results: results,
            totalVectors: vectorStore.length
        });
        
    } catch (error) {
        res.status(500).json({ 
            error: 'Batch vectorization failed',
            details: error.message 
        });
    }
});

// Delete file and its vectors
app.delete('/files/:filename', async (req, res) => {
    try {
        const filePath = path.join('/workspace', req.params.filename);
        await fs.unlink(filePath);
        
        // Remove vectors for this file
        const initialCount = vectorStore.length;
        vectorStore = vectorStore.filter(v => v.filename !== req.params.filename);
        const removedVectors = initialCount - vectorStore.length;
        
        res.json({ 
            message: 'File and vectors deleted successfully',
            removedVectors: removedVectors
        });
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

// Vector store statistics
app.get('/vectors/stats', (req, res) => {
    const fileStats = {};
    
    vectorStore.forEach(chunk => {
        if (!fileStats[chunk.filename]) {
            fileStats[chunk.filename] = {
                chunkCount: 0,
                lastUpdate: chunk.timestamp
            };
        }
        fileStats[chunk.filename].chunkCount++;
        if (chunk.timestamp > fileStats[chunk.filename].lastUpdate) {
            fileStats[chunk.filename].lastUpdate = chunk.timestamp;
        }
    });
    
    res.json({
        totalVectors: vectorStore.length,
        fileCount: Object.keys(fileStats).length,
        fileStats: fileStats,
        embeddingDimensions: vectorStore.length > 0 ? vectorStore[0].embedding.length : 0
    });
});

// Create workspace directory if it doesn't exist
async function initWorkspace() {
    try {
        await fs.mkdir('/workspace', { recursive: true });
        console.log('Workspace directory initialized');
        
        // Check embedding service connectivity
        try {
            await axios.get(`${EMBEDDING_SERVICE}/health`, { timeout: 5000 });
            console.log('Embedding service is available');
        } catch (error) {
            console.warn('Embedding service not available:', error.message);
        }
        
    } catch (error) {
        console.error('Error creating workspace:', error);
    }
}

// Start server
app.listen(PORT, async () => {
    await initWorkspace();
    console.log(`Enhanced MCP Filesystem Server running on port ${PORT}`);
    console.log(`Vectorization enabled with embedding service: ${EMBEDDING_SERVICE}`);
});
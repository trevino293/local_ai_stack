// mcp-server/server_vectordb.js - Enhanced MCP server with vector database integration
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const EMBEDDING_SERVICE = process.env.EMBEDDING_SERVICE || 'http://embedding-service:8080';
const VECTOR_DB_TYPE = process.env.VECTOR_DB_TYPE || 'qdrant';

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
    },
    fileFilter: (req, file, cb) => {
        // Allow text files
        const allowedTypes = /\.(txt|md|json|csv|yaml|yml|log)$/i;
        if (allowedTypes.test(file.originalname)) {
            cb(null, true);
        } else {
            cb(new Error('Only text files are allowed'), false);
        }
    }
});

// Cache for file metadata
let fileMetadataCache = new Map();

/**
 * Enhanced file processing with vector database integration
 */
class VectorFileManager {
    constructor() {
        this.embeddingService = EMBEDDING_SERVICE;
        this.vectorDbType = VECTOR_DB_TYPE;
    }

    async processFile(filename, content) {
        console.log(`Processing file with vector database: ${filename}`);
        
        try {
            // Send document to embedding service for processing
            const response = await axios.post(`${this.embeddingService}/embed/document`, {
                filename: filename,
                content: content,
                chunk_size: 512,
                overlap: 50
            }, { timeout: 60000 });

            if (response.data.status === 'success') {
                console.log(`Successfully processed ${filename}: ${response.data.chunks_created} chunks created`);
                
                // Update file metadata cache
                fileMetadataCache.set(filename, {
                    processed: true,
                    chunks: response.data.chunks_created,
                    lastProcessed: new Date().toISOString(),
                    processingMethod: response.data.processing_method
                });

                return {
                    success: true,
                    chunks: response.data.chunks_created,
                    method: response.data.processing_method
                };
            } else {
                throw new Error('Embedding service returned error status');
            }
        } catch (error) {
            console.error(`Failed to process file ${filename}:`, error.message);
            
            // Update cache with error status
            fileMetadataCache.set(filename, {
                processed: false,
                error: error.message,
                lastAttempted: new Date().toISOString()
            });

            return {
                success: false,
                error: error.message
            };
        }
    }

    async searchSimilar(query, options = {}) {
        const { topK = 5, minSimilarity = 0.3, filters = {} } = options;
        
        try {
            console.log(`Searching for: "${query}" (topK: ${topK}, minSimilarity: ${minSimilarity})`);
            
            const response = await axios.post(`${this.embeddingService}/search`, {
                query: query,
                topK: topK,
                minSimilarity: minSimilarity,
                filters: filters
            }, { timeout: 30000 });

            if (response.data.results) {
                console.log(`Found ${response.data.results.length} similar chunks`);
                return {
                    success: true,
                    results: response.data.results,
                    totalResults: response.data.total_results,
                    method: response.data.processing_method
                };
            } else {
                return { success: true, results: [], totalResults: 0 };
            }
        } catch (error) {
            console.error(`Search failed:`, error.message);
            return {
                success: false,
                error: error.message,
                results: []
            };
        }
    }

    async deleteFileEmbeddings(filename) {
        try {
            const response = await axios.delete(`${this.embeddingService}/delete/${filename}`, {
                timeout: 10000
            });

            if (response.data.status === 'success') {
                console.log(`Deleted embeddings for: ${filename}`);
                fileMetadataCache.delete(filename);
                return { success: true };
            } else {
                throw new Error(response.data.error || 'Delete operation failed');
            }
        } catch (error) {
            console.error(`Failed to delete embeddings for ${filename}:`, error.message);
            return { success: false, error: error.message };
        }
    }

    async getVectorStats() {
        try {
            const response = await axios.get(`${this.embeddingService}/stats`, {
                timeout: 10000
            });

            return {
                success: true,
                stats: response.data
            };
        } catch (error) {
            console.error('Failed to get vector stats:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }
}

// Initialize vector file manager
const vectorManager = new VectorFileManager();

// Routes

/**
 * Enhanced status endpoint with vector database information
 */
app.get('/status', async (req, res) => {
    try {
        const vectorStats = await vectorManager.getVectorStats();
        
        res.json({ 
            status: 'online',
            version: '3.0.0',
            workspace: '/workspace',
            vectorization: {
                enabled: true,
                vectorDatabase: VECTOR_DB_TYPE,
                embeddingService: EMBEDDING_SERVICE,
                ...vectorStats.stats
            },
            fileCache: {
                totalFiles: fileMetadataCache.size,
                processedFiles: Array.from(fileMetadataCache.values()).filter(f => f.processed).length
            }
        });
    } catch (error) {
        res.json({
            status: 'online',
            version: '3.0.0',
            workspace: '/workspace',
            vectorization: {
                enabled: true,
                vectorDatabase: VECTOR_DB_TYPE,
                embeddingService: EMBEDDING_SERVICE,
                error: error.message
            }
        });
    }
});

/**
 * Health check for embedding service and vector database
 */
app.get('/health', async (req, res) => {
    try {
        const embeddingHealth = await axios.get(`${EMBEDDING_SERVICE}/health`, { timeout: 5000 });
        
        res.json({
            status: 'healthy',
            embeddingService: 'online',
            vectorDatabase: embeddingHealth.data.vector_database || VECTOR_DB_TYPE,
            embeddingMethod: embeddingHealth.data.embedding_method,
            model: embeddingHealth.data.model
        });
    } catch (error) {
        res.status(503).json({
            status: 'degraded',
            embeddingService: 'offline',
            error: error.message
        });
    }
});

/**
 * Enhanced file listing with vector processing status
 */
app.get('/files', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const fileList = [];
        
        for (const file of files) {
            const stats = await fs.stat(path.join('/workspace', file));
            if (stats.isFile()) {
                const metadata = fileMetadataCache.get(file) || { processed: false };
                
                fileList.push({
                    name: file,
                    size: stats.size,
                    modified: stats.mtime,
                    vectorized: metadata.processed || false,
                    chunkCount: metadata.chunks || 0,
                    lastProcessed: metadata.lastProcessed || null,
                    processingMethod: metadata.processingMethod || null,
                    error: metadata.error || null
                });
            }
        }
        
        res.json(fileList);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Get file content
 */
app.get('/files/:filename', async (req, res) => {
    try {
        const filePath = path.join('/workspace', req.params.filename);
        const content = await fs.readFile(filePath, 'utf8');
        res.type('text/plain').send(content);
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

/**
 * Enhanced file upload with automatic vector processing
 */
app.post('/files', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    try {
        const filePath = path.join('/workspace', req.file.filename);
        const content = await fs.readFile(filePath, 'utf8');
        
        console.log(`Processing uploaded file: ${req.file.filename} (${content.length} characters)`);
        
        // Process with vector database
        const processingResult = await vectorManager.processFile(req.file.filename, content);
        
        if (processingResult.success) {
            res.json({
                message: 'File uploaded and vectorized successfully',
                filename: req.file.filename,
                chunksCreated: processingResult.chunks,
                vectorizationStatus: 'success',
                processingMethod: processingResult.method,
                fileSize: req.file.size
            });
        } else {
            res.status(206).json({
                message: 'File uploaded but vectorization failed',
                filename: req.file.filename,
                vectorizationStatus: 'failed',
                error: processingResult.error,
                fileSize: req.file.size
            });
        }
    } catch (error) {
        console.error('File upload error:', error);
        res.status(500).json({
            error: 'File upload failed',
            details: error.message
        });
    }
});

/**
 * Enhanced semantic search endpoint
 */
app.post('/search', async (req, res) => {
    const { query, topK = 5, minSimilarity = 0.3, filters = {} } = req.body;
    
    if (!query) {
        return res.status(400).json({ error: 'Query is required' });
    }
    
    try {
        console.log(`Search request: "${query}" (topK: ${topK}, similarity: ${minSimilarity})`);
        
        const searchResult = await vectorManager.searchSimilar(query, {
            topK,
            minSimilarity,
            filters
        });
        
        if (searchResult.success) {
            // Format results for compatibility
            const formattedResults = searchResult.results.map(result => ({
                filename: result.filename,
                chunk: result.text,
                similarity: result.similarity,
                chunkIndex: result.chunk_index,
                timestamp: result.timestamp,
                metadata: result.metadata
            }));
            
            res.json({
                query: query,
                results: formattedResults,
                totalResults: searchResult.totalResults,
                processingMethod: searchResult.method,
                searchStats: {
                    vectorDatabase: VECTOR_DB_TYPE,
                    queryTime: Date.now(),
                    requestedResults: topK,
                    minSimilarity: minSimilarity
                }
            });
        } else {
            res.status(500).json({
                error: 'Search failed',
                details: searchResult.error,
                query: query
            });
        }
        
    } catch (error) {
        console.error('Search endpoint error:', error);
        res.status(500).json({
            error: 'Search endpoint failed',
            details: error.message
        });
    }
});

/**
 * Batch processing endpoint for existing files
 */
app.post('/vectorize/batch', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const results = [];
        
        console.log(`Starting batch vectorization of ${files.length} files`);
        
        for (const filename of files) {
            const stats = await fs.stat(path.join('/workspace', filename));
            if (stats.isFile()) {
                try {
                    const content = await fs.readFile(path.join('/workspace', filename), 'utf8');
                    
                    console.log(`Batch processing: ${filename}`);
                    const processingResult = await vectorManager.processFile(filename, content);
                    
                    results.push({
                        filename: filename,
                        status: processingResult.success ? 'success' : 'error',
                        chunksCreated: processingResult.chunks || 0,
                        error: processingResult.error || null,
                        method: processingResult.method || null
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
        
        const successCount = results.filter(r => r.status === 'success').length;
        const totalChunks = results.reduce((sum, r) => sum + (r.chunksCreated || 0), 0);
        
        res.json({
            message: 'Batch vectorization completed',
            results: results,
            summary: {
                totalFiles: results.length,
                successfulFiles: successCount,
                failedFiles: results.length - successCount,
                totalChunks: totalChunks
            }
        });
        
    } catch (error) {
        res.status(500).json({
            error: 'Batch vectorization failed',
            details: error.message
        });
    }
});

/**
 * Enhanced delete endpoint with vector cleanup
 */
app.delete('/files/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        const filePath = path.join('/workspace', filename);
        
        // Delete file from filesystem
        await fs.unlink(filePath);
        
        // Delete vectors from database
        const deleteResult = await vectorManager.deleteFileEmbeddings(filename);
        
        res.json({
            message: 'File and vectors deleted successfully',
            filename: filename,
            vectorDeletion: deleteResult.success ? 'success' : 'failed',
            vectorError: deleteResult.error || null
        });
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'File not found' });
        } else {
            res.status(500).json({ error: error.message });
        }
    }
});

/**
 * Vector database statistics endpoint
 */
app.get('/vectors/stats', async (req, res) => {
    try {
        const vectorStats = await vectorManager.getVectorStats();
        
        if (vectorStats.success) {
            res.json({
                success: true,
                vectorDatabase: VECTOR_DB_TYPE,
                embeddingService: EMBEDDING_SERVICE,
                ...vectorStats.stats,
                fileCache: {
                    totalCachedFiles: fileMetadataCache.size,
                    processedFiles: Array.from(fileMetadataCache.values()).filter(f => f.processed).length
                }
            });
        } else {
            res.status(503).json({
                success: false,
                error: vectorStats.error
            });
        }
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * Search suggestions endpoint
 */
app.get('/search/suggestions', async (req, res) => {
    try {
        // Simple implementation - in production, this could be more sophisticated
        const suggestions = [
            "system features capabilities",
            "configuration settings",
            "API endpoints documentation",
            "installation requirements",
            "troubleshooting guide"
        ];
        
        res.json({
            suggestions: suggestions,
            vectorDatabase: VECTOR_DB_TYPE
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Initialize workspace and check services
 */
async function initializeServices() {
    try {
        // Create workspace directory
        await fs.mkdir('/workspace', { recursive: true });
        console.log('Workspace directory initialized');
        
        // Check embedding service connectivity
        try {
            const healthCheck = await axios.get(`${EMBEDDING_SERVICE}/health`, { timeout: 10000 });
            console.log(`Embedding service connected: ${healthCheck.data.status}`);
            console.log(`Vector database: ${healthCheck.data.vector_database}`);
            console.log(`Embedding method: ${healthCheck.data.embedding_method}`);
        } catch (error) {
            console.warn(`Embedding service not available: ${error.message}`);
        }
        
        // Load existing file metadata
        try {
            const files = await fs.readdir('/workspace');
            console.log(`Found ${files.length} existing files in workspace`);
            
            // Initialize metadata cache for existing files
            for (const file of files) {
                const stats = await fs.stat(path.join('/workspace', file));
                if (stats.isFile()) {
                    fileMetadataCache.set(file, {
                        processed: false, // Will be updated if vectorization is detected
                        lastModified: stats.mtime.toISOString()
                    });
                }
            }
        } catch (error) {
            console.error('Error loading existing files:', error.message);
        }
        
    } catch (error) {
        console.error('Error initializing services:', error);
    }
}

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large. Maximum size is 50MB.' });
        }
    }
    
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, async () => {
    await initializeServices();
    console.log(`Enhanced MCP Filesystem Server running on port ${PORT}`);
    console.log(`Vector database integration: ${VECTOR_DB_TYPE}`);
    console.log(`Embedding service: ${EMBEDDING_SERVICE}`);
});
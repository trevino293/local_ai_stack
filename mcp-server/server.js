// mcp-server/server.js - Complete fixed version with embedding proxy integration
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Updated environment variables for embedding proxy
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

// Cache for file metadata with enhanced tracking
let fileMetadataCache = new Map();

/**
 * Enhanced vector file manager with embedding proxy integration
 */
class VectorFileManager {
    constructor() {
        this.embeddingProxyUrl = EMBEDDING_PROXY_URL;
        this.vectorDbUrl = VECTOR_DB_URL;
        this.vectorDbType = VECTOR_DB_TYPE;
        this.collectionName = COLLECTION_NAME;
        
        console.log(`Initialized VectorFileManager:`);
        console.log(`  Embedding Proxy: ${this.embeddingProxyUrl}`);
        console.log(`  Vector DB: ${this.vectorDbUrl} (${this.vectorDbType})`);
        console.log(`  Collection: ${this.collectionName}`);
    }

    async processFile(filename, content) {
        console.log(`Processing file with embedding proxy: ${filename} (${content.length} chars)`);
        
        const startTime = Date.now();
        
        try {
            // Send document to embedding proxy for processing
            const response = await axios.post(`${this.embeddingProxyUrl}/embed/document`, {
                filename: filename,
                content: content,
                chunk_size: 512,
                overlap: 50
            }, { 
                timeout: 120000,  // 2 minutes for large files
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const processingTime = Date.now() - startTime;

            if (response.data.status === 'success') {
                console.log(`✅ Successfully processed ${filename}: ${response.data.chunks_created} chunks in ${processingTime}ms`);
                
                // Update file metadata cache with detailed info
                fileMetadataCache.set(filename, {
                    processed: true,
                    chunks: response.data.chunks_created,
                    lastProcessed: new Date().toISOString(),
                    processingMethod: response.data.processing_method,
                    vectorDatabase: response.data.vector_database,
                    processingTimeMs: processingTime,
                    fileSize: content.length,
                    success: true
                });

                return {
                    success: true,
                    chunks: response.data.chunks_created,
                    method: response.data.processing_method,
                    vectorDatabase: response.data.vector_database,
                    processingTime: processingTime
                };
            } else {
                throw new Error(response.data.error || 'Embedding proxy returned error status');
            }
        } catch (error) {
            const processingTime = Date.now() - startTime;
            console.error(`❌ Failed to process file ${filename} after ${processingTime}ms:`, error.message);
            
            // Update cache with detailed error status
            fileMetadataCache.set(filename, {
                processed: false,
                error: error.message,
                lastAttempted: new Date().toISOString(),
                processingTimeMs: processingTime,
                fileSize: content.length,
                success: false,
                errorType: error.code || 'PROCESSING_ERROR'
            });

            return {
                success: false,
                error: error.message,
                processingTime: processingTime
            };
        }
    }

    async searchSimilar(query, options = {}) {
        const { topK = 5, minSimilarity = 0.3, filters = {} } = options;
        
        console.log(`🔍 Searching for: "${query}" (topK: ${topK}, minSimilarity: ${minSimilarity})`);
        
        const startTime = Date.now();
        
        try {
            const response = await axios.post(`${this.embeddingProxyUrl}/search`, {
                query: query,
                topK: topK,
                minSimilarity: minSimilarity,
                filters: filters
            }, { 
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const searchTime = Date.now() - startTime;

            if (response.data.results) {
                const resultCount = response.data.results.length;
                const avgSimilarity = resultCount > 0 
                    ? (response.data.results.reduce((sum, r) => sum + (r.similarity || 0), 0) / resultCount).toFixed(3)
                    : '0.000';

                console.log(`✅ Search completed: ${resultCount} results, avg similarity: ${avgSimilarity}, time: ${searchTime}ms`);
                
                return {
                    success: true,
                    results: response.data.results,
                    totalResults: response.data.total_results || resultCount,
                    method: response.data.processing_method,
                    vectorDatabase: response.data.vector_database,
                    searchTime: searchTime,
                    avgSimilarity: parseFloat(avgSimilarity),
                    query: query
                };
            } else {
                console.log(`🔍 Search completed: 0 results for "${query}"`);
                return { 
                    success: true, 
                    results: [], 
                    totalResults: 0,
                    searchTime: searchTime,
                    query: query
                };
            }
        } catch (error) {
            const searchTime = Date.now() - startTime;
            console.error(`❌ Search failed after ${searchTime}ms:`, error.message);
            return {
                success: false,
                error: error.message,
                results: [],
                searchTime: searchTime,
                query: query
            };
        }
    }

    async deleteFileEmbeddings(filename) {
        console.log(`🗑️ Deleting embeddings for: ${filename}`);
        
        try {
            const response = await axios.delete(`${this.embeddingProxyUrl}/delete/${filename}`, {
                timeout: 15000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.data.status === 'success') {
                console.log(`✅ Deleted embeddings for: ${filename}`);
                fileMetadataCache.delete(filename);
                return { success: true };
            } else {
                throw new Error(response.data.error || 'Delete operation failed');
            }
        } catch (error) {
            console.error(`❌ Failed to delete embeddings for ${filename}:`, error.message);
            return { success: false, error: error.message };
        }
    }

    async getVectorStats() {
        try {
            const response = await axios.get(`${this.embeddingProxyUrl}/stats`, {
                timeout: 10000,
                headers: {
                    'Accept': 'application/json'
                }
            });

            return {
                success: true,
                stats: response.data
            };
        } catch (error) {
            console.error('❌ Failed to get vector stats:', error.message);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async checkEmbeddingProxyHealth() {
        try {
            const response = await axios.get(`${this.embeddingProxyUrl}/health`, {
                timeout: 5000,
                headers: {
                    'Accept': 'application/json'
                }
            });

            return {
                healthy: response.status === 200,
                data: response.data,
                responseTime: response.headers['x-response-time'] || 'unknown'
            };
        } catch (error) {
            return {
                healthy: false,
                error: error.message,
                code: error.code || 'CONNECTION_ERROR'
            };
        }
    }

    async initializeVectorCollection() {
        console.log(`🗃️ Initializing vector collection: ${this.collectionName}`);
        
        try {
            const response = await axios.post(`${this.embeddingProxyUrl}/collections/init`, {}, {
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.data.status === 'success') {
                console.log(`✅ Vector collection initialized: ${this.collectionName}`);
                return { success: true, message: response.data.message };
            } else {
                throw new Error(response.data.error || 'Collection initialization failed');
            }
        } catch (error) {
            console.error(`❌ Failed to initialize collection:`, error.message);
            return { success: false, error: error.message };
        }
    }

    getCacheStats() {
        const cacheValues = Array.from(fileMetadataCache.values());
        return {
            totalFiles: fileMetadataCache.size,
            processedFiles: cacheValues.filter(f => f.processed).length,
            failedFiles: cacheValues.filter(f => !f.success && f.error).length,
            totalChunks: cacheValues.reduce((sum, f) => sum + (f.chunks || 0), 0),
            avgProcessingTime: cacheValues.length > 0 
                ? Math.round(cacheValues.reduce((sum, f) => sum + (f.processingTimeMs || 0), 0) / cacheValues.length)
                : 0
        };
    }
}

// Initialize enhanced vector file manager
const vectorManager = new VectorFileManager();

// Routes

/**
 * Enhanced status endpoint with comprehensive system information
 */
app.get('/status', async (req, res) => {
    try {
        console.log('📊 Status check requested');
        
        const [vectorStats, proxyHealth] = await Promise.all([
            vectorManager.getVectorStats(),
            vectorManager.checkEmbeddingProxyHealth()
        ]);
        
        const cacheStats = vectorManager.getCacheStats();
        
        res.json({ 
            status: 'online',
            version: '3.0.0-prebuilt',
            timestamp: new Date().toISOString(),
            workspace: '/workspace',
            vectorization: {
                enabled: true,
                type: 'prebuilt-proxy',
                embeddingProxy: {
                    url: EMBEDDING_PROXY_URL,
                    healthy: proxyHealth.healthy,
                    responseTime: proxyHealth.responseTime,
                    error: proxyHealth.error || null
                },
                vectorDatabase: {
                    type: VECTOR_DB_TYPE,
                    url: VECTOR_DB_URL,
                    collection: COLLECTION_NAME
                },
                stats: vectorStats.success ? vectorStats.stats : { error: vectorStats.error }
            },
            fileCache: cacheStats,
            performance: {
                buildTime: 'optimized',
                searchSpeed: 'enhanced',
                storageType: 'persistent'
            }
        });
    } catch (error) {
        console.error('❌ Status check failed:', error);
        res.status(500).json({
            status: 'error',
            version: '3.0.0-prebuilt',
            timestamp: new Date().toISOString(),
            error: error.message,
            vectorization: {
                enabled: true,
                embeddingProxy: EMBEDDING_PROXY_URL,
                vectorDatabase: VECTOR_DB_TYPE,
                error: 'Status check failed'
            }
        });
    }
});

/**
 * Comprehensive health check endpoint
 */
app.get('/health', async (req, res) => {
    try {
        console.log('🏥 Health check requested');
        
        const proxyHealth = await vectorManager.checkEmbeddingProxyHealth();
        
        if (proxyHealth.healthy) {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                services: {
                    embeddingProxy: {
                        status: 'online',
                        url: EMBEDDING_PROXY_URL,
                        responseTime: proxyHealth.responseTime
                    },
                    vectorDatabase: {
                        type: proxyHealth.data.vector_database || VECTOR_DB_TYPE,
                        status: proxyHealth.data.database_status || 'connected',
                        collection: COLLECTION_NAME
                    }
                },
                capabilities: {
                    embeddingMethod: proxyHealth.data.embedding_method || 'prebuilt',
                    model: proxyHealth.data.model || 'sentence-transformers/all-MiniLM-L6-v2',
                    version: proxyHealth.data.version || '2.0.0-prebuilt'
                }
            });
        } else {
            res.status(503).json({
                status: 'degraded',
                timestamp: new Date().toISOString(),
                services: {
                    embeddingProxy: {
                        status: 'offline',
                        url: EMBEDDING_PROXY_URL,
                        error: proxyHealth.error,
                        code: proxyHealth.code
                    }
                },
                error: 'Embedding proxy unavailable'
            });
        }
    } catch (error) {
        console.error('❌ Health check failed:', error);
        res.status(503).json({
            status: 'error',
            timestamp: new Date().toISOString(),
            error: error.message
        });
    }
});

/**
 * Enhanced file listing with detailed vector processing status
 */
app.get('/files', async (req, res) => {
    try {
        console.log('📁 File listing requested');
        
        const files = await fs.readdir('/workspace');
        const fileList = [];
        
        for (const file of files) {
            try {
                const stats = await fs.stat(path.join('/workspace', file));
                if (stats.isFile()) {
                    const metadata = fileMetadataCache.get(file) || { 
                        processed: false, 
                        lastChecked: new Date().toISOString() 
                    };
                    
                    fileList.push({
                        name: file,
                        size: stats.size,
                        modified: stats.mtime,
                        created: stats.birthtime,
                        // Vector processing status
                        vectorized: metadata.processed || false,
                        chunkCount: metadata.chunks || 0,
                        lastProcessed: metadata.lastProcessed || null,
                        lastAttempted: metadata.lastAttempted || null,
                        // Processing details
                        processingMethod: metadata.processingMethod || null,
                        vectorDatabase: metadata.vectorDatabase || null,
                        processingTimeMs: metadata.processingTimeMs || null,
                        // Error information
                        error: metadata.error || null,
                        errorType: metadata.errorType || null,
                        success: metadata.success !== false // Default to true if not set
                    });
                }
            } catch (fileError) {
                console.warn(`⚠️ Error reading file ${file}:`, fileError.message);
                // Include problematic files in listing with error status
                fileList.push({
                    name: file,
                    size: 0,
                    modified: null,
                    vectorized: false,
                    error: `File access error: ${fileError.message}`,
                    success: false
                });
            }
        }
        
        console.log(`📁 Listed ${fileList.length} files`);
        res.json(fileList);
        
    } catch (error) {
        console.error('❌ File listing failed:', error);
        res.status(500).json({ 
            error: 'Failed to list files',
            details: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Get file content with enhanced error handling
 */
app.get('/files/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        console.log(`📄 Content requested for: ${filename}`);
        
        const filePath = path.join('/workspace', filename);
        const content = await fs.readFile(filePath, 'utf8');
        
        res.type('text/plain').send(content);
        
    } catch (error) {
        console.error(`❌ Failed to read file ${req.params.filename}:`, error.message);
        
        if (error.code === 'ENOENT') {
            res.status(404).json({ 
                error: 'File not found',
                filename: req.params.filename,
                timestamp: new Date().toISOString()
            });
        } else if (error.code === 'EACCES') {
            res.status(403).json({ 
                error: 'File access denied',
                filename: req.params.filename,
                timestamp: new Date().toISOString()
            });
        } else {
            res.status(500).json({ 
                error: 'Failed to read file',
                details: error.message,
                filename: req.params.filename,
                timestamp: new Date().toISOString()
            });
        }
    }
});

/**
 * Enhanced file upload with comprehensive vector processing
 */
app.post('/files', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ 
            error: 'No file uploaded',
            timestamp: new Date().toISOString()
        });
    }
    
    const filename = req.file.filename;
    console.log(`⬆️ File upload started: ${filename} (${req.file.size} bytes)`);
    
    try {
        const filePath = path.join('/workspace', filename);
        const content = await fs.readFile(filePath, 'utf8');
        
        console.log(`📝 Processing uploaded file: ${filename} (${content.length} characters)`);
        
        // Process with embedding proxy
        const processingResult = await vectorManager.processFile(filename, content);
        
        if (processingResult.success) {
            const responseData = {
                message: 'File uploaded and vectorized successfully',
                filename: filename,
                fileSize: req.file.size,
                contentLength: content.length,
                // Vectorization results
                chunksCreated: processingResult.chunks,
                vectorizationStatus: 'success',
                processingMethod: processingResult.method,
                vectorDatabase: processingResult.vectorDatabase,
                processingTimeMs: processingResult.processingTime,
                // System info
                timestamp: new Date().toISOString(),
                embeddingProxy: EMBEDDING_PROXY_URL
            };
            
            console.log(`✅ Upload successful: ${filename}`);
            res.json(responseData);
            
        } else {
            const responseData = {
                message: 'File uploaded but vectorization failed',
                filename: filename,
                fileSize: req.file.size,
                contentLength: content.length,
                // Vectorization results
                vectorizationStatus: 'failed',
                error: processingResult.error,
                processingTimeMs: processingResult.processingTime,
                // System info
                timestamp: new Date().toISOString(),
                embeddingProxy: EMBEDDING_PROXY_URL
            };
            
            console.log(`⚠️ Upload with vectorization failure: ${filename}`);
            res.status(206).json(responseData); // 206 Partial Content
        }
        
    } catch (error) {
        console.error(`❌ File upload failed for ${filename}:`, error);
        res.status(500).json({
            error: 'File upload failed',
            details: error.message,
            filename: filename,
            fileSize: req.file?.size || 0,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Enhanced semantic search endpoint with comprehensive metrics
 */
app.post('/search', async (req, res) => {
    const { query, topK = 5, minSimilarity = 0.3, filters = {} } = req.body;
    
    if (!query) {
        return res.status(400).json({ 
            error: 'Query is required',
            timestamp: new Date().toISOString()
        });
    }
    
    console.log(`🔍 Search request: "${query}" (topK: ${topK}, similarity: ${minSimilarity})`);
    
    try {
        const searchResult = await vectorManager.searchSimilar(query, {
            topK,
            minSimilarity,
            filters
        });
        
        if (searchResult.success) {
            // Format results for API compatibility
            const formattedResults = searchResult.results.map(result => ({
                filename: result.filename,
                chunk: result.text,
                similarity: result.similarity,
                chunkIndex: result.chunk_index,
                timestamp: result.timestamp,
                metadata: result.metadata || {}
            }));
            
            const responseData = {
                query: query,
                results: formattedResults,
                totalResults: searchResult.totalResults,
                // Enhanced metadata
                processingMethod: searchResult.method,
                vectorDatabase: searchResult.vectorDatabase,
                searchStats: {
                    embeddingProxy: EMBEDDING_PROXY_URL,
                    vectorDbType: VECTOR_DB_TYPE,
                    collection: COLLECTION_NAME,
                    queryTime: new Date().toISOString(),
                    searchTimeMs: searchResult.searchTime,
                    avgSimilarity: searchResult.avgSimilarity,
                    requestedResults: topK,
                    minSimilarity: minSimilarity,
                    actualResults: formattedResults.length
                }
            };
            
            console.log(`✅ Search completed: ${formattedResults.length} results`);
            res.json(responseData);
            
        } else {
            console.log(`❌ Search failed: ${searchResult.error}`);
            res.status(500).json({
                error: 'Search failed',
                details: searchResult.error,
                query: query,
                searchTimeMs: searchResult.searchTime,
                timestamp: new Date().toISOString()
            });
        }
        
    } catch (error) {
        console.error('❌ Search endpoint error:', error);
        res.status(500).json({
            error: 'Search endpoint failed',
            details: error.message,
            query: query,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Comprehensive batch processing endpoint
 */
app.post('/vectorize/batch', async (req, res) => {
    console.log('🔄 Batch vectorization started');
    
    try {
        const files = await fs.readdir('/workspace');
        const results = [];
        const startTime = Date.now();
        
        console.log(`📦 Starting batch processing of ${files.length} files`);
        
        let processedCount = 0;
        let successCount = 0;
        let totalChunks = 0;
        
        for (const filename of files) {
            try {
                const stats = await fs.stat(path.join('/workspace', filename));
                if (stats.isFile()) {
                    processedCount++;
                    console.log(`📝 Batch processing (${processedCount}/${files.length}): ${filename}`);
                    
                    const content = await fs.readFile(path.join('/workspace', filename), 'utf8');
                    const processingResult = await vectorManager.processFile(filename, content);
                    
                    const resultEntry = {
                        filename: filename,
                        status: processingResult.success ? 'success' : 'error',
                        fileSize: stats.size,
                        contentLength: content.length,
                        chunksCreated: processingResult.chunks || 0,
                        processingTimeMs: processingResult.processingTime || 0,
                        error: processingResult.error || null,
                        method: processingResult.method || null,
                        vectorDatabase: processingResult.vectorDatabase || null
                    };
                    
                    results.push(resultEntry);
                    
                    if (processingResult.success) {
                        successCount++;
                        totalChunks += processingResult.chunks || 0;
                    }
                }
            } catch (error) {
                console.error(`❌ Batch processing error for ${filename}:`, error.message);
                results.push({
                    filename: filename,
                    status: 'error',
                    error: error.message,
                    chunksCreated: 0,
                    processingTimeMs: 0
                });
            }
        }
        
        const totalTime = Date.now() - startTime;
        
        const responseData = {
            message: 'Batch vectorization completed',
            results: results,
            summary: {
                totalFiles: results.length,
                processedFiles: processedCount,
                successfulFiles: successCount,
                failedFiles: results.length - successCount,
                totalChunks: totalChunks,
                totalProcessingTimeMs: totalTime,
                avgTimePerFile: results.length > 0 ? Math.round(totalTime / results.length) : 0
            },
            vectorization: {
                embeddingProxy: EMBEDDING_PROXY_URL,
                vectorDatabase: VECTOR_DB_TYPE,
                collection: COLLECTION_NAME
            },
            timestamp: new Date().toISOString()
        };
        
        console.log(`✅ Batch processing completed: ${successCount}/${processedCount} successful`);
        res.json(responseData);
        
    } catch (error) {
        console.error('❌ Batch vectorization failed:', error);
        res.status(500).json({
            error: 'Batch vectorization failed',
            details: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Enhanced delete endpoint with comprehensive cleanup
 */
app.delete('/files/:filename', async (req, res) => {
    const filename = req.params.filename;
    console.log(`🗑️ Delete request for: ${filename}`);
    
    try {
        const filePath = path.join('/workspace', filename);
        
        // Check if file exists before attempting deletion
        try {
            await fs.access(filePath);
        } catch (accessError) {
            return res.status(404).json({ 
                error: 'File not found',
                filename: filename,
                timestamp: new Date().toISOString()
            });
        }
        
        // Delete file from filesystem
        await fs.unlink(filePath);
        console.log(`📁 File deleted from filesystem: ${filename}`);
        
        // Delete vectors via embedding proxy
        const deleteResult = await vectorManager.deleteFileEmbeddings(filename);
        
        const responseData = {
            message: 'File and vectors deleted successfully',
            filename: filename,
            filesystem: {
                status: 'deleted',
                path: filePath
            },
            vectorDeletion: {
                status: deleteResult.success ? 'success' : 'failed',
                error: deleteResult.error || null,
                embeddingProxy: EMBEDDING_PROXY_URL
            },
            timestamp: new Date().toISOString()
        };
        
        console.log(`✅ Delete completed for: ${filename}`);
        res.json(responseData);
        
    } catch (error) {
        console.error(`❌ Delete failed for ${filename}:`, error);
        
        if (error.code === 'ENOENT') {
            res.status(404).json({ 
                error: 'File not found',
                filename: filename,
                timestamp: new Date().toISOString()
            });
        } else if (error.code === 'EACCES') {
            res.status(403).json({ 
                error: 'File access denied',
                filename: filename,
                timestamp: new Date().toISOString()
            });
        } else {
            res.status(500).json({ 
                error: 'Delete operation failed',
                details: error.message,
                filename: filename,
                timestamp: new Date().toISOString()
            });
        }
    }
});

/**
 * Comprehensive vector database statistics endpoint
 */
app.get('/vectors/stats', async (req, res) => {
    console.log('📊 Vector stats requested');
    
    try {
        const [vectorStats, proxyHealth] = await Promise.all([
            vectorManager.getVectorStats(),
            vectorManager.checkEmbeddingProxyHealth()
        ]);
        
        const cacheStats = vectorManager.getCacheStats();
        
        if (vectorStats.success) {
            const responseData = {
                success: true,
                timestamp: new Date().toISOString(),
                // Core vector database info
                vectorDatabase: {
                    type: VECTOR_DB_TYPE,
                    url: VECTOR_DB_URL,
                    collection: COLLECTION_NAME,
                    ...vectorStats.stats
                },
                // Embedding proxy info
                embeddingProxy: {
                    url: EMBEDDING_PROXY_URL,
                    healthy: proxyHealth.healthy,
                    responseTime: proxyHealth.responseTime,
                    error: proxyHealth.error || null
                },
                // File processing cache
                fileCache: cacheStats,
                // Performance metrics
                performance: {
                    buildTime: 'optimized',
                    searchSpeed: 'production-grade',
                    storageType: 'persistent',
                    embeddingMethod: 'prebuilt'
                }
            };
            
            console.log('✅ Vector stats retrieved successfully');
            res.json(responseData);
            
        } else {
            console.log('⚠️ Vector stats retrieval failed');
            res.status(503).json({
                success: false,
                error: vectorStats.error,
                embeddingProxy: EMBEDDING_PROXY_URL,
                vectorDatabase: VECTOR_DB_TYPE,
                timestamp: new Date().toISOString()
            });
        }
        
    } catch (error) {
        console.error('❌ Vector stats endpoint failed:', error);
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Search suggestions endpoint
 */
app.get('/search/suggestions', async (req, res) => {
    try {
        // Enhanced suggestions based on available files and common queries
        const suggestions = [
            "system features and capabilities",
            "vector database configuration",
            "API endpoints and documentation", 
            "installation and setup requirements",
            "troubleshooting and error handling",
            "performance optimization",
            "embedding proxy configuration",
            "prebuilt vector database benefits"
        ];
        
        res.json({
            suggestions: suggestions,
            vectorization: {
                embeddingProxy: EMBEDDING_PROXY_URL,
                vectorDatabase: VECTOR_DB_TYPE,
                collection: COLLECTION_NAME
            },
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('❌ Search suggestions failed:', error);
        res.status(500).json({ 
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Initialize collection endpoint
 */
app.post('/collections/init', async (req, res) => {
    console.log('🗃️ Manual collection initialization requested');
    
    try {
        const result = await vectorManager.initializeVectorCollection();
        
        if (result.success) {
            res.json({
                status: 'success',
                message: result.message,
                collection: COLLECTION_NAME,
                vectorDatabase: VECTOR_DB_TYPE,
                embeddingProxy: EMBEDDING_PROXY_URL,
                timestamp: new Date().toISOString()
            });
        } else {
            res.status(500).json({
                status: 'error',
                error: result.error,
                collection: COLLECTION_NAME,
                timestamp: new Date().toISOString()
            });
        }
        
    } catch (error) {
        console.error('❌ Collection initialization failed:', error);
        res.status(500).json({
            status: 'error',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * System information endpoint
 */
app.get('/system/info', async (req, res) => {
    try {
        const cacheStats = vectorManager.getCacheStats();
        const proxyHealth = await vectorManager.checkEmbeddingProxyHealth();
        
        res.json({
            system: {
                name: 'Enhanced MCP Filesystem Server',
                version: '3.0.0-prebuilt',
                type: 'embedding-proxy-integrated',
                workspace: '/workspace'
            },
            vectorization: {
                enabled: true,
                type: 'prebuilt-proxy',
                embeddingProxy: {
                    url: EMBEDDING_PROXY_URL,
                    healthy: proxyHealth.healthy,
                    capabilities: proxyHealth.data || {}
                },
                vectorDatabase: {
                    type: VECTOR_DB_TYPE,
                    url: VECTOR_DB_URL,
                    collection: COLLECTION_NAME
                }
            },
            performance: {
                buildTime: 'optimized (2-3 minutes)',
                searchSpeed: 'production-grade',
                storageType: 'persistent',
                scalability: 'enterprise-ready'
            },
            cache: cacheStats,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('❌ System info failed:', error);
        res.status(500).json({
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

/**
 * Initialize workspace and services on startup
 */
async function initializeServices() {
    console.log('🚀 Initializing Enhanced MCP Filesystem Server...');
    
    try {
        // Create workspace directory
        await fs.mkdir('/workspace', { recursive: true });
        console.log('✅ Workspace directory initialized: /workspace');
        
        // Check embedding proxy connectivity
        console.log('🔗 Checking embedding proxy connectivity...');
        const proxyHealth = await vectorManager.checkEmbeddingProxyHealth();
        
        if (proxyHealth.healthy) {
            console.log(`✅ Embedding proxy connected successfully`);
            console.log(`   Status: ${proxyHealth.data.status}`);
            console.log(`   Vector DB: ${proxyHealth.data.vector_database}`);
            console.log(`   Embedding method: ${proxyHealth.data.embedding_method}`);
            console.log(`   Model: ${proxyHealth.data.model}`);
            
            // Try to initialize collection
            console.log('🗃️ Initializing vector collection...');
            const initResult = await vectorManager.initializeVectorCollection();
            if (initResult.success) {
                console.log('✅ Vector collection initialized successfully');
            } else {
                console.log(`⚠️ Collection initialization: ${initResult.error}`);
            }
            
        } else {
            console.warn(`⚠️ Embedding proxy not available: ${proxyHealth.error}`);
            console.warn('   Server will continue but vectorization will be limited');
        }
        
        // Load existing file metadata
        console.log('📁 Scanning existing files...');
        try {
            const files = await fs.readdir('/workspace');
            console.log(`📁 Found ${files.length} existing files in workspace`);
            
            // Initialize metadata cache for existing files
            let fileCount = 0;
            for (const file of files) {
                try {
                    const stats = await fs.stat(path.join('/workspace', file));
                    if (stats.isFile()) {
                        fileCount++;
                        fileMetadataCache.set(file, {
                            processed: false, // Will be detected/updated during operations
                            lastModified: stats.mtime.toISOString(),
                            fileSize: stats.size,
                            discovered: new Date().toISOString()
                        });
                    }
                } catch (fileError) {
                    console.warn(`⚠️ Cannot access file ${file}: ${fileError.message}`);
                }
            }
            
            console.log(`✅ Initialized metadata cache for ${fileCount} files`);
            
        } catch (error) {
            console.error('❌ Error scanning existing files:', error.message);
        }
        
        console.log('🎉 Enhanced MCP Filesystem Server initialization complete!');
        console.log('');
        console.log('📊 Configuration Summary:');
        console.log(`   Embedding Proxy: ${EMBEDDING_PROXY_URL}`);
        console.log(`   Vector Database: ${VECTOR_DB_TYPE} at ${VECTOR_DB_URL}`);
        console.log(`   Collection: ${COLLECTION_NAME}`);
        console.log(`   Workspace: /workspace`);
        console.log('');
        
    } catch (error) {
        console.error('❌ Error during service initialization:', error);
        console.error('   Server will start but functionality may be limited');
    }
}

// Enhanced error handling middleware
app.use((error, req, res, next) => {
    console.error('🚨 Unhandled error:', error);
    
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ 
                error: 'File too large. Maximum size is 50MB.',
                maxSize: '50MB',
                timestamp: new Date().toISOString()
            });
        }
        if (error.code === 'LIMIT_UNEXPECTED_FILE') {
            return res.status(400).json({ 
                error: 'Unexpected file field',
                expectedField: 'file',
                timestamp: new Date().toISOString()
            });
        }
    }
    
    res.status(500).json({ 
        error: 'Internal server error',
        details: process.env.NODE_ENV === 'development' ? error.message : 'An unexpected error occurred',
        timestamp: new Date().toISOString()
    });
});

// Handle 404 routes
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        path: req.originalUrl,
        method: req.method,
        timestamp: new Date().toISOString(),
        availableEndpoints: [
            'GET /status',
            'GET /health', 
            'GET /files',
            'POST /files',
            'GET /files/:filename',
            'DELETE /files/:filename',
            'POST /search',
            'POST /vectorize/batch',
            'GET /vectors/stats',
            'GET /search/suggestions',
            'POST /collections/init',
            'GET /system/info'
        ]
    });
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
    console.log('📴 Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('📴 Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

// Start server with enhanced initialization
app.listen(PORT, async () => {
    console.log('🚀 Enhanced MCP Filesystem Server starting...');
    console.log(`📡 Server running on port ${PORT}`);
    console.log('');
    
    await initializeServices();
    
    console.log('');
    console.log('🌐 Server ready! Available endpoints:');
    console.log(`   Health: http://localhost:${PORT}/health`);
    console.log(`   Status: http://localhost:${PORT}/status`);
    console.log(`   Files: http://localhost:${PORT}/files`);
    console.log(`   Stats: http://localhost:${PORT}/vectors/stats`);
    console.log('');
});
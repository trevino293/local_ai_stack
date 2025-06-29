const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

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

// Routes

// Status endpoint
app.get('/status', (req, res) => {
    res.json({ 
        status: 'online',
        version: '1.0.0',
        workspace: '/workspace'
    });
});

// List files
app.get('/files', async (req, res) => {
    try {
        const files = await fs.readdir('/workspace');
        const fileList = [];
        
        for (const file of files) {
            const stats = await fs.stat(path.join('/workspace', file));
            if (stats.isFile()) {
                fileList.push(file);
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

// Upload file
app.post('/files', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    res.json({ 
        message: 'File uploaded successfully',
        filename: req.file.filename 
    });
});

// Delete file
app.delete('/files/:filename', async (req, res) => {
    try {
        const filePath = path.join('/workspace', req.params.filename);
        await fs.unlink(filePath);
        res.json({ message: 'File deleted successfully' });
    } catch (error) {
        res.status(404).json({ error: 'File not found' });
    }
});

// Create workspace directory if it doesn't exist
async function initWorkspace() {
    try {
        await fs.mkdir('/workspace', { recursive: true });
        console.log('Workspace directory initialized');
    } catch (error) {
        console.error('Error creating workspace:', error);
    }
}

// Start server
app.listen(PORT, async () => {
    await initWorkspace();
    console.log(`MCP Filesystem Server running on port ${PORT}`);
});
// Global state
let selectedModel = 'llama2';
let selectedFiles = [];
let chatHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadFiles();
    loadChatHistory();
    checkStatus();
    
    // Set up event listeners
    document.getElementById('modelSelect').addEventListener('change', (e) => {
        selectedModel = e.target.value;
    });
    
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    document.getElementById('messageInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            sendMessage();
        }
    });
    
    // Check status every 10 seconds
    setInterval(checkStatus, 10000);
});

// Load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name} (${formatSize(model.size)})`;
                select.appendChild(option);
            });
            selectedModel = data.models[0].name;
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            select.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Load context files
async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        const files = await response.json();
        
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        if (files.error) {
            fileList.innerHTML = '<p>Unable to load files</p>';
            return;
        }
        
        files.forEach(file => {
            const item = document.createElement('div');
            item.className = 'file-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = file;
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    selectedFiles.push(file);
                } else {
                    selectedFiles = selectedFiles.filter(f => f !== file);
                }
            });
            
            const label = document.createElement('label');
            label.textContent = file;
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = () => deleteFile(file);
            
            item.appendChild(checkbox);
            item.appendChild(label);
            item.appendChild(deleteBtn);
            fileList.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/files', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            loadFiles();
            event.target.value = '';
        } else {
            alert('Failed to upload file');
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
    }
}

// Delete file
async function deleteFile(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    
    try {
        const response = await fetch(`/api/files/${filename}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            loadFiles();
            selectedFiles = selectedFiles.filter(f => f !== filename);
        }
    } catch (error) {
        console.error('Error deleting file:', error);
    }
}

// Send message
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    // Add user message to chat
    addMessageToChat('user', message);
    messageInput.value = '';
    
    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'spinner';
    loadingDiv.id = 'loading';
    document.getElementById('chatMessages').appendChild(loadingDiv);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: selectedModel,
                message: message,
                context_files: selectedFiles
            })
        });
        
        const data = await response.json();
        
        // Remove loading
        const loading = document.getElementById('loading');
        if (loading) loading.remove();
        
        if (data.error) {
            addMessageToChat('assistant', `Error: ${data.error}`);
        } else {
            addMessageToChat('assistant', data.response);
            loadChatHistory();
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToChat('assistant', 'Error: Failed to send message');
    } finally {
        sendButton.disabled = false;
    }
}

// Add message to chat
function addMessageToChat(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <span>${new Date().toLocaleTimeString()}</span>
    `;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Load chat history
async function loadChatHistory() {
    try {
        const response = await fetch('/api/chat/history');
        const history = await response.json();
        
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';
        
        history.reverse().forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.onclick = () => showHistoryItem(history.length - 1 - index);
            
            const timestamp = document.createElement('div');
            timestamp.className = 'history-timestamp';
            timestamp.textContent = new Date(item.timestamp).toLocaleString();
            
            const preview = document.createElement('div');
            preview.className = 'history-preview';
            preview.textContent = item.message.substring(0, 50) + '...';
            
            historyItem.appendChild(timestamp);
            historyItem.appendChild(preview);
            historyList.appendChild(historyItem);
        });
        
        chatHistory = history;
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Show history item
function showHistoryItem(index) {
    const item = chatHistory[index];
    if (!item) return;
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    addMessageToChat('user', item.message);
    addMessageToChat('assistant', item.response);
}

// Check system status
async function checkStatus() {
    // Check Ollama status
    try {
        const response = await fetch('/api/models');
        const ollamaStatus = document.getElementById('ollamaStatus');
        const ollamaDot = document.getElementById('ollamaDot');
        
        if (response.ok) {
            ollamaStatus.textContent = 'Online';
            ollamaDot.classList.add('online');
        } else {
            ollamaStatus.textContent = 'Offline';
            ollamaDot.classList.remove('online');
        }
    } catch (error) {
        document.getElementById('ollamaStatus').textContent = 'Offline';
        document.getElementById('ollamaDot').classList.remove('online');
    }
    
    // Check MCP status
    try {
        const response = await fetch('/api/mcp/status');
        const mcpStatus = document.getElementById('mcpStatus');
        const mcpDot = document.getElementById('mcpDot');
        
        if (response.ok) {
            mcpStatus.textContent = 'Online';
            mcpDot.classList.add('online');
        } else {
            mcpStatus.textContent = 'Offline';
            mcpDot.classList.remove('online');
        }
    } catch (error) {
        document.getElementById('mcpStatus').textContent = 'Offline';
        document.getElementById('mcpDot').classList.remove('online');
    }
}

// Utility functions
function formatSize(bytes) {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}
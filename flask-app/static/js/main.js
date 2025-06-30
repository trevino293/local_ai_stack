// Global state
let selectedModel = 'llama2';
let selectedFiles = [];
let systemFiles = []; // Files that are always included
let chatHistory = [];

// Model parameters
let modelParams = {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    seed: -1,
    num_predict: -1
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadFiles();
    loadChatHistory();
    checkStatus();
    initializeModelParams();
    
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
    
    // Model parameter event listeners
    setupModelParamListeners();
    
    // Check status every 10 seconds
    setInterval(checkStatus, 10000);
});

// Initialize model parameters UI
function initializeModelParams() {
    document.getElementById('temperature').value = modelParams.temperature;
    document.getElementById('top_p').value = modelParams.top_p;
    document.getElementById('top_k').value = modelParams.top_k;
    document.getElementById('repeat_penalty').value = modelParams.repeat_penalty;
    document.getElementById('seed').value = modelParams.seed;
    document.getElementById('num_predict').value = modelParams.num_predict;
    
    updateParamDisplays();
}

// Setup model parameter event listeners
function setupModelParamListeners() {
    const params = ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'num_predict'];
    
    params.forEach(param => {
        const element = document.getElementById(param);
        if (element) {
            element.addEventListener('input', (e) => {
                const value = param === 'seed' || param === 'num_predict' || param === 'top_k' 
                    ? parseInt(e.target.value) 
                    : parseFloat(e.target.value);
                modelParams[param] = value;
                updateParamDisplays();
            });
        }
    });
    
    // Reset to defaults button
    document.getElementById('resetParams').addEventListener('click', resetModelParams);
}

// Update parameter display values
function updateParamDisplays() {
    document.getElementById('temperatureValue').textContent = modelParams.temperature.toFixed(2);
    document.getElementById('topPValue').textContent = modelParams.top_p.toFixed(2);
    document.getElementById('topKValue').textContent = modelParams.top_k;
    document.getElementById('repeatPenaltyValue').textContent = modelParams.repeat_penalty.toFixed(2);
    document.getElementById('seedValue').textContent = modelParams.seed === -1 ? 'Random' : modelParams.seed;
    document.getElementById('numPredictValue').textContent = modelParams.num_predict === -1 ? 'Auto' : modelParams.num_predict;
}

// Reset model parameters to defaults
function resetModelParams() {
    modelParams = {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: -1,
        num_predict: -1
    };
    initializeModelParams();
}

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
        
        // Identify system files (files that should always be included)
        systemFiles = files.filter(file => 
            file.toLowerCase().includes('admin') || 
            file.toLowerCase().includes('system') ||
            file.toLowerCase().includes('default') ||
            file.toLowerCase().includes('config')
        );
        
        // Add system files to selected files if not already present
        systemFiles.forEach(file => {
            if (!selectedFiles.includes(file)) {
                selectedFiles.push(file);
            }
        });
        
        files.forEach(file => {
            const item = document.createElement('div');
            item.className = 'file-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = file;
            
            // System files are always checked and disabled
            const isSystemFile = systemFiles.includes(file);
            if (isSystemFile) {
                checkbox.checked = true;
                checkbox.disabled = true;
                item.classList.add('system-file');
            } else {
                checkbox.checked = selectedFiles.includes(file);
                checkbox.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        selectedFiles.push(file);
                    } else {
                        selectedFiles = selectedFiles.filter(f => f !== file);
                    }
                });
            }
            
            const label = document.createElement('label');
            label.textContent = file;
            if (isSystemFile) {
                label.innerHTML += ' <span class="system-badge">SYSTEM</span>';
            }
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.disabled = isSystemFile; // Prevent deletion of system files
            deleteBtn.onclick = () => deleteFile(file);
            
            item.appendChild(checkbox);
            item.appendChild(label);
            if (!isSystemFile) { // Only show delete button for non-system files
                item.appendChild(deleteBtn);
            }
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
    if (systemFiles.includes(filename)) {
        alert('Cannot delete system files');
        return;
    }
    
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
                context_files: selectedFiles,
                model_params: modelParams
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
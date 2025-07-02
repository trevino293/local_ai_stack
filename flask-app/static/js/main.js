// Simplified main.js - Fixed loading and clean interface
let selectedModel = 'llama2';
let selectedFiles = [];
let systemFiles = [];
let chatHistory = [];
let savedConfigurations = [];
let currentActiveConfig = null;
let currentConversationId = 'default';
let currentProcessingMode = 'fast';
let config_id_counter = 1;

// Model parameters with presets
let modelParams = {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    seed: -1,
    num_predict: -1
};

const presets = {
    creative: {
        temperature: 1.2,
        top_p: 0.95,
        top_k: 50,
        repeat_penalty: 1.0,
        seed: -1,
        num_predict: -1
    },
    balanced: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: -1,
        num_predict: -1
    },
    precise: {
        temperature: 0.2,
        top_p: 0.7,
        top_k: 20,
        repeat_penalty: 1.2,
        seed: -1,
        num_predict: -1
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Local AI Stack...');
    initializeApplication();
    setupEventListeners();
    startSystemMonitoring();
});

async function initializeApplication() {
    try {
        await loadModels();
        await loadFiles();
        await loadChatHistory();
        await loadSavedConfigurations();
        initializeModelParams();
        await checkSystemStatus();
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Application initialization error:', error);
    }
}

// Event Listeners
function setupEventListeners() {
    // Model selection
    document.getElementById('modelSelect').addEventListener('change', handleModelChange);
    
    // Configuration
    document.getElementById('configToggle').addEventListener('click', handleConfigToggle);
    document.addEventListener('click', handleGlobalClick);
    
    // File management
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);
    
    // Chat
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    document.getElementById('messageInput').addEventListener('keydown', handleMessageInput);
    
    // Processing mode
    document.getElementById('fastModeBtn').addEventListener('click', () => setProcessingMode('fast'));
    document.getElementById('detailedModeBtn').addEventListener('click', () => setProcessingMode('detailed'));
    
    // Clear history
    document.getElementById('clearHistoryBtn').addEventListener('click', clearChatHistory);
    
    // Parameter controls
    setupParameterEventListeners();
    setupPresetEventListeners();
    
    // Configuration management
    document.getElementById('saveParams').addEventListener('click', saveModelParameters);
    document.getElementById('resetParams').addEventListener('click', resetToDefaults);
    document.getElementById('newConfigBtn').addEventListener('click', showConfigSaveSection);
    document.getElementById('saveNewConfigBtn').addEventListener('click', saveNewConfiguration);
    document.getElementById('cancelSaveBtn').addEventListener('click', hideConfigSaveSection);
}

// Model Management
async function loadModels() {
    try {
        console.log('Loading models from Ollama...');
        const response = await fetch('/api/models');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name} (${formatFileSize(model.size || 0)})`;
                modelSelect.appendChild(option);
            });
            selectedModel = data.models[0].name;
            console.log(`Loaded ${data.models.length} models, selected: ${selectedModel}`);
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSelect.appendChild(option);
            console.warn('No models found');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

function handleModelChange(event) {
    selectedModel = event.target.value;
    console.log('Model changed to:', selectedModel);
}

// File Management
async function loadFiles() {
    try {
        console.log('Loading files...');
        const response = await fetch('/api/files');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const files = await response.json();
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        if (files.error) {
            fileList.innerHTML = '<p>Unable to load context files</p>';
            return;
        }
        
        const fileArray = Array.isArray(files) ? files : [];
        
        if (fileArray.length === 0) {
            fileList.innerHTML = '<p>No files uploaded yet</p>';
            updateVectorStats(0, 0);
            return;
        }
        
        // Clear previous selections
        systemFiles = [];
        
        // Process enhanced file data
        let totalVectors = 0;
        let vectorizedFiles = 0;
        
        fileArray.forEach(fileInfo => {
            let filename, isVectorized = false, chunkCount = 0;
            
            // Handle both simple strings and enhanced objects
            if (typeof fileInfo === 'string') {
                filename = fileInfo;
            } else {
                filename = fileInfo.name || fileInfo.filename;
                isVectorized = fileInfo.vectorized || false;
                chunkCount = fileInfo.chunkCount || 0;
            }
            
            // Identify system files
            if (filename && (
                filename.toLowerCase().includes('admin') || 
                filename.toLowerCase().includes('system') ||
                filename.toLowerCase().includes('config')
            )) {
                systemFiles.push(filename);
                
                // Auto-select system files if not already selected
                if (!selectedFiles.includes(filename)) {
                    selectedFiles.push(filename);
                }
            }
            
            // Count vectors
            if (isVectorized) {
                vectorizedFiles++;
                totalVectors += chunkCount;
            }
        });
        
        // Render files
        fileArray.forEach(fileInfo => {
            const fileItem = createFileListItem(fileInfo);
            fileList.appendChild(fileItem);
        });
        
        // Update vector stats with actual data
        updateVectorStats(totalVectors, vectorizedFiles);
        
        console.log(`Loaded ${fileArray.length} files (${systemFiles.length} system files, ${vectorizedFiles} vectorized)`);
        
    } catch (error) {
        console.error('Error loading files:', error);
        document.getElementById('fileList').innerHTML = '<p style="color: var(--error);">Failed to load files</p>';
        updateVectorStats(0, 0);
    }
}

function createFileListItem(filename) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const isSystemFile = systemFiles.includes(filename);
    if (isSystemFile) {
        fileItem.classList.add('system-file');
    }
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = filename;
    checkbox.checked = selectedFiles.includes(filename);
    checkbox.disabled = isSystemFile;
    
    if (!isSystemFile) {
        checkbox.addEventListener('change', (event) => {
            handleFileSelection(filename, event.target.checked);
        });
    }
    
    const label = document.createElement('label');
    label.style.cssText = 'flex: 1; margin-left: 8px; cursor: pointer;';
    label.textContent = filename;
    
    if (isSystemFile) {
        label.innerHTML += ' <span class="system-badge">SYSTEM</span>';
    }
    
    fileItem.appendChild(checkbox);
    fileItem.appendChild(label);
    
    if (!isSystemFile) {
        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.onclick = () => deleteFile(filename);
        fileItem.appendChild(deleteButton);
    }
    
    return fileItem;
}

function handleFileSelection(filename, isSelected) {
    if (isSelected) {
        if (!selectedFiles.includes(filename)) {
            selectedFiles.push(filename);
        }
    } else {
        selectedFiles = selectedFiles.filter(file => file !== filename);
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadLabel = document.querySelector('.file-input-label');
    const originalText = uploadLabel.textContent;
    uploadLabel.textContent = '🔄 Uploading...';
    uploadLabel.style.background = 'var(--system-accent)';
    
    try {
        const response = await fetch('/api/files', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            await loadFiles();
            event.target.value = '';
            uploadLabel.textContent = '✅ Uploaded!';
            uploadLabel.style.background = '#10b981';
            console.log('File uploaded successfully:', file.name);
        } else {
            uploadLabel.textContent = '❌ Upload failed';
            uploadLabel.style.background = '#ef4444';
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadLabel.textContent = '❌ Network error';
        uploadLabel.style.background = '#ef4444';
    } finally {
        setTimeout(() => {
            uploadLabel.textContent = originalText;
            uploadLabel.style.background = '';
        }, 3000);
    }
}

async function deleteFile(filename) {
    if (!confirm(`Delete ${filename}? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/files/${filename}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            await loadFiles();
            selectedFiles = selectedFiles.filter(file => file !== filename);
            console.log('File deleted:', filename);
        } else {
            alert('Failed to delete file');
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Error deleting file');
    }
}

// Chat Management
function handleMessageInput(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
        sendMessage();
    }
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const messageText = messageInput.value.trim();
    
    if (!messageText) return;
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    // Add user message
    addMessageToInterface('user', messageText);
    messageInput.value = '';
    
    // Show loading indicator
    showDeliberationIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: selectedModel,
                message: messageText,
                context_files: selectedFiles,
                model_params: modelParams,
                conversation_id: currentConversationId,
                fast_mode: currentProcessingMode === 'fast'
            })
        });
        
        const data = await response.json();
        removeDeliberationIndicator();
        
        if (data.error) {
            addMessageToInterface('assistant', `Error: ${data.error}`);
        } else {
            addEnhancedMessageToInterface('assistant', data);
            await loadChatHistory();
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeDeliberationIndicator();
        addMessageToInterface('assistant', 'Error: Failed to communicate with AI service');
    } finally {
        sendButton.disabled = false;
    }
}

function addMessageToInterface(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    
    const messageHeader = document.createElement('div');
    messageHeader.className = 'message-header';
    messageHeader.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
    `;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    chatMessages.appendChild(messageElement);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addEnhancedMessageToInterface(role, responseData) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role} enhanced-message`;
    
    const confidence = responseData.metadata?.confidence_score || 7;
    const chunksUsed = responseData.metadata?.context_chunks_used || 0;
    const processingMode = responseData.metadata?.processing_mode || 'standard';
    
    const messageHeader = document.createElement('div');
    messageHeader.className = 'message-header';
    messageHeader.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <div style="display: flex; gap: 10px; font-size: 0.8rem;">
            <span style="color: #10b981;">Confidence: ${confidence}/10</span>
            <span style="color: var(--accent-blue);">Mode: ${processingMode}</span>
            <span style="color: var(--system-accent);">Chunks: ${chunksUsed}</span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        </div>
    `;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = responseData.response || 'Response generated';
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    
    // Add citations if available
    if (responseData.citations && responseData.citations.length > 0) {
        const citationsSection = createCitationsSection(responseData.citations);
        messageElement.appendChild(citationsSection);
    }
    
    // Add reasoning if available (detailed mode)
    if (responseData.metadata?.reasoning_chain) {
        const reasoningSection = createReasoningSection(responseData.metadata.reasoning_chain);
        messageElement.appendChild(reasoningSection);
    }
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createCitationsSection(citations) {
    const section = document.createElement('div');
    section.style.cssText = `
        margin-top: 10px; padding: 10px; background: var(--bg-primary);
        border-radius: 6px; border: 1px solid var(--border-color);
    `;
    
    const header = document.createElement('div');
    header.style.cssText = 'font-weight: 600; margin-bottom: 8px; color: var(--accent-light-blue);';
    header.textContent = `📚 Sources (${citations.length})`;
    
    const citationsList = document.createElement('div');
    citationsList.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px;';
    
    citations.forEach(citation => {
        const citationItem = document.createElement('span');
        citationItem.style.cssText = `
            background: var(--bg-secondary); padding: 4px 8px; border-radius: 12px;
            font-size: 0.8rem; border: 1px solid var(--border-color);
        `;
        citationItem.innerHTML = `
            <span style="background: ${citation.type === 'SYSTEM' ? 'var(--system-accent)' : 'var(--accent-blue)'}; 
                         color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem; margin-right: 6px;">
                ${citation.type}
            </span>
            ${citation.file}
        `;
        citationsList.appendChild(citationItem);
    });
    
    section.appendChild(header);
    section.appendChild(citationsList);
    return section;
}

function createReasoningSection(reasoningChain) {
    const section = document.createElement('div');
    section.style.cssText = `
        margin-top: 10px; padding: 15px; background: var(--bg-tertiary);
        border-radius: 8px; border: 1px solid var(--system-accent);
    `;
    
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex; justify-content: space-between; align-items: center;
        cursor: pointer; margin-bottom: 10px;
    `;
    header.innerHTML = `
        <span style="font-weight: 600; color: var(--system-accent);">🧠 Reasoning Process</span>
        <button onclick="toggleDeliberation(this)" style="background: none; border: none; color: var(--text-secondary); cursor: pointer;">
            <svg class="chevron-icon" width="16" height="16" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
        </button>
    `;
    
    const content = document.createElement('div');
    content.className = 'deliberation-content collapsed';
    content.style.cssText = 'max-height: 0; overflow: hidden; transition: max-height 0.3s ease;';
    
    if (Array.isArray(reasoningChain)) {
        reasoningChain.forEach((step, index) => {
            const stepElement = document.createElement('div');
            stepElement.style.cssText = `
                margin-bottom: 8px; padding: 8px; background: var(--bg-secondary);
                border-radius: 4px; font-size: 0.85rem;
            `;
            stepElement.innerHTML = `
                <strong>Step ${index + 1}:</strong> ${typeof step === 'object' ? JSON.stringify(step, null, 2) : step}
            `;
            content.appendChild(stepElement);
        });
    }
    
    section.appendChild(header);
    section.appendChild(content);
    return section;
}

function showDeliberationIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    
    const indicator = document.createElement('div');
    indicator.id = 'deliberation-indicator';
    indicator.style.cssText = `
        padding: 15px; background: var(--bg-secondary); border-radius: 8px;
        margin: 15px 0; border: 1px solid var(--border-color);
    `;
    indicator.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="spinner"></div>
            <span style="color: var(--text-primary);">Processing with vector search...</span>
        </div>
    `;
    
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeDeliberationIndicator() {
    const indicator = document.getElementById('deliberation-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Chat History
async function loadChatHistory() {
    try {
        const response = await fetch(`/api/chat/history?conversation_id=${currentConversationId}`);
        const historyData = await response.json();
        
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';
        
        if (historyData.length === 0) {
            historyList.innerHTML = '<p>No chat history yet</p>';
            return;
        }
        
        historyData.reverse().slice(0, 10).forEach((item, index) => {
            const historyElement = document.createElement('div');
            historyElement.className = 'history-item';
            historyElement.style.cssText = `
                padding: 10px; background: var(--bg-tertiary); margin-bottom: 8px;
                border-radius: 5px; cursor: pointer; border: 1px solid var(--border-color);
            `;
            historyElement.innerHTML = `
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">
                    ${new Date(item.timestamp).toLocaleString()}
                </div>
                <div style="color: var(--text-primary); font-size: 0.9rem;">
                    ${item.message.substring(0, 50)}${item.message.length > 50 ? '...' : ''}
                </div>
            `;
            historyElement.onclick = () => displayHistoryItem(item);
            historyList.appendChild(historyElement);
        });
        
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

function displayHistoryItem(item) {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    addMessageToInterface('user', item.message);
    if (item.metadata) {
        addEnhancedMessageToInterface('assistant', item);
    } else {
        addMessageToInterface('assistant', item.response);
    }
}

async function clearChatHistory() {
    if (!confirm('Clear all chat history? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/chat/history', {
            method: 'DELETE'
        });
        
        if (response.ok) {
            document.getElementById('historyList').innerHTML = '<p>No chat history yet</p>';
            document.getElementById('chatMessages').innerHTML = `
                <div class="message assistant">
                    <div class="message-header">
                        <span>Assistant</span>
                        <span class="timestamp">Ready</span>
                    </div>
                    <div class="message-content">Chat history cleared. How can I help you today?</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error clearing history:', error);
    }
}

// Processing Mode
function setProcessingMode(mode) {
    currentProcessingMode = mode;
    
    const fastBtn = document.getElementById('fastModeBtn');
    const detailedBtn = document.getElementById('detailedModeBtn');
    
    if (mode === 'fast') {
        fastBtn.style.background = 'var(--accent-blue)';
        fastBtn.style.color = 'white';
        detailedBtn.style.background = 'var(--bg-tertiary)';
        detailedBtn.style.color = 'var(--text-secondary)';
    } else {
        detailedBtn.style.background = 'var(--system-accent)';
        detailedBtn.style.color = 'white';
        fastBtn.style.background = 'var(--bg-tertiary)';
        fastBtn.style.color = 'var(--text-secondary)';
    }
    
    console.log('Processing mode set to:', mode);
}

// System Status Monitoring
function startSystemMonitoring() {
    checkSystemStatus();
    setInterval(checkSystemStatus, 30000); // Check every 30 seconds
}

async function checkSystemStatus() {
    await Promise.all([
        checkOllamaStatus(),
        checkMCPStatus(),
        checkEmbeddingStatus()
    ]);
}

async function checkOllamaStatus() {
    try {
        const response = await fetch('/api/models', { signal: AbortSignal.timeout(5000) });
        updateStatusIndicator('ollama', response.ok);
    } catch (error) {
        updateStatusIndicator('ollama', false);
    }
}

async function checkMCPStatus() {
    try {
        const response = await fetch('/api/mcp/status', { signal: AbortSignal.timeout(5000) });
        updateStatusIndicator('mcp', response.ok);
    } catch (error) {
        updateStatusIndicator('mcp', false);
    }
}

async function checkEmbeddingStatus() {
    try {
        const response = await fetch('/api/embedding/health', { signal: AbortSignal.timeout(5000) });
        updateStatusIndicator('embedding', response.ok);
    } catch (error) {
        updateStatusIndicator('embedding', false);
    }
}

function updateStatusIndicator(service, isOnline) {
    const statusText = document.getElementById(`${service}Status`);
    const statusDot = document.getElementById(`${service}Dot`);
    
    if (statusText && statusDot) {
        if (isOnline) {
            statusText.textContent = 'Online';
            statusDot.classList.add('online');
        } else {
            statusText.textContent = 'Offline';
            statusDot.classList.remove('online');
        }
    }
}

// Vector Database Stats
function updateVectorStats(fileCount) {
    const totalVectorsEl = document.getElementById('totalVectors');
    const indexedFilesEl = document.getElementById('indexedFiles');
    
    if (totalVectorsEl) totalVectorsEl.textContent = fileCount * 10; // Estimate
    if (indexedFilesEl) indexedFilesEl.textContent = fileCount;
}

// Configuration Management
function handleConfigToggle(event) {
    event.stopPropagation();
    const configButton = document.querySelector('.config-button');
    const configDropdown = document.getElementById('configDropdown');
    
    configButton.classList.toggle('active');
    configDropdown.classList.toggle('show');
}

function handleGlobalClick(event) {
    if (!event.target.closest('.config-button')) {
        const configButton = document.querySelector('.config-button');
        const configDropdown = document.getElementById('configDropdown');
        
        configButton.classList.remove('active');
        configDropdown.classList.remove('show');
    }
}

// Parameter Management
function setupParameterEventListeners() {
    const params = ['temperature', 'top_p', 'top_k', 'repeat_penalty'];
    
    params.forEach(paramName => {
        const element = document.getElementById(paramName);
        if (element) {
            element.addEventListener('input', (event) => {
                const value = parseFloat(event.target.value);
                modelParams[paramName] = value;
                updateParameterDisplays();
                clearActivePresetSelection();
            });
        }
    });
}

function setupPresetEventListeners() {
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', () => {
            const presetName = button.dataset.preset;
            if (presets[presetName]) {
                modelParams = { ...presets[presetName] };
                initializeModelParams();
                setActivePresetButton(button);
            }
        });
    });
}

function initializeModelParams() {
    document.getElementById('temperature').value = modelParams.temperature;
    document.getElementById('top_p').value = modelParams.top_p;
    document.getElementById('top_k').value = modelParams.top_k;
    document.getElementById('repeat_penalty').value = modelParams.repeat_penalty;
    
    updateParameterDisplays();
}

function updateParameterDisplays() {
    document.getElementById('temperatureValue').textContent = modelParams.temperature.toFixed(2);
    document.getElementById('topPValue').textContent = modelParams.top_p.toFixed(2);
    document.getElementById('topKValue').textContent = modelParams.top_k;
    document.getElementById('repeatPenaltyValue').textContent = modelParams.repeat_penalty.toFixed(2);
}

function setActivePresetButton(activeButton) {
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.classList.remove('active');
    });
    activeButton.classList.add('active');
}

function clearActivePresetSelection() {
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.classList.remove('active');
    });
}

async function saveModelParameters() {
    try {
        const response = await fetch('/api/model-params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                params: modelParams
            })
        });
        
        const result = await response.json();
        showSaveParametersFeedback(result.status === 'success');
    } catch (error) {
        console.error('Error saving parameters:', error);
        showSaveParametersFeedback(false);
    }
}

function resetToDefaults() {
    modelParams = { ...presets.balanced };
    initializeModelParams();
    clearActivePresetSelection();
    document.querySelector('[data-preset="balanced"]').classList.add('active');
}

function showSaveParametersFeedback(success) {
    const saveButton = document.getElementById('saveParams');
    const originalText = saveButton.textContent;
    
    if (success) {
        saveButton.textContent = 'Saved!';
        saveButton.style.background = '#059669';
    } else {
        saveButton.textContent = 'Error';
        saveButton.style.background = '#ef4444';
    }
    
    setTimeout(() => {
        saveButton.textContent = originalText;
        saveButton.style.background = '';
    }, 2000);
}

// Saved Configurations
async function loadSavedConfigurations() {
    try {
        const response = await fetch('/api/saved-configs');
        const data = await response.json();
        
        if (data.configurations) {
            savedConfigurations = data.configurations;
            renderSavedConfigurationsList();
        }
    } catch (error) {
        console.error('Error loading saved configurations:', error);
    }
}

function renderSavedConfigurationsList() {
    const configsList = document.getElementById('savedConfigsList');
    
    if (savedConfigurations.length === 0) {
        configsList.innerHTML = '<p class="no-configs">No saved configurations</p>';
        return;
    }
    
    configsList.innerHTML = '';
    
    savedConfigurations.forEach(config => {
        const configItem = document.createElement('div');
        configItem.className = 'saved-config-item';
        configItem.innerHTML = `
            <div class="config-item-info" style="cursor: pointer; flex: 1;">
                <div class="config-item-name">${config.name}</div>
                <div class="config-item-details">T:${config.params.temperature} • P:${config.params.top_p}</div>
            </div>
            <button class="config-delete-btn" onclick="deleteSavedConfiguration(${config.id})">×</button>
        `;
        
        configItem.querySelector('.config-item-info').onclick = () => applySavedConfiguration(config);
        configsList.appendChild(configItem);
    });
}

function showConfigSaveSection() {
    document.getElementById('configSaveSection').style.display = 'block';
    document.getElementById('configNameInput').focus();
}

function hideConfigSaveSection() {
    document.getElementById('configSaveSection').style.display = 'none';
    document.getElementById('configNameInput').value = '';
}

async function saveNewConfiguration() {
    const nameInput = document.getElementById('configNameInput');
    const configName = nameInput.value.trim();
    
    if (!configName) {
        nameInput.focus();
        return;
    }
    
    try {
        const response = await fetch('/api/saved-configs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: configName,
                params: modelParams,
                model: selectedModel
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            await loadSavedConfigurations();
            hideConfigSaveSection();
        } else {
            alert(result.error || 'Failed to save configuration');
        }
    } catch (error) {
        console.error('Error saving configuration:', error);
        alert('Network error occurred');
    }
}

function applySavedConfiguration(config) {
    modelParams = { ...config.params };
    initializeModelParams();
    clearActivePresetSelection();
    console.log('Applied configuration:', config.name);
}

async function deleteSavedConfiguration(configId) {
    const config = savedConfigurations.find(c => c.id === configId);
    if (!config) return;
    
    if (!confirm(`Delete configuration "${config.name}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/saved-configs/${configId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            await loadSavedConfigurations();
        } else {
            alert('Failed to delete configuration');
        }
    } catch (error) {
        console.error('Error deleting configuration:', error);
        alert('Error occurred during deletion');
    }
}

// Utility Functions
function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    
    const unitIndex = Math.floor(Math.log(bytes) / Math.log(1024));
    const size = Math.round(bytes / Math.pow(1024, unitIndex) * 100) / 100;
    
    return `${size} ${units[unitIndex]}`;
}

console.log('Simplified Local AI Stack main.js loaded successfully!');
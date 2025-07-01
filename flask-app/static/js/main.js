// Global state management
let selectedModel = 'llama2';
let selectedFiles = [];
let systemFiles = [];
let chatHistory = [];
let savedConfigurations = [];
let currentActiveConfig = null;
let currentConversationId = 'default';
let conversationState = 'active'; // 'active', 'new', 'loading'

// Enhanced vectorization globals
let currentProcessingMode = 'fast';
let vectorizationStats = {
    totalVectors: 0,
    indexedFiles: 0,
    avgSimilarity: 0,
    searchTime: 0
};

// Model parameters with comprehensive preset system
let modelParams = {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    seed: -1,
    num_predict: -1
};

// Predefined parameter presets for different use cases
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

// Application initialization
document.addEventListener('DOMContentLoaded', () => {
    initializeApplication();
    setupEventListeners();
    startSystemMonitoring();
    addNewConversationButton();
    restoreConversationState();
});

async function initializeApplication() {
    try {
        await loadModels();
        await loadFiles();
        await loadChatHistory();
        await loadSavedParameters();
        await loadSavedConfigurations();
        initializeModelParams();
        await checkEnhancedSystemStatus();
        initializeEnhancedFeatures(); // Initialize enhanced features
    } catch (error) {
        console.error('Application initialization error:', error);
        displaySystemError('Failed to initialize application components');
    }
}

function setupEventListeners() {
    // Model selection and configuration
    document.getElementById('modelSelect').addEventListener('change', handleModelChange);
    document.getElementById('configToggle').addEventListener('click', handleConfigToggle);
    
    // Configuration dropdown management
    document.addEventListener('click', handleGlobalClick);
    
    // Enhanced file management with vectorization
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);
    
    // Chat interface
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    document.getElementById('messageInput').addEventListener('keydown', handleMessageInput);
    
    // Parameter management
    setupParameterEventListeners();
    setupPresetEventListeners();
    
    // Configuration actions
    document.getElementById('saveParams').addEventListener('click', saveModelParameters);
    document.getElementById('resetParams').addEventListener('click', resetToDefaults);
    
    // Saved configurations management
    document.getElementById('newConfigBtn').addEventListener('click', showConfigSaveSection);
    document.getElementById('saveNewConfigBtn').addEventListener('click', saveNewConfiguration);
    document.getElementById('cancelSaveBtn').addEventListener('click', hideConfigSaveSection);
    
    // Configuration name input handling
    document.getElementById('configNameInput').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            saveNewConfiguration();
        } else if (event.key === 'Escape') {
            hideConfigSaveSection();
        }
    });
    
    // Enhanced clear history with vectorization reset
    document.getElementById('clearHistoryBtn').addEventListener('click', async function() {
        if (confirm('Clear all chat history? This action cannot be undone.')) {
            try {
                const response = await fetch('/api/chat/history', {
                    method: 'DELETE'
                });

                if (response.ok) {
                    document.getElementById('historyList').innerHTML = '<p>No chat history yet</p>';
                    document.getElementById('chatMessages').innerHTML = `
                        <div class="message assistant enhanced-message">
                            <div class="message-header">
                                <span>Assistant</span>
                                <div class="message-metadata">
                                    <span class="confidence-indicator" style="color: #10b981">Confidence: 9/10 (system-ready)</span>
                                    <span class="reasoning-pattern" style="color: var(--system-accent);">Vectorized-RAG</span>
                                    <span class="timestamp">Ready</span>
                                </div>
                            </div>
                            <div class="message-content">Chat history cleared. Vectorized RAG pipeline is ready for new conversations with enhanced semantic search!</div>
                        </div>
                    `;
                    
                    // Reset conversation state
                    if (typeof currentConversationId !== 'undefined') {
                        currentConversationId = 'default';
                        conversationState = 'new';
                        if (typeof updateConversationUI === 'function') {
                            updateConversationUI();
                        }
                    }
                } else {
                    alert('Failed to clear chat history');
                }
            } catch (error) {
                console.error('Error clearing history:', error);
                alert('Error occurred while clearing history');
            }
        }
    });
}

function startSystemMonitoring() {
    // Enhanced system monitoring every 10 seconds
    setInterval(checkEnhancedSystemStatus, 10000);
}

// Enhanced initialization
function initializeEnhancedFeatures() {
    // Setup processing mode toggle
    setupProcessingModeToggle();
    
    // Initialize vectorization monitoring
    updateVectorizationStats();
    
    // Enhanced system monitoring every 15 seconds
    setInterval(checkEnhancedSystemStatus, 15000);
    
    // Update vectorization stats every 30 seconds
    setInterval(updateVectorizationStats, 30000);
}

// Processing mode management
function setupProcessingModeToggle() {
    const fastBtn = document.getElementById('fastModeBtn');
    const detailedBtn = document.getElementById('detailedModeBtn');
    const indicator = document.getElementById('processingModeIndicator');
    const currentModeSpan = document.getElementById('currentMode');

    function setMode(mode) {
        currentProcessingMode = mode;
        
        // Update button styles
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.style.background = 'var(--bg-tertiary)';
            btn.style.color = 'var(--text-secondary)';
            btn.style.border = '1px solid var(--border-color)';
        });

        if (mode === 'fast') {
            if (fastBtn) {
                fastBtn.classList.add('active');
                fastBtn.style.background = 'var(--accent-blue)';
                fastBtn.style.color = 'white';
                fastBtn.style.border = 'none';
            }
            if (indicator) indicator.textContent = 'Fast Mode';
            if (currentModeSpan) currentModeSpan.textContent = 'Fast';
        } else {
            if (detailedBtn) {
                detailedBtn.classList.add('active');
                detailedBtn.style.background = 'var(--system-accent)';
                detailedBtn.style.color = 'white';
                detailedBtn.style.border = 'none';
            }
            if (indicator) indicator.textContent = 'Detailed Mode';
            if (currentModeSpan) currentModeSpan.textContent = 'Detailed';
        }
        
        // Save preference in memory (not localStorage due to restrictions)
        currentProcessingMode = mode;
    }

    // Set default mode
    setMode('fast');

    if (fastBtn) fastBtn.addEventListener('click', () => setMode('fast'));
    if (detailedBtn) detailedBtn.addEventListener('click', () => setMode('detailed'));
}

// Conversation management functions
function startNewConversation() {
    currentConversationId = 'conv_' + Date.now();
    conversationState = 'new';
    
    // Clear current chat display
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    // Add welcome message
    addMessageToInterface('assistant', 'Started new conversation. How can I help you?');
    
    // Update UI to show new conversation
    updateConversationUI();
    saveConversationState();
}

function updateConversationUI() {
    // Add conversation indicator to the chat header
    const chatHeader = document.querySelector('.chat-container h2');
    if (chatHeader) {
        const conversationInfo = document.createElement('span');
        conversationInfo.id = 'conversationInfo';
        conversationInfo.style.fontSize = '0.8rem';
        conversationInfo.style.color = 'var(--text-secondary)';
        conversationInfo.style.marginLeft = '10px';
        
        if (conversationState === 'new') {
            conversationInfo.textContent = '(New Conversation)';
        } else if (conversationState === 'active') {
            conversationInfo.textContent = '(Continuing Conversation)';
        }
        
        // Remove existing conversation info
        const existing = document.getElementById('conversationInfo');
        if (existing) existing.remove();
        
        chatHeader.appendChild(conversationInfo);
    }
}

function addNewConversationButton() {
    const chatContainer = document.querySelector('.chat-container');
    const chatHeader = chatContainer.querySelector('h2') || chatContainer.firstElementChild;
    
    if (!document.getElementById('newConversationBtn')) {
        const newConvButton = document.createElement('button');
        newConvButton.id = 'newConversationBtn';
        newConvButton.textContent = 'New Conversation';
        newConvButton.style.cssText = `
            background: var(--accent-blue);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 15px;
            font-size: 0.9rem;
            transition: background 0.3s;
        `;
        newConvButton.onmouseover = () => newConvButton.style.background = 'var(--accent-light-blue)';
        newConvButton.onmouseout = () => newConvButton.style.background = 'var(--accent-blue)';
        newConvButton.onclick = startNewConversation;
        
        chatHeader.style.display = 'flex';
        chatHeader.style.justifyContent = 'space-between';
        chatHeader.style.alignItems = 'center';
        chatHeader.appendChild(newConvButton);
    }
}

function saveConversationState() {
    const state = {
        conversationId: currentConversationId,
        timestamp: new Date().toISOString()
    };
    // Store in memory only due to localStorage restrictions
    window.currentConversationState = state;
}

function restoreConversationState() {
    // Restore from memory if available
    if (window.currentConversationState) {
        const state = window.currentConversationState;
        currentConversationId = state.conversationId;
        conversationState = 'active';
        updateConversationUI();
    }
}

// Model management functions
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name} (${formatFileSize(model.size)})`;
                modelSelect.appendChild(option);
            });
            selectedModel = data.models[0].name;
            await loadModelSpecificParameters(selectedModel);
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSelect.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        displaySystemError('Unable to load available models');
    }
}

async function handleModelChange(event) {
    selectedModel = event.target.value;
    if (selectedModel) {
        await loadModelSpecificParameters(selectedModel);
    }
}

async function loadModelSpecificParameters(modelName) {
    try {
        const response = await fetch(`/api/model-params?model=${encodeURIComponent(modelName)}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            modelParams = { ...data.params };
            initializeModelParams();
            showParameterLoadFeedback(data.timestamp);
        }
    } catch (error) {
        console.error('Error loading model parameters:', error);
    }
}

// Configuration interface management
function handleConfigToggle(event) {
    event.stopPropagation();
    toggleConfigurationDropdown();
}

function handleGlobalClick(event) {
    if (!event.target.closest('.config-button')) {
        closeConfigurationDropdown();
    }
}

function toggleConfigurationDropdown() {
    const configButton = document.querySelector('.config-button');
    const configDropdown = document.getElementById('configDropdown');
    
    configButton.classList.toggle('active');
    configDropdown.classList.toggle('show');
}

function closeConfigurationDropdown() {
    const configButton = document.querySelector('.config-button');
    const configDropdown = document.getElementById('configDropdown');
    
    configButton.classList.remove('active');
    configDropdown.classList.remove('show');
}

// Parameter management system
function setupParameterEventListeners() {
    const parameterElements = ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'num_predict'];
    
    parameterElements.forEach(paramName => {
        const element = document.getElementById(paramName);
        if (element) {
            element.addEventListener('input', (event) => {
                handleParameterChange(paramName, event.target.value);
            });
        }
    });
}

function handleParameterChange(paramName, value) {
    const numericValue = ['seed', 'num_predict', 'top_k'].includes(paramName) 
        ? parseInt(value) 
        : parseFloat(value);
    
    modelParams[paramName] = numericValue;
    updateParameterDisplays();
    clearActivePresetSelection();
    clearActiveConfigurationSelection();
}

function setupPresetEventListeners() {
    document.querySelectorAll('.preset-btn').forEach(button => {
        button.addEventListener('click', () => {
            const presetName = button.dataset.preset;
            applyParameterPreset(presetName);
            setActivePresetButton(button);
        });
    });
}

function applyParameterPreset(presetName) {
    if (presets[presetName]) {
        modelParams = { ...presets[presetName] };
        initializeModelParams();
        clearActiveConfigurationSelection();
    }
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

function initializeModelParams() {
    document.getElementById('temperature').value = modelParams.temperature;
    document.getElementById('top_p').value = modelParams.top_p;
    document.getElementById('top_k').value = modelParams.top_k;
    document.getElementById('repeat_penalty').value = modelParams.repeat_penalty;
    document.getElementById('seed').value = modelParams.seed;
    document.getElementById('num_predict').value = modelParams.num_predict;
    
    updateParameterDisplays();
}

function updateParameterDisplays() {
    document.getElementById('temperatureValue').textContent = modelParams.temperature.toFixed(2);
    document.getElementById('topPValue').textContent = modelParams.top_p.toFixed(2);
    document.getElementById('topKValue').textContent = modelParams.top_k;
    document.getElementById('repeatPenaltyValue').textContent = modelParams.repeat_penalty.toFixed(2);
    document.getElementById('seedValue').textContent = modelParams.seed === -1 ? 'Random' : modelParams.seed;
    document.getElementById('numPredictValue').textContent = modelParams.num_predict === -1 ? 'Auto' : modelParams.num_predict;
}

async function saveModelParameters() {
    try {
        const response = await fetch('/api/model-params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: selectedModel,
                params: modelParams
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showSaveParametersFeedback(true);
        } else {
            showSaveParametersFeedback(false, result.error);
        }
    } catch (error) {
        console.error('Error saving parameters:', error);
        showSaveParametersFeedback(false, 'Network error occurred');
    }
}

async function loadSavedParameters() {
    try {
        const response = await fetch('/api/presets');
        const data = await response.json();
        
        if (data.presets) {
            Object.assign(presets, data.presets);
        }
    } catch (error) {
        console.error('Error loading saved parameters:', error);
    }
}

function resetToDefaults() {
    modelParams = { ...presets.balanced };
    initializeModelParams();
    clearActivePresetSelection();
    document.querySelector('[data-preset="balanced"]').classList.add('active');
}

// File management system
async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        const files = await response.json();
        
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        if (files.error) {
            fileList.innerHTML = '<p>Unable to load context files</p>';
            return;
        }
        
        // Identify system files automatically
        systemFiles = files.filter(filename => 
            filename.toLowerCase().includes('admin') || 
            filename.toLowerCase().includes('system') ||
            filename.toLowerCase().includes('default') ||
            filename.toLowerCase().includes('config') ||
            filename.toLowerCase().includes('haag') ||
            filename.toLowerCase().includes('response-instructions')
        );
        
        // Ensure system files are selected
        systemFiles.forEach(filename => {
            if (!selectedFiles.includes(filename)) {
                selectedFiles.push(filename);
            }
        });
        
        // Render file list interface
        files.forEach(filename => {
            const fileItem = createFileListItem(filename);
            fileList.appendChild(fileItem);
        });
    } catch (error) {
        console.error('Error loading files:', error);
        displaySystemError('Failed to load context files');
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
    label.textContent = filename;
    if (isSystemFile) {
        label.innerHTML += ' <span class="system-badge">SYSTEM</span>';
    }
    
    fileItem.appendChild(checkbox);
    fileItem.appendChild(label);
    
    if (!isSystemFile) {
        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.onclick = () => deleteContextFile(filename);
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

// Enhanced file upload with vectorization feedback
async function handleFileUpload(event) {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    // Show vectorization progress
    const uploadLabel = document.querySelector('.file-input-label');
    const originalText = uploadLabel.textContent;
    
    uploadLabel.textContent = '🔄 Uploading & Vectorizing...';
    uploadLabel.style.background = 'var(--system-accent)';
    
    try {
        const response = await fetch('/api/files', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            await loadFiles();
            event.target.value = '';
            
            // Show vectorization success
            uploadLabel.textContent = `✅ ${uploadedFile.name} vectorized!`;
            uploadLabel.style.background = '#10b981';
            
            // Update vectorization stats
            await updateVectorizationStats();
            
            // Log vectorization details
            console.log('File vectorization completed:', {
                filename: result.filename,
                chunksCreated: result.chunksCreated,
                vectorizationStatus: result.vectorizationStatus
            });
            
        } else {
            uploadLabel.textContent = '❌ Upload failed';
            uploadLabel.style.background = '#ef4444';
        }
    } catch (error) {
        console.error('File upload error:', error);
        uploadLabel.textContent = '❌ Network error';
        uploadLabel.style.background = '#ef4444';
    }
    
    // Reset label after 3 seconds
    setTimeout(() => {
        uploadLabel.textContent = originalText;
        uploadLabel.style.background = '';
    }, 3000);
}

async function deleteContextFile(filename) {
    if (systemFiles.includes(filename)) {
        alert('System files cannot be deleted');
        return;
    }
    
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
            // Update vectorization stats after deletion
            await updateVectorizationStats();
        } else {
            alert('Failed to delete file');
        }
    } catch (error) {
        console.error('File deletion error:', error);
        alert('Error occurred during file deletion');
    }
}

// Enhanced chat interface management
function handleMessageInput(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
        sendMessage();
    }
}

// Enhanced sendMessage function with vectorization support
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const messageText = messageInput.value.trim();
    
    if (!messageText) return;
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    addMessageToInterface('user', messageText);
    messageInput.value = '';
    
    // Show enhanced deliberation indicator
    displayEnhancedDeliberationIndicator();
    
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
                fast_mode: currentProcessingMode === 'fast'  // Use current processing mode
            })
        });
        
        const responseData = await response.json();
        
        removeDeliberationIndicator();
        
        if (responseData.error) {
            addMessageToInterface('assistant', `Error: ${responseData.error}`);
        } else {
            // Add enhanced message with vectorization data
            addVectorizedMessageToInterface('assistant', responseData);
            await loadChatHistory();
            
            // Update conversation state
            conversationState = 'active';
            updateConversationUI();
            saveConversationState();
            
            // Update vectorization stats
            updateVectorizationStatsFromResponse(responseData);
            
            // Log vectorization results for debugging
            console.log('Vectorization Results:', {
                processing_mode: responseData.metadata?.processing_mode,
                context_chunks_used: responseData.metadata?.context_chunks_used,
                confidence_score: responseData.metadata?.confidence_score,
                search_results: responseData.metadata?.search_results?.length || 0
            });
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeDeliberationIndicator();
        addMessageToInterface('assistant', 'Error: Failed to communicate with the AI service');
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
        <span>${new Date().toLocaleTimeString()}</span>
    `;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    chatMessages.appendChild(messageElement);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Enhanced message interface with vectorization details
function addVectorizedMessageToInterface(role, responseData) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role} enhanced-message`;
    
    // Enhanced message header with vectorization metadata
    const messageHeader = document.createElement('div');
    messageHeader.className = 'message-header';
    
    const confidence = responseData.metadata?.confidence_score || responseData.confidence_score || 7;
    const processingMode = responseData.metadata?.processing_mode || 'standard';
    const chunksUsed = responseData.metadata?.context_chunks_used || 0;
    const confidenceColor = confidence >= 8 ? '#10b981' : confidence >= 6 ? '#f59e0b' : '#ef4444';
    
    messageHeader.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <div class="message-metadata">
            <span class="confidence-indicator" style="color: ${confidenceColor}">
                Confidence: ${confidence}/10 (${processingMode})
            </span>
            <span class="reasoning-pattern" style="color: var(--system-accent); font-size: 0.8rem;">
                🔍 ${chunksUsed} chunks analyzed
            </span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        </div>
    `;
    
    // Main response content
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = responseData.response || 'Response generated';
    
    // Vectorization insights section
    const vectorizationSection = createVectorizationInsightsSection(responseData);
    
    // Enhanced deliberation section
    const deliberationSection = createEnhancedDeliberationSection(responseData);
    
    // Citations section with vectorization info
    const citationsSection = createEnhancedCitationsSection(responseData.citations || []);
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    messageElement.appendChild(vectorizationSection);
    messageElement.appendChild(deliberationSection);
    messageElement.appendChild(citationsSection);
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// New vectorization insights section
function createVectorizationInsightsSection(responseData) {
    const section = document.createElement('div');
    section.className = 'vectorization-insights';
    section.style.cssText = `
        margin-top: 12px;
        padding: 12px;
        background: linear-gradient(135deg, var(--bg-primary) 0%, rgba(139, 92, 246, 0.05) 100%);
        border-radius: 6px;
        border: 1px solid var(--system-accent);
    `;
    
    const searchResults = responseData.metadata?.search_results || responseData.search_results || [];
    const avgSimilarity = searchResults.length > 0 
        ? (searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length).toFixed(3)
        : '0.000';
    
    section.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: var(--system-accent); font-weight: 600; font-size: 0.9rem;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z"/>
            </svg>
            Semantic Search Results
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; font-size: 0.8rem;">
            <div>
                <span style="color: var(--text-secondary);">Chunks Found:</span>
                <span style="color: var(--text-primary); font-weight: 600;">${searchResults.length}</span>
            </div>
            <div>
                <span style="color: var(--text-secondary);">Avg Similarity:</span>
                <span style="color: var(--accent-light-blue); font-weight: 600;">${avgSimilarity}</span>
            </div>
            <div>
                <span style="color: var(--text-secondary);">Mode:</span>
                <span style="color: var(--success); font-weight: 600;">${responseData.metadata?.processing_mode || 'standard'}</span>
            </div>
            <div>
                <span style="color: var(--text-secondary);">Sources:</span>
                <span style="color: var(--text-primary); font-weight: 600;">${(responseData.citations || []).length}</span>
            </div>
        </div>
    `;
    
    return section;
}

// Enhanced message interface with multi-step reasoning
function createEnhancedDeliberationSection(responseData) {
    const deliberationSection = document.createElement('div');
    deliberationSection.className = 'deliberation-section enhanced';
    
    const deliberationHeader = document.createElement('div');
    deliberationHeader.className = 'deliberation-header';
    deliberationHeader.innerHTML = `
        <span class="deliberation-title">
            <svg class="deliberation-icon" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            Multi-Step Reasoning Process
        </span>
        <button class="deliberation-toggle" onclick="toggleDeliberation(this)">
            <svg class="chevron-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
        </button>
    `;
    
    const deliberationContent = document.createElement('div');
    deliberationContent.className = 'deliberation-content collapsed';
    
    const reasoningChain = getNestedValue(responseData, 'metadata.reasoning_chain') || [];
    const confidenceBreakdown = getNestedValue(responseData, 'metadata.confidence_breakdown') || {};
    const validationFlags = getNestedValue(responseData, 'metadata.validation_flags') || [];
    
    deliberationContent.innerHTML = `
        <div class="reasoning-overview">
            <div class="deliberation-item">
                <strong>Reasoning Pattern:</strong> ${getNestedValue(responseData, 'metadata.reasoning_pattern') || 'Vectorized-RAG'}
            </div>
            <div class="deliberation-item">
                <strong>Problem Components:</strong> ${formatComponents(reasoningChain[0])}
            </div>
            <div class="deliberation-item">
                <strong>Evidence Quality:</strong> ${formatEvidenceQuality(reasoningChain[1])}
            </div>
        </div>
        
        <div class="reasoning-chain">
            <h4 style="color: var(--accent-light-blue); margin: 15px 0 10px 0;">Reasoning Chain</h4>
            ${createReasoningSteps(reasoningChain)}
        </div>
        
        <div class="confidence-breakdown">
            <h4 style="color: var(--accent-light-blue); margin: 15px 0 10px 0;">Confidence Analysis</h4>
            ${createConfidenceBreakdown(confidenceBreakdown)}
        </div>
        
        ${validationFlags.length > 0 ? `
        <div class="validation-flags">
            <h4 style="color: var(--warning); margin: 15px 0 10px 0;">Validation Notes</h4>
            ${validationFlags.map(flag => `<div class="flag-item">⚠️ ${flag}</div>`).join('')}
        </div>
        ` : ''}
    `;
    
    deliberationSection.appendChild(deliberationHeader);
    deliberationSection.appendChild(deliberationContent);
    
    return deliberationSection;
}

// Enhanced citations section with similarity scores
function createEnhancedCitationsSection(citations) {
    if (!citations || citations.length === 0) return document.createElement('div');
    
    const citationsSection = document.createElement('div');
    citationsSection.className = 'citations-section';
    citationsSection.style.cssText = `
        margin-top: 12px;
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 6px;
        border: 1px solid var(--border-color);
    `;
    
    const citationsHeader = document.createElement('div');
    citationsHeader.className = 'citations-header';
    citationsHeader.innerHTML = `
        <svg class="citation-icon" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 0v12h8V4H6z" clip-rule="evenodd"/>
        </svg>
        <span style="color: var(--accent-light-blue); font-weight: 600; font-size: 0.85rem; margin-left: 8px;">
            Sources Referenced (${citations.length})
        </span>
    `;
    
    const citationsList = document.createElement('div');
    citationsList.className = 'citations-list';
    citationsList.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;';
    
    citations.forEach((citation, index) => {
        const citationItem = document.createElement('div');
        citationItem.className = `citation-item ${citation.type.toLowerCase()}-citation`;
        citationItem.style.cssText = `
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            font-size: 0.8rem;
            transition: all 0.3s;
            cursor: pointer;
        `;
        
        citationItem.innerHTML = `
            <span class="citation-badge" style="
                background: ${citation.type === 'SYSTEM' ? 'var(--system-accent)' : 'var(--accent-blue)'};
                color: white;
                padding: 2px 6px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 0.7rem;
            ">${citation.type}</span>
            <span class="citation-file" style="color: var(--text-primary); font-weight: 500;">${citation.file}</span>
            ${citation.relevance ? `<span style="color: var(--text-secondary); font-size: 0.7rem;">(${citation.relevance})</span>` : ''}
        `;
        
        citationItem.addEventListener('mouseenter', function() {
            this.style.background = 'var(--bg-primary)';
            this.style.borderColor = citation.type === 'SYSTEM' ? 'var(--system-accent)' : 'var(--accent-blue)';
            this.style.transform = 'translateY(-1px)';
            this.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
        });
        
        citationItem.addEventListener('mouseleave', function() {
            this.style.background = 'var(--bg-secondary)';
            this.style.borderColor = 'var(--border-color)';
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
        
        citationsList.appendChild(citationItem);
    });
    
    citationsSection.appendChild(citationsHeader);
    citationsSection.appendChild(citationsList);
    
    return citationsSection;
}

function toggleDeliberation(button) {
    const content = button.closest('.deliberation-section').querySelector('.deliberation-content');
    const icon = button.querySelector('.chevron-icon');
    
    content.classList.toggle('collapsed');
    icon.style.transform = content.classList.contains('collapsed') ? '' : 'rotate(180deg)';
}

// Enhanced deliberation indicator with vectorization stages
function displayEnhancedDeliberationIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    
    const deliberationElement = document.createElement('div');
    deliberationElement.className = 'deliberation-indicator enhanced';
    deliberationElement.id = 'deliberation-indicator';
    
    deliberationElement.innerHTML = `
        <div class="deliberation-stages" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stage active" style="
                padding: 12px; text-align: center; background: linear-gradient(135deg, var(--system-accent) 0%, rgba(139, 92, 246, 0.8) 100%);
                color: white; border-radius: 8px; animation: pulse 2s infinite;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Vectorizing query...</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Semantic search...</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">🧠</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Reasoning...</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">💡</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Analysis...</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">✅</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Verification...</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">📝</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Synthesis...</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(deliberationElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Simulate stage progression with realistic timings
    const stages = deliberationElement.querySelectorAll('.stage');
    let currentStage = 0;
    
    const stageTimings = [1000, 800, 1200, 1000, 800, 600]; // Different timings for each stage
    
    function activateNextStage() {
        if (currentStage < stages.length - 1) {
            // Deactivate current stage
            stages[currentStage].style.opacity = '0.7';
            stages[currentStage].style.background = 'var(--bg-secondary)';
            stages[currentStage].style.animation = 'none';
            
            currentStage++;
            
            // Activate next stage
            const nextStage = stages[currentStage];
            nextStage.style.opacity = '1';
            nextStage.style.background = 'linear-gradient(135deg, var(--system-accent) 0%, rgba(139, 92, 246, 0.8) 100%)';
            nextStage.style.color = 'white';
            nextStage.style.animation = 'pulse 2s infinite';
            nextStage.style.boxShadow = '0 4px 12px rgba(139, 92, 246, 0.3)';
            
            setTimeout(activateNextStage, stageTimings[currentStage] || 1000);
        }
    }
    
    setTimeout(activateNextStage, stageTimings[0]);
}

function removeDeliberationIndicator() {
    const deliberationElement = document.getElementById('deliberation-indicator');
    if (deliberationElement) {
        deliberationElement.remove();
    }
}

// Update vectorization stats from response
function updateVectorizationStatsFromResponse(responseData) {
    const searchResults = responseData.metadata?.search_results || responseData.search_results || [];
    
    if (searchResults.length > 0) {
        const avgSimilarity = searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length;
        vectorizationStats.avgSimilarity = avgSimilarity.toFixed(3);
        
        // Update UI elements
        const avgSimilarityEl = document.getElementById('avgSimilarity');
        const chunksPerQueryEl = document.getElementById('chunksPerQuery');
        
        if (avgSimilarityEl) avgSimilarityEl.textContent = vectorizationStats.avgSimilarity;
        if (chunksPerQueryEl) chunksPerQueryEl.textContent = searchResults.length;
    }
    
    // Update search time (simulated for now)
    vectorizationStats.searchTime = `${Math.round(Math.random() * 200 + 50)}ms`;
    const searchTimeEl = document.getElementById('searchTime');
    if (searchTimeEl) searchTimeEl.textContent = vectorizationStats.searchTime;
}

// Enhanced vectorization status updates
async function updateVectorizationStats() {
    try {
        // Get MCP server stats
        const mcpResponse = await fetch('/api/mcp/status');
        if (mcpResponse.ok) {
            const mcpData = await mcpResponse.json();
            vectorizationStats.totalVectors = mcpData.vectorization?.totalVectors || 0;
        }

        // Get vector statistics if available
        try {
            const vectorStatsResponse = await fetch('/api/vectors/stats');
            if (vectorStatsResponse.ok) {
                const vectorData = await vectorStatsResponse.json();
                vectorizationStats.totalVectors = vectorData.totalVectors || vectorizationStats.totalVectors;
                vectorizationStats.indexedFiles = vectorData.fileCount || 0;
            }
        } catch (e) {
            // Endpoint might not exist yet
        }

        // Get file list to count indexed files
        const filesResponse = await fetch('/api/files');
        if (filesResponse.ok) {
            const filesData = await filesResponse.json();
            if (Array.isArray(filesData)) {
                vectorizationStats.indexedFiles = filesData.length;
            } else if (filesData.length) {
                vectorizationStats.indexedFiles = filesData.length;
            }
        }

        // Update UI elements
        const totalVectorsEl = document.getElementById('totalVectors');
        const indexedFilesEl = document.getElementById('indexedFiles');
        
        if (totalVectorsEl) totalVectorsEl.textContent = vectorizationStats.totalVectors;
        if (indexedFilesEl) indexedFilesEl.textContent = vectorizationStats.indexedFiles;

    } catch (error) {
        console.error('Error updating vectorization stats:', error);
        // Set fallback values
        const totalVectorsEl = document.getElementById('totalVectors');
        const indexedFilesEl = document.getElementById('indexedFiles');
        
        if (totalVectorsEl) totalVectorsEl.textContent = 'Error';
        if (indexedFilesEl) indexedFilesEl.textContent = 'Error';
    }
}

// Enhanced chat history management
async function loadChatHistory() {
    try {
        const response = await fetch(`/api/chat/history?conversation_id=${currentConversationId}`);
        const historyData = await response.json();
        
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';
        
        historyData.reverse().forEach((historyItem, index) => {
            const historyElement = createEnhancedHistoryListItem(historyItem, historyData.length - 1 - index);
            historyList.appendChild(historyElement);
        });
        
        chatHistory = historyData;
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Enhanced history item creation with vectorization metadata
function createEnhancedHistoryListItem(historyItem, index) {
    const historyElement = document.createElement('div');
    historyElement.className = 'history-item enhanced';
    historyElement.onclick = () => displayHistoryItem(index);
    
    const timestampElement = document.createElement('div');
    timestampElement.className = 'history-timestamp';
    timestampElement.textContent = new Date(historyItem.timestamp).toLocaleString();
    
    const previewElement = document.createElement('div');
    previewElement.className = 'history-preview';
    previewElement.textContent = historyItem.message.substring(0, 50) + '...';
    
    // Enhanced metadata with vectorization info
    const metadataElement = document.createElement('div');
    metadataElement.className = 'history-metadata';
    
    const confidence = historyItem.metadata?.confidence_score || historyItem.confidence_score || 7;
    const chunksUsed = historyItem.metadata?.context_chunks_used || 0;
    const processingMode = historyItem.metadata?.processing_mode || 'standard';
    
    const confidenceColor = confidence >= 8 ? '#10b981' : confidence >= 6 ? '#f59e0b' : '#ef4444';
    
    metadataElement.innerHTML = `
        <div class="history-confidence" style="display: flex; align-items: center; gap: 6px; font-size: 0.75rem;">
            <span class="confidence-dot" style="width: 8px; height: 8px; border-radius: 50%; background: ${confidenceColor}"></span>
            <span class="confidence-value" style="color: var(--text-secondary); font-weight: 500;">${confidence}/10</span>
        </div>
        <div class="history-vectorization" style="font-size: 0.7rem; color: var(--system-accent); display: flex; align-items: center; gap: 4px;">
            <span>🔍</span>
            <span>${chunksUsed} chunks (${processingMode})</span>
        </div>
    `;
    
    historyElement.appendChild(timestampElement);
    historyElement.appendChild(previewElement);
    historyElement.appendChild(metadataElement);
    
    return historyElement;
}

function displayHistoryItem(index) {
    const historyItem = chatHistory[index];
    if (!historyItem) return;
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    addMessageToInterface('user', historyItem.message);
    
    // Display with deliberation data if available
    if (historyItem.deliberation_summary || historyItem.metadata) {
        addVectorizedMessageToInterface('assistant', historyItem);
    } else {
        addMessageToInterface('assistant', historyItem.response);
    }
}

// Enhanced system status monitoring
async function checkEnhancedSystemStatus() {
    await Promise.all([
        checkOllamaStatus(),
        checkMCPServerStatus(),
        checkEmbeddingServiceStatus()
    ]);
    
    // Update vectorization stats
    await updateVectorizationStats();
}

async function checkOllamaStatus() {
    try {
        const response = await fetch('/api/models');
        updateStatusIndicator('ollama', response.ok);
    } catch (error) {
        updateStatusIndicator('ollama', false);
    }
}

async function checkMCPServerStatus() {
    try {
        const response = await fetch('/api/mcp/status');
        updateStatusIndicator('mcp', response.ok);
    } catch (error) {
        updateStatusIndicator('mcp', false);
    }
}

async function checkEmbeddingServiceStatus() {
    try {
        const response = await fetch('/api/embedding/health');
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

// Saved Configuration Management System
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
        const configItem = createConfigurationListItem(config);
        configsList.appendChild(configItem);
    });
}

function createConfigurationListItem(config) {
    const configItem = document.createElement('div');
    configItem.className = 'saved-config-item';
    configItem.dataset.configId = config.id;
    
    if (currentActiveConfig && currentActiveConfig.id === config.id) {
        configItem.classList.add('active');
    }
    
    const configInfo = document.createElement('div');
    configInfo.className = 'config-item-info';
    configInfo.onclick = () => applySavedConfiguration(config);
    
    const configName = document.createElement('div');
    configName.className = 'config-item-name';
    configName.textContent = config.name;
    
    const configDetails = document.createElement('div');
    configDetails.className = 'config-item-details';
    configDetails.textContent = `T:${config.params.temperature} • P:${config.params.top_p} • K:${config.params.top_k}`;
    
    configInfo.appendChild(configName);
    configInfo.appendChild(configDetails);
    
    const configActions = document.createElement('div');
    configActions.className = 'config-item-actions';
    
    const deleteButton = document.createElement('button');
    deleteButton.className = 'config-delete-btn';
    deleteButton.textContent = '×';
    deleteButton.onclick = (event) => {
        event.stopPropagation();
        deleteSavedConfiguration(config.id);
    };
    
    configActions.appendChild(deleteButton);
    
    configItem.appendChild(configInfo);
    configItem.appendChild(configActions);
    
    return configItem;
}

function showConfigSaveSection() {
    const saveSection = document.getElementById('configSaveSection');
    const nameInput = document.getElementById('configNameInput');
    
    saveSection.style.display = 'block';
    nameInput.focus();
    nameInput.value = '';
}

function hideConfigSaveSection() {
    const saveSection = document.getElementById('configSaveSection');
    saveSection.style.display = 'none';
}

async function saveNewConfiguration() {
    const nameInput = document.getElementById('configNameInput');
    const configName = nameInput.value.trim();
    
    if (!configName) {
        nameInput.focus();
        return;
    }
    
    if (configName.length > 50) {
        alert('Configuration name must be 50 characters or less');
        return;
    }
    
    // Check for duplicate names
    if (savedConfigurations.some(config => config.name.toLowerCase() === configName.toLowerCase())) {
        alert('A configuration with this name already exists');
        return;
    }
    
    try {
        const response = await fetch('/api/saved-configs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
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
            showConfigurationSaveFeedback(true, configName);
        } else {
            showConfigurationSaveFeedback(false, result.error);
        }
    } catch (error) {
        console.error('Error saving configuration:', error);
        showConfigurationSaveFeedback(false, 'Network error occurred');
    }
}

async function applySavedConfiguration(config) {
    modelParams = { ...config.params };
    currentActiveConfig = config;
    
    initializeModelParams();
    clearActivePresetSelection();
    updateActiveConfigurationDisplay();
    
    showConfigurationApplyFeedback(config.name);
}

async function deleteSavedConfiguration(configId) {
    const config = savedConfigurations.find(c => c.id === configId);
    if (!config) return;
    
    if (!confirm(`Delete configuration "${config.name}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/saved-configs/${configId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            if (currentActiveConfig && currentActiveConfig.id === configId) {
                currentActiveConfig = null;
            }
            await loadSavedConfigurations();
            showConfigurationDeleteFeedback(config.name);
        } else {
            alert('Failed to delete configuration');
        }
    } catch (error) {
        console.error('Error deleting configuration:', error);
        alert('Error occurred during configuration deletion');
    }
}

function updateActiveConfigurationDisplay() {
    document.querySelectorAll('.saved-config-item').forEach(item => {
        item.classList.remove('active');
    });
    
    if (currentActiveConfig) {
        const activeItem = document.querySelector(`[data-config-id="${currentActiveConfig.id}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
    }
}

function clearActiveConfigurationSelection() {
    currentActiveConfig = null;
    updateActiveConfigurationDisplay();
}

// Helper functions for deliberation data processing
function getNestedValue(obj, path, defaultValue = null) {
    if (!obj || !path) return defaultValue;
    
    try {
        return path.split('.').reduce((current, key) => {
            return current && current[key] !== undefined ? current[key] : defaultValue;
        }, obj);
    } catch (error) {
        return defaultValue;
    }
}

function formatComponents(decomposition) {
    if (!decomposition) return 'Single component analysis';
    
    // Handle different data types
    if (typeof decomposition === 'string') return decomposition;
    if (Array.isArray(decomposition)) return decomposition.join(', ');
    if (decomposition.components) {
        return decomposition.components.map(comp => {
            return typeof comp === 'object' ? (comp.aspect || comp.description || JSON.stringify(comp)) : comp;
        }).join(', ');
    }
    
    return 'Processing completed';
}

function formatEvidenceQuality(evidence) {
    if (!evidence) return 'Moderate';
    
    // Handle different data types
    if (typeof evidence === 'string') return evidence;
    if (evidence.overall_evidence_quality) {
        return evidence.overall_evidence_quality.charAt(0).toUpperCase() + evidence.overall_evidence_quality.slice(1);
    }
    
    return 'Moderate';
}

function createReasoningSteps(reasoningChain) {
    if (!Array.isArray(reasoningChain) || reasoningChain.length === 0) {
        return '<div class="reasoning-step"><div class="step-content">No detailed reasoning chain available</div></div>';
    }
    
    const stepNames = [
        'Problem Decomposition',
        'Evidence Gathering', 
        'Pattern Identification',
        'Hypothesis Formation',
        'Verification',
        'Synthesis'
    ];
    
    return reasoningChain.map((step, index) => `
        <div class="reasoning-step">
            <div class="step-header">
                <span class="step-number">${index + 1}</span>
                <span class="step-name">${stepNames[index] || `Step ${index + 1}`}</span>
            </div>
            <div class="step-content">
                ${formatStepContent(step, index)}
            </div>
        </div>
    `).join('');
}

function formatStepContent(step, stepIndex) {
    if (!step) return 'Processing completed';
    
    // Handle string responses from LLM
    if (typeof step === 'string') {
        return `<div>Status: <span class="highlight">Completed</span></div><div>Details: ${step.substring(0, 100)}</div>`;
    }
    
    // Handle non-object responses
    if (typeof step !== 'object') {
        return `<div>Result: <span class="highlight">${step}</span></div>`;
    }
    
    try {
        switch(stepIndex) {
            case 0: // Decomposition
                return `
                    <div>Query Type: <span class="highlight">${step.query_type || 'analytical'}</span></div>
                    <div>Complexity: <span class="highlight">${step.complexity || 'moderate'}</span></div>
                    ${step.components ? `<div>Components: ${Array.isArray(step.components) ? step.components.length : 'N/A'}</div>` : ''}
                `;
            case 1: // Evidence
                return `
                    <div>Sources Analyzed: <span class="highlight">${step.ranked_sources?.length || 0}</span></div>
                    <div>Evidence Quality: <span class="highlight">${step.overall_evidence_quality || 'moderate'}</span></div>
                    ${step.coverage_assessment ? `<div>Coverage: ${formatCoverage(step.coverage_assessment)}</div>` : ''}
                `;
            case 2: // Pattern
                return `
                    <div>Pattern: <span class="highlight">${step.pattern_type || 'analytical'}</span></div>
                    <div>Steps: <span class="highlight">${step.reasoning_steps?.length || 0}</span></div>
                `;
            case 3: // Hypothesis
                return `
                    <div>Approaches: <span class="highlight">${step.candidate_approaches?.length || 1}</span></div>
                    ${step.primary_approach ? `<div>Strategy: ${typeof step.primary_approach === 'object' ? step.primary_approach.strategy || 'Standard' : step.primary_approach}</div>` : ''}
                `;
            case 4: // Verification
                return `
                    <div>Logic Score: <span class="highlight">${step.logical_consistency?.score || step.overall_confidence || 'N/A'}</span></div>
                    <div>Completeness: <span class="highlight">${step.completeness_assessment?.information_sufficiency || 'N/A'}</span></div>
                `;
            case 5: // Synthesis
                return `
                    <div>Strategy: <span class="highlight">${step.strategy || 'comprehensive'}</span></div>
                    <div>Citations: <span class="highlight">${step.citation_targets?.length || 0}</span></div>
                `;
            default:
                // Fallback for any step format
                if (typeof step === 'object') {
                    const keys = Object.keys(step).slice(0, 3);
                    return keys.map(key => `<div>${key}: <span class="highlight">${step[key]}</span></div>`).join('');
                }
                return `<div>Result: <span class="highlight">${JSON.stringify(step).substring(0, 100)}...</span></div>`;
        }
    } catch (error) {
        console.error('Error formatting step content:', error);
        return `<div>Processing: <span class="highlight">Step ${stepIndex + 1} completed</span></div>`;
    }
}

function formatCoverage(coverage) {
    if (!coverage || typeof coverage !== 'object') return 'Analysis completed';
    
    const well = coverage.well_covered?.length || 0;
    const partial = coverage.partially_covered?.length || 0;
    const gaps = coverage.gaps?.length || 0;
    return `${well} complete, ${partial} partial, ${gaps} gaps`;
}

function createConfidenceBreakdown(breakdown) {
    if (!breakdown || Object.keys(breakdown).length === 0) {
        return '<div class="confidence-item">No detailed breakdown available</div>';
    }
    
    return Object.entries(breakdown).map(([factor, score]) => `
        <div class="confidence-item">
            <div class="confidence-factor">
                <span class="factor-name">${formatFactorName(factor)}</span>
                <div class="confidence-bar-small">
                    <div class="confidence-fill-small" style="width: ${score * 10}%"></div>
                </div>
                <span class="factor-score">${score}/10</span>
            </div>
        </div>
    `).join('');
}

function formatFactorName(factor) {
    return factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// User feedback systems
function showSaveParametersFeedback(success, errorMessage = null) {
    const saveButton = document.getElementById('saveParams');
    const originalText = saveButton.textContent;
    
    if (success) {
        saveButton.textContent = 'Saved!';
        saveButton.style.background = '#059669';
        
        setTimeout(() => {
            saveButton.textContent = originalText;
            saveButton.style.background = '';
        }, 2000);
    } else {
        saveButton.textContent = 'Error';
        saveButton.style.background = '#ef4444';
        console.error('Parameter save error:', errorMessage);
        
        setTimeout(() => {
            saveButton.textContent = originalText;
            saveButton.style.background = '';
        }, 2000);
    }
}

function showParameterLoadFeedback(timestamp) {
    if (timestamp) {
        const loadTime = new Date(timestamp).toLocaleString();
        console.log(`Parameters loaded from: ${loadTime}`);
    }
}

function showUploadFeedback(success, message) {
    const uploadLabel = document.querySelector('.file-input-label');
    const originalText = uploadLabel.textContent;
    
    if (success) {
        uploadLabel.textContent = `✓ ${message} uploaded`;
        uploadLabel.style.background = '#10b981';
    } else {
        uploadLabel.textContent = `✗ ${message}`;
        uploadLabel.style.background = '#ef4444';
    }
    
    setTimeout(() => {
        uploadLabel.textContent = originalText;
        uploadLabel.style.background = '';
    }, 3000);
}

function showConfigurationSaveFeedback(success, message) {
    const saveButton = document.getElementById('saveNewConfigBtn');
    const originalText = saveButton.textContent;
    
    if (success) {
        saveButton.textContent = 'Saved!';
        saveButton.style.background = '#059669';
    } else {
        saveButton.textContent = 'Error';
        saveButton.style.background = '#ef4444';
        console.error('Configuration save error:', message);
    }
    
    setTimeout(() => {
        saveButton.textContent = originalText;
        saveButton.style.background = '';
    }, 2000);
}

function showConfigurationApplyFeedback(configName) {
    const configTitle = document.querySelector('.config-title');
    const originalText = configTitle.textContent;
    
    configTitle.textContent = `✓ ${configName}`;
    configTitle.style.color = '#10b981';
    
    setTimeout(() => {
        configTitle.textContent = originalText;
        configTitle.style.color = '';
    }, 2000);
}

function showConfigurationDeleteFeedback(configName) {
    console.log(`Deleted configuration: ${configName}`);
}

function displaySystemError(message) {
    console.error('System error:', message);
}

// Utility functions
function formatFileSize(bytes) {
    const sizeUnits = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    
    const unitIndex = Math.floor(Math.log(bytes) / Math.log(1024));
    const size = Math.round(bytes / Math.pow(1024, unitIndex) * 100) / 100;
    
    return `${size} ${sizeUnits[unitIndex]}`;
}
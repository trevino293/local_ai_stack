// Global state management
let selectedModel = 'llama2';
let selectedFiles = [];
let systemFiles = [];
let chatHistory = [];
let savedConfigurations = [];
let currentActiveConfig = null;
let currentConversationId = 'default';
let conversationState = 'active'; // 'active', 'new', 'loading'

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
        await checkSystemStatus();
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
    
    // File management
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
}

function startSystemMonitoring() {
    // Monitor system status every 10 seconds
    setInterval(checkSystemStatus, 10000);
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
    try {
        sessionStorage.setItem('conversationState', JSON.stringify(state));
    } catch (error) {
        console.log('Could not save conversation state');
    }
}

function restoreConversationState() {
    try {
        const saved = sessionStorage.getItem('conversationState');
        if (saved) {
            const state = JSON.parse(saved);
            currentConversationId = state.conversationId;
            conversationState = 'active';
            updateConversationUI();
        }
    } catch (error) {
        console.log('No previous conversation state found');
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

async function handleFileUpload(event) {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    try {
        const response = await fetch('/api/files', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            await loadFiles();
            event.target.value = '';
            showUploadFeedback(true, uploadedFile.name);
        } else {
            showUploadFeedback(false, 'Upload failed');
        }
    } catch (error) {
        console.error('File upload error:', error);
        showUploadFeedback(false, 'Network error during upload');
    }
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

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const messageText = messageInput.value.trim();
    
    if (!messageText) return;
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    addMessageToInterface('user', messageText);
    messageInput.value = '';
    
    // Show deliberation indicator
    displayDeliberationIndicator();
    
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
                conversation_id: currentConversationId
            })
        });
        
        const responseData = await response.json();
        
        removeDeliberationIndicator();
        
        if (responseData.error) {
            addMessageToInterface('assistant', `Error: ${responseData.error}`);
        } else {
            // Add enhanced message with deliberation data
            addEnhancedMessageToInterface('assistant', responseData);
            await loadChatHistory();
            
            // Update conversation state
            conversationState = 'active';
            updateConversationUI();
            saveConversationState();
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

// Enhanced message interface with multi-step reasoning
function addEnhancedMessageToInterface(role, responseData) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role} enhanced-message`;
    
    // Enhanced message header with detailed confidence
    const messageHeader = document.createElement('div');
    messageHeader.className = 'message-header';
    
    const confidence = getNestedValue(responseData, 'deliberation_summary.confidence') || 7;
    const reasoningQuality = getNestedValue(responseData, 'metadata.reasoning_quality') || 'adequate';
    const confidenceColor = confidence >= 8 ? '#10b981' : confidence >= 6 ? '#f59e0b' : '#ef4444';
    
    messageHeader.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <div class="message-metadata">
            <span class="confidence-indicator" style="color: ${confidenceColor}">
                Confidence: ${confidence}/10 (${reasoningQuality})
            </span>
            <span class="reasoning-pattern" style="color: var(--system-accent); font-size: 0.8rem;">
                ${getNestedValue(responseData, 'metadata.reasoning_pattern') || 'Standard'}
            </span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        </div>
    `;
    
    // Main response content
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = responseData.response || 'Response generated';
    
    // Enhanced deliberation section with reasoning chain
    const deliberationSection = createEnhancedDeliberationSection(responseData);
    
    // Citations section
    const citationsSection = createCitationsSection(responseData.citations || []);
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    messageElement.appendChild(deliberationSection);
    messageElement.appendChild(citationsSection);
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Safe property access helper
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

// New enhanced deliberation section with error handling
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
                <strong>Reasoning Pattern:</strong> ${getNestedValue(responseData, 'metadata.reasoning_pattern') || 'Standard'}
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

// Fixed helper functions for formatting reasoning data
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

function createCitationsSection(citations) {
    if (!citations || citations.length === 0) return document.createElement('div');
    
    const citationsSection = document.createElement('div');
    citationsSection.className = 'citations-section';
    
    const citationsHeader = document.createElement('div');
    citationsHeader.className = 'citations-header';
    citationsHeader.innerHTML = `
        <svg class="citation-icon" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 0v12h8V4H6z" clip-rule="evenodd"/>
        </svg>
        Sources Referenced
    `;
    
    const citationsList = document.createElement('div');
    citationsList.className = 'citations-list';
    
    citations.forEach(citation => {
        const citationItem = document.createElement('div');
        citationItem.className = `citation-item ${citation.type.toLowerCase()}-citation`;
        citationItem.innerHTML = `
            <span class="citation-badge">${citation.type}</span>
            <span class="citation-file">${citation.file}</span>
        `;
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

// Enhanced deliberation indicator with more stages
function displayDeliberationIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    
    const deliberationElement = document.createElement('div');
    deliberationElement.className = 'deliberation-indicator enhanced';
    deliberationElement.id = 'deliberation-indicator';
    
    deliberationElement.innerHTML = `
        <div class="deliberation-stages">
            <div class="stage active">
                <div class="stage-icon">🔍</div>
                <div class="stage-text">Decomposing problem...</div>
            </div>
            <div class="stage">
                <div class="stage-icon">📊</div>
                <div class="stage-text">Gathering evidence...</div>
            </div>
            <div class="stage">
                <div class="stage-icon">🧠</div>
                <div class="stage-text">Identifying patterns...</div>
            </div>
            <div class="stage">
                <div class="stage-icon">💡</div>
                <div class="stage-text">Forming hypotheses...</div>
            </div>
            <div class="stage">
                <div class="stage-icon">✅</div>
                <div class="stage-text">Verifying reasoning...</div>
            </div>
            <div class="stage">
                <div class="stage-icon">📝</div>
                <div class="stage-text">Synthesizing response...</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(deliberationElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Simulate stage progression
    let currentStage = 0;
    const stages = deliberationElement.querySelectorAll('.stage');
    const stageInterval = setInterval(() => {
        if (currentStage < stages.length - 1) {
            stages[currentStage].classList.remove('active');
            currentStage++;
            stages[currentStage].classList.add('active');
        } else {
            clearInterval(stageInterval);
        }
    }, 1500);
}

function removeDeliberationIndicator() {
    const deliberationElement = document.getElementById('deliberation-indicator');
    if (deliberationElement) {
        deliberationElement.remove();
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
            const historyElement = createHistoryListItem(historyItem, historyData.length - 1 - index);
            historyList.appendChild(historyElement);
        });
        
        chatHistory = historyData;
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Enhanced history item creation
function createHistoryListItem(historyItem, index) {
    const historyElement = document.createElement('div');
    historyElement.className = 'history-item enhanced';
    historyElement.onclick = () => displayHistoryItem(index);
    
    const timestampElement = document.createElement('div');
    timestampElement.className = 'history-timestamp';
    timestampElement.textContent = new Date(historyItem.timestamp).toLocaleString();
    
    const previewElement = document.createElement('div');
    previewElement.className = 'history-preview';
    previewElement.textContent = historyItem.message.substring(0, 50) + '...';
    
    // Enhanced confidence and reasoning indicators
    const metadataElement = document.createElement('div');
    metadataElement.className = 'history-metadata';
    
    const confidence = getNestedValue(historyItem, 'deliberation_summary.confidence');
    if (confidence) {
        const reasoningPattern = getNestedValue(historyItem, 'metadata.reasoning_pattern') || 'standard';
        const confidenceColor = confidence >= 8 ? '#10b981' : confidence >= 6 ? '#f59e0b' : '#ef4444';
        
        metadataElement.innerHTML = `
            <div class="history-confidence">
                <span class="confidence-dot" style="background: ${confidenceColor}"></span>
                <span class="confidence-value">${confidence}/10</span>
            </div>
            <div class="history-pattern" style="font-size: 0.7rem; color: var(--system-accent);">
                ${reasoningPattern}
            </div>
        `;
    }
    
    historyElement.appendChild(timestampElement);
    historyElement.appendChild(previewElement);
    if (metadataElement.innerHTML) {
        historyElement.appendChild(metadataElement);
    }
    
    return historyElement;
}

function displayHistoryItem(index) {
    const historyItem = chatHistory[index];
    if (!historyItem) return;
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    addMessageToInterface('user', historyItem.message);
    
    // Display with deliberation data if available
    if (historyItem.deliberation_summary) {
        addEnhancedMessageToInterface('assistant', historyItem);
    } else {
        addMessageToInterface('assistant', historyItem.response);
    }
}

// System status monitoring
async function checkSystemStatus() {
    await Promise.all([
        checkOllamaStatus(),
        checkMCPServerStatus()
    ]);
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

function updateStatusIndicator(service, isOnline) {
    const statusText = document.getElementById(`${service}Status`);
    const statusDot = document.getElementById(`${service}Dot`);
    
    if (isOnline) {
        statusText.textContent = 'Online';
        statusDot.classList.add('online');
    } else {
        statusText.textContent = 'Offline';
        statusDot.classList.remove('online');
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
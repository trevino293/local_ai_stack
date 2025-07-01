// Complete Enhanced main.js with vector database support
// Global state management with vector database integration
let selectedModel = 'llama2';
let selectedFiles = [];
let systemFiles = [];
let chatHistory = [];
let savedConfigurations = [];
let currentActiveConfig = null;
let currentConversationId = 'default';
let conversationState = 'active';

// Enhanced vectorization globals with database stats
let currentProcessingMode = 'fast';
let vectorDatabaseStats = {
    totalVectors: 0,
    indexedFiles: 0,
    avgSimilarity: 0,
    searchTime: 0,
    databaseType: 'unknown',
    embeddingMethod: 'unknown',
    collectionName: 'unknown',
    status: 'unknown'
};

// Vector database performance metrics
let performanceMetrics = {
    lastSearchTime: 0,
    averageSearchTime: 0,
    searchCount: 0,
    lastVectorizationTime: 0,
    vectorizationCount: 0,
    cacheHitRate: 0
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

// Enhanced parameter presets optimized for vector database usage
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
    },
    analytical: {
        temperature: 0.4,
        top_p: 0.8,
        top_k: 30,
        repeat_penalty: 1.15,
        seed: -1,
        num_predict: -1
    }
};

// Application initialization with vector database support
document.addEventListener('DOMContentLoaded', () => {
    initializeApplication();
    setupEventListeners();
    startEnhancedSystemMonitoring();
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
        await initializeVectorDatabaseFeatures();
        initializeEnhancedFeatures();
        
        // Show initialization success
        displaySystemNotification('Vector database system initialized successfully', 'success');
    } catch (error) {
        console.error('Application initialization error:', error);
        displaySystemError('Failed to initialize vector database components');
    }
}

async function initializeVectorDatabaseFeatures() {
    try {
        // Load vector database statistics
        await updateVectorDatabaseStats();
        
        // Initialize performance monitoring
        initializePerformanceMonitoring();
        
        // Setup vector database UI elements
        setupVectorDatabaseUI();
        
        // Check if migration from in-memory system is needed
        await checkMigrationStatus();
        
    } catch (error) {
        console.error('Vector database initialization error:', error);
        displaySystemNotification('Vector database features unavailable', 'warning');
    }
}

function setupEventListeners() {
    // Model selection and configuration
    document.getElementById('modelSelect').addEventListener('change', handleModelChange);
    document.getElementById('configToggle').addEventListener('click', handleConfigToggle);
    
    // Configuration dropdown management
    document.addEventListener('click', handleGlobalClick);
    
    // Enhanced file management with vector database
    document.getElementById('fileInput').addEventListener('change', handleEnhancedFileUpload);
    
    // Chat interface
    document.getElementById('sendButton').addEventListener('click', sendEnhancedMessage);
    document.getElementById('messageInput').addEventListener('keydown', handleMessageInput);
    
    // Parameter management
    setupParameterEventListeners();
    setupPresetEventListeners();
    
    // Configuration actions
    document.getElementById('saveParams').addEventListener('click', saveModelParameters);
    document.getElementById('resetParams').addEventListener('click', resetToDefaults);
    
    // Enhanced vector database actions
    setupVectorDatabaseEventListeners();
    
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
    
    // Enhanced clear history with vector database cleanup
    setupEnhancedClearHistory();
}

function setupVectorDatabaseEventListeners() {
    // Batch vectorization button
    const batchVectorizeBtn = document.getElementById('batchVectorizeBtn');
    if (batchVectorizeBtn) {
        batchVectorizeBtn.addEventListener('click', handleBatchVectorization);
    }
    
    // Vector database refresh button
    const refreshVectorStatsBtn = document.getElementById('refreshVectorStatsBtn');
    if (refreshVectorStatsBtn) {
        refreshVectorStatsBtn.addEventListener('click', updateVectorDatabaseStats);
    }
    
    // Search similarity threshold slider
    const similarityThresholdSlider = document.getElementById('similarityThreshold');
    if (similarityThresholdSlider) {
        similarityThresholdSlider.addEventListener('input', handleSimilarityThresholdChange);
    }
    
    // Advanced search options
    const advancedSearchToggle = document.getElementById('advancedSearchToggle');
    if (advancedSearchToggle) {
        advancedSearchToggle.addEventListener('click', toggleAdvancedSearchOptions);
    }
}

function handleSimilarityThresholdChange(event) {
    const value = parseFloat(event.target.value).toFixed(2);
    const similarityValue = document.getElementById('similarityThresholdValue');
    const currentSimilarityDisplay = document.getElementById('currentSimilarityDisplay');
    
    if (similarityValue) similarityValue.textContent = value;
    if (currentSimilarityDisplay) currentSimilarityDisplay.textContent = value;
}

function startEnhancedSystemMonitoring() {
    // Enhanced system monitoring every 15 seconds
    setInterval(checkEnhancedSystemStatus, 15000);
    
    // Vector database stats every 30 seconds
    setInterval(updateVectorDatabaseStats, 30000);
    
    // Performance metrics every 60 seconds
    setInterval(updatePerformanceMetrics, 60000);
}

// Enhanced file upload with vector database integration
async function handleEnhancedFileUpload(event) {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    // Show enhanced upload progress
    const uploadLabel = document.querySelector('.file-input-label');
    const originalText = uploadLabel.textContent;
    
    // Create progress indicator
    const progressContainer = createUploadProgressIndicator();
    uploadLabel.parentNode.appendChild(progressContainer);
    
    uploadLabel.textContent = '🔄 Uploading & Vectorizing...';
    uploadLabel.style.background = 'var(--system-accent)';
    
    try {
        const startTime = Date.now();
        
        const response = await fetch('/api/files', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        const processingTime = Date.now() - startTime;
        
        if (response.ok) {
            await loadFiles();
            event.target.value = '';
            
            // Show vectorization success with details
            uploadLabel.textContent = `✅ ${uploadedFile.name} vectorized!`;
            uploadLabel.style.background = '#10b981';
            
            // Update performance metrics
            performanceMetrics.lastVectorizationTime = processingTime;
            performanceMetrics.vectorizationCount++;
            
            // Update vector database stats
            await updateVectorDatabaseStats();
            
            // Show detailed success notification
            displayVectorizationSuccessNotification(result, processingTime);
            
            // Log vectorization details
            console.log('Enhanced file vectorization completed:', {
                filename: result.filename,
                chunksCreated: result.chunksCreated,
                vectorizationStatus: result.vectorizationStatus,
                processingMethod: result.processingMethod,
                processingTime: processingTime
            });
            
        } else {
            uploadLabel.textContent = '❌ Upload failed';
            uploadLabel.style.background = '#ef4444';
            
            displaySystemNotification(`Upload failed: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Enhanced file upload error:', error);
        uploadLabel.textContent = '❌ Network error';
        uploadLabel.style.background = '#ef4444';
        displaySystemNotification('Network error during file upload', 'error');
    } finally {
        // Remove progress indicator
        if (progressContainer) {
            progressContainer.remove();
        }
        
        // Reset label after 3 seconds
        setTimeout(() => {
            uploadLabel.textContent = originalText;
            uploadLabel.style.background = '';
        }, 3000);
    }
}

// Enhanced chat message sending with vector database optimization
async function sendEnhancedMessage() {
    const messageInput = document.getElementById('messageInput');
    const messageText = messageInput.value.trim();
    
    if (!messageText) return;
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    
    addMessageToInterface('assistant', 'Started new conversation. How can I help you?');
    
    updateConversationUI();
    saveConversationState();
}

function updateConversationUI() {
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
    window.currentConversationState = state;
}

function restoreConversationState() {
    if (window.currentConversationState) {
        const state = window.currentConversationState;
        currentConversationId = state.conversationId;
        conversationState = 'active';
        updateConversationUI();
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

// Enhanced deliberation section creation
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
            Enhanced Reasoning Process
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
                <strong>Processing Mode:</strong> ${getNestedValue(responseData, 'metadata.processing_mode') || 'Enhanced-RAG'}
            </div>
            <div class="deliberation-item">
                <strong>Vector Database:</strong> ${vectorDatabaseStats.databaseType} (${vectorDatabaseStats.embeddingMethod})
            </div>
            <div class="deliberation-item">
                <strong>Context Quality:</strong> ${formatContextQuality(responseData)}
            </div>
        </div>
        
        <div class="reasoning-chain">
            <h4 style="color: var(--accent-light-blue); margin: 15px 0 10px 0;">Reasoning Analysis</h4>
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

// Enhanced citations section with vector database info
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
            Vector Sources Referenced (${citations.length})
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

function formatContextQuality(responseData) {
    const searchResults = responseData.metadata?.search_results || [];
    if (searchResults.length === 0) return 'No context found';
    
    const avgSimilarity = searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length;
    
    if (avgSimilarity > 0.8) return 'Excellent match';
    if (avgSimilarity > 0.6) return 'Good match';
    if (avgSimilarity > 0.4) return 'Fair match';
    return 'Limited match';
}

function createReasoningSteps(reasoningChain) {
    if (!Array.isArray(reasoningChain) || reasoningChain.length === 0) {
        return '<div class="reasoning-step"><div class="step-content">Enhanced vector-based reasoning completed</div></div>';
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
    
    if (typeof step === 'string') {
        return `<div>Status: <span class="highlight">Completed</span></div><div>Details: ${step.substring(0, 100)}</div>`;
    }
    
    if (typeof step !== 'object') {
        return `<div>Result: <span class="highlight">${step}</span></div>`;
    }
    
    try {
        switch(stepIndex) {
            case 0:
                return `
                    <div>Query Type: <span class="highlight">${step.query_type || 'analytical'}</span></div>
                    <div>Complexity: <span class="highlight">${step.complexity || 'moderate'}</span></div>
                    ${step.components ? `<div>Components: ${Array.isArray(step.components) ? step.components.length : 'N/A'}</div>` : ''}
                `;
            case 1:
                return `
                    <div>Sources Analyzed: <span class="highlight">${step.ranked_sources?.length || 0}</span></div>
                    <div>Evidence Quality: <span class="highlight">${step.overall_evidence_quality || 'moderate'}</span></div>
                    ${step.coverage_assessment ? `<div>Coverage: ${formatCoverage(step.coverage_assessment)}</div>` : ''}
                `;
            case 2:
                return `
                    <div>Pattern: <span class="highlight">${step.pattern_type || 'analytical'}</span></div>
                    <div>Steps: <span class="highlight">${step.reasoning_steps?.length || 0}</span></div>
                `;
            case 3:
                return `
                    <div>Approaches: <span class="highlight">${step.candidate_approaches?.length || 1}</span></div>
                    ${step.primary_approach ? `<div>Strategy: ${typeof step.primary_approach === 'object' ? step.primary_approach.strategy || 'Standard' : step.primary_approach}</div>` : ''}
                `;
            case 4:
                return `
                    <div>Logic Score: <span class="highlight">${step.logical_consistency?.score || step.overall_confidence || 'N/A'}</span></div>
                    <div>Completeness: <span class="highlight">${step.completeness_assessment?.information_sufficiency || 'N/A'}</span></div>
                `;
            case 5:
                return `
                    <div>Strategy: <span class="highlight">${step.strategy || 'comprehensive'}</span></div>
                    <div>Citations: <span class="highlight">${step.citation_targets?.length || 0}</span></div>
                `;
            default:
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
        return '<div class="confidence-item">Vector database confidence analysis completed</div>';
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

// Advanced search and filtering functions
function toggleAdvancedSearchOptions() {
    const advancedOptions = document.getElementById('advancedSearchOptions');
    if (advancedOptions) {
        advancedOptions.style.display = advancedOptions.style.display === 'none' ? 'block' : 'none';
    }
}

// Global function to handle deliberation toggle (for HTML onclick)
function toggleDeliberation(button) {
    const content = button.closest('.deliberation-section').querySelector('.deliberation-content');
    const icon = button.querySelector('.chevron-icon');

    content.classList.toggle('collapsed');
    icon.style.transform = content.classList.contains('collapsed') ? '' : 'rotate(180deg)';
}

// Add CSS animations for enhanced features
if (!document.getElementById('enhanced-vector-animations')) {
    const style = document.createElement('style');
    style.id = 'enhanced-vector-animations';
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        @keyframes progressBar {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .vector-enhanced {
            position: relative;
        }
        
        .vector-enhanced::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--system-accent));
            border-radius: 4px 4px 0 0;
        }
        
        .status-indicator.connected {
            color: var(--success);
        }
        
        .status-indicator.disconnected {
            color: var(--error);
        }
        
        .status-indicator.unknown {
            color: var(--warning);
        }
        
        .vectorization-insights.enhanced {
            animation: fadeInUp 0.3s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(style);
}

// Initialize everything when DOM is ready
console.log('Enhanced Vector Database Main.js loaded successfully!');Interface('user', messageText);
    messageInput.value = '';
    
    // Show enhanced deliberation indicator with vector database stages
    displayEnhancedVectorDeliberationIndicator();
    
    try {
        const startTime = Date.now();
        
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
                fast_mode: currentProcessingMode === 'fast',
                vector_db_options: {
                    similarity_threshold: getCurrentSimilarityThreshold(),
                    max_chunks: currentProcessingMode === 'fast' ? 3 : 5,
                    enable_filtering: getAdvancedSearchEnabled()
                }
            })
        });
        
        const responseData = await response.json();
        const searchTime = Date.now() - startTime;
        
        removeDeliberationIndicator();
        
        if (responseData.error) {
            addMessageToInterface('assistant', `Error: ${responseData.error}`);
        } else {
            // Add enhanced message with vector database data
            addEnhancedVectorizedMessageToInterface('assistant', responseData);
            await loadChatHistory();
            
            // Update conversation state
            conversationState = 'active';
            updateConversationUI();
            saveConversationState();
            
            // Update performance metrics
            updateSearchPerformanceMetrics(searchTime, responseData);
            
            // Update vector database stats
            await updateVectorDatabaseStats();
            
            // Log enhanced vectorization results
            console.log('Enhanced Vector RAG Results:', {
                processing_mode: responseData.metadata?.processing_mode,
                context_chunks_used: responseData.metadata?.context_chunks_used,
                confidence_score: responseData.metadata?.confidence_score,
                search_results: responseData.metadata?.search_results?.length || 0,
                vector_database: vectorDatabaseStats.databaseType,
                search_time: searchTime,
                embedding_method: vectorDatabaseStats.embeddingMethod
            });
        }
    } catch (error) {
        console.error('Enhanced chat error:', error);
        removeDeliberationIndicator();
        addMessageToInterface('assistant', 'Error: Failed to communicate with the enhanced AI service');
    } finally {
        sendButton.disabled = false;
    }
}

// Enhanced vector database statistics updates
async function updateVectorDatabaseStats() {
    try {
        // Get vector database stats from embedding service
        const embeddingStatsResponse = await fetch('/api/embedding/stats');
        if (embeddingStatsResponse.ok) {
            const embeddingData = await embeddingStatsResponse.json();
            vectorDatabaseStats.databaseType = embeddingData.vector_database || 'unknown';
            vectorDatabaseStats.embeddingMethod = embeddingData.embedding_method || 'unknown';
            vectorDatabaseStats.totalVectors = embeddingData.total_vectors || 0;
            vectorDatabaseStats.collectionName = embeddingData.collection_name || 'unknown';
            vectorDatabaseStats.status = embeddingData.database_status || 'unknown';
        }

        // Get MCP server stats
        const mcpResponse = await fetch('/api/mcp/status');
        if (mcpResponse.ok) {
            const mcpData = await mcpResponse.json();
            if (mcpData.vectorization) {
                vectorDatabaseStats.totalVectors = mcpData.vectorization.totalVectors || vectorDatabaseStats.totalVectors;
            }
        }

        // Get vector statistics from MCP server
        try {
            const vectorStatsResponse = await fetch('/api/vectors/stats');
            if (vectorStatsResponse.ok) {
                const vectorData = await vectorStatsResponse.json();
                vectorDatabaseStats.totalVectors = vectorData.totalVectors || vectorDatabaseStats.totalVectors;
                vectorDatabaseStats.indexedFiles = vectorData.fileCount || 0;
            }
        } catch (e) {
            // Vector stats endpoint might not be available
        }

        // Get file list to count indexed files
        const filesResponse = await fetch('/api/files');
        if (filesResponse.ok) {
            const filesData = await filesResponse.json();
            if (Array.isArray(filesData)) {
                vectorDatabaseStats.indexedFiles = filesData.filter(f => f.vectorized !== false).length;
            }
        }

        // Update UI elements with enhanced information
        updateVectorDatabaseUI();

    } catch (error) {
        console.error('Error updating vector database stats:', error);
        displaySystemNotification('Failed to update vector database statistics', 'warning');
    }
}

function updateVectorDatabaseUI() {
    // Update basic stats
    const totalVectorsEl = document.getElementById('totalVectors');
    const indexedFilesEl = document.getElementById('indexedFiles');
    const databaseTypeEl = document.getElementById('databaseType');
    const embeddingMethodEl = document.getElementById('embeddingMethod');
    const collectionNameEl = document.getElementById('collectionName');
    const databaseStatusEl = document.getElementById('databaseStatus');
    const activeDatabaseType = document.getElementById('activeDatabaseType');
    
    if (totalVectorsEl) totalVectorsEl.textContent = vectorDatabaseStats.totalVectors.toLocaleString();
    if (indexedFilesEl) indexedFilesEl.textContent = vectorDatabaseStats.indexedFiles;
    if (databaseTypeEl) databaseTypeEl.textContent = vectorDatabaseStats.databaseType;
    if (embeddingMethodEl) embeddingMethodEl.textContent = vectorDatabaseStats.embeddingMethod;
    if (collectionNameEl) collectionNameEl.textContent = vectorDatabaseStats.collectionName;
    if (activeDatabaseType) activeDatabaseType.textContent = vectorDatabaseStats.databaseType;
    if (databaseStatusEl) {
        databaseStatusEl.textContent = vectorDatabaseStats.status;
        databaseStatusEl.className = `status-indicator ${vectorDatabaseStats.status}`;
    }
    
    // Update performance metrics
    updatePerformanceMetricsUI();
}

function updatePerformanceMetricsUI() {
    const avgSearchTimeEl = document.getElementById('avgSearchTime');
    const searchCountEl = document.getElementById('searchCount');
    const cacheHitRateEl = document.getElementById('cacheHitRate');
    const lastVectorizationTimeEl = document.getElementById('lastVectorizationTime');
    const realTimeSearchTimeEl = document.getElementById('realTimeSearchTime');
    
    if (avgSearchTimeEl) avgSearchTimeEl.textContent = `${performanceMetrics.averageSearchTime}ms`;
    if (searchCountEl) searchCountEl.textContent = performanceMetrics.searchCount;
    if (cacheHitRateEl) cacheHitRateEl.textContent = `${performanceMetrics.cacheHitRate}%`;
    if (lastVectorizationTimeEl) lastVectorizationTimeEl.textContent = `${performanceMetrics.lastVectorizationTime}ms`;
    if (realTimeSearchTimeEl) realTimeSearchTimeEl.textContent = `~${performanceMetrics.averageSearchTime || 125}ms`;
}

function updateSearchPerformanceMetrics(searchTime, responseData) {
    performanceMetrics.lastSearchTime = searchTime;
    performanceMetrics.searchCount++;
    
    // Calculate rolling average
    if (performanceMetrics.averageSearchTime === 0) {
        performanceMetrics.averageSearchTime = searchTime;
    } else {
        performanceMetrics.averageSearchTime = Math.round(
            (performanceMetrics.averageSearchTime * 0.8) + (searchTime * 0.2)
        );
    }
    
    // Update similarity stats from search results
    const searchResults = responseData.metadata?.search_results || [];
    if (searchResults.length > 0) {
        const avgSimilarity = searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length;
        vectorDatabaseStats.avgSimilarity = avgSimilarity.toFixed(3);
        
        const avgSimilarityEl = document.getElementById('avgSimilarity');
        if (avgSimilarityEl) avgSimilarityEl.textContent = vectorDatabaseStats.avgSimilarity;
    }
}

// Enhanced deliberation indicator with vector database stages
function displayEnhancedVectorDeliberationIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    
    const deliberationElement = document.createElement('div');
    deliberationElement.className = 'deliberation-indicator enhanced vector-enhanced';
    deliberationElement.id = 'deliberation-indicator';
    
    deliberationElement.innerHTML = `
        <div class="deliberation-stages" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
            <div class="stage active vector-stage" style="
                padding: 12px; text-align: center; 
                background: linear-gradient(135deg, var(--system-accent) 0%, rgba(139, 92, 246, 0.8) 100%);
                color: white; border-radius: 8px; animation: pulse 2s infinite;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Vector Search...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">${vectorDatabaseStats.databaseType}</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Similarity Ranking...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">Threshold: ${getCurrentSimilarityThreshold()}</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">🧠</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">RAG Generation...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">${currentProcessingMode} mode</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">💡</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Context Analysis...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">Multi-step reasoning</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">✅</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Verification...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">Confidence scoring</div>
            </div>
            <div class="stage" style="
                padding: 12px; text-align: center; background: var(--bg-tertiary); color: var(--text-secondary);
                border-radius: 8px; opacity: 0.5; transition: all 0.5s ease;
            ">
                <div class="stage-icon" style="font-size: 1.8rem; margin-bottom: 8px;">📝</div>
                <div class="stage-text" style="font-size: 0.8rem; font-weight: 500;">Synthesis...</div>
                <div class="stage-subtext" style="font-size: 0.7rem; opacity: 0.8;">Citation tracking</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(deliberationElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Enhanced stage progression with vector database timings
    const stages = deliberationElement.querySelectorAll('.stage');
    let currentStage = 0;
    
    const stageTimings = [1200, 800, 1500, 1000, 800, 600]; // Optimized for vector DB
    
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

// Enhanced message interface with vector database metadata
function addEnhancedVectorizedMessageToInterface(role, responseData) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role} enhanced-message vector-enhanced`;
    
    // Enhanced message header with vector database metadata
    const messageHeader = document.createElement('div');
    messageHeader.className = 'message-header';
    
    const confidence = responseData.metadata?.confidence_score || responseData.confidence_score || 7;
    const processingMode = responseData.metadata?.processing_mode || 'standard';
    const chunksUsed = responseData.metadata?.context_chunks_used || 0;
    const searchResults = responseData.metadata?.search_results || [];
    const avgSimilarity = searchResults.length > 0 
        ? (searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length).toFixed(3)
        : '0.000';
    
    const confidenceColor = confidence >= 8 ? '#10b981' : confidence >= 6 ? '#f59e0b' : '#ef4444';
    
    messageHeader.innerHTML = `
        <span>${role === 'user' ? 'You' : 'Assistant'}</span>
        <div class="message-metadata">
            <span class="confidence-indicator" style="color: ${confidenceColor}">
                Confidence: ${confidence}/10 (${processingMode})
            </span>
            <span class="vector-database-indicator" style="color: var(--system-accent); font-size: 0.8rem;">
                📊 ${vectorDatabaseStats.databaseType} • ${avgSimilarity} similarity
            </span>
            <span class="reasoning-pattern" style="color: var(--system-accent); font-size: 0.8rem;">
                🔍 ${chunksUsed} chunks • ${vectorDatabaseStats.embeddingMethod}
            </span>
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
        </div>
    `;
    
    // Main response content
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = responseData.response || 'Response generated';
    
    // Enhanced vectorization insights section
    const vectorizationSection = createEnhancedVectorizationInsightsSection(responseData);
    
    // Enhanced deliberation section
    const deliberationSection = createEnhancedDeliberationSection(responseData);
    
    // Enhanced citations section with vector database info
    const citationsSection = createEnhancedCitationsSection(responseData.citations || []);
    
    messageElement.appendChild(messageHeader);
    messageElement.appendChild(messageContent);
    messageElement.appendChild(vectorizationSection);
    messageElement.appendChild(deliberationSection);
    messageElement.appendChild(citationsSection);
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Enhanced vectorization insights with database details
function createEnhancedVectorizationInsightsSection(responseData) {
    const section = document.createElement('div');
    section.className = 'vectorization-insights enhanced';
    section.style.cssText = `
        margin-top: 12px;
        padding: 15px;
        background: linear-gradient(135deg, var(--bg-primary) 0%, rgba(139, 92, 246, 0.08) 100%);
        border-radius: 8px;
        border: 1px solid var(--system-accent);
    `;
    
    const searchResults = responseData.metadata?.search_results || responseData.search_results || [];
    const avgSimilarity = searchResults.length > 0 
        ? (searchResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / searchResults.length).toFixed(3)
        : '0.000';
    
    const highestSimilarity = searchResults.length > 0
        ? Math.max(...searchResults.map(r => r.similarity || 0)).toFixed(3)
        : '0.000';
    
    section.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; color: var(--system-accent); font-weight: 600; font-size: 0.95rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z"/>
            </svg>
            Vector Database Search Results
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; font-size: 0.85rem;">
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Database:</span>
                <span style="color: var(--accent-light-blue); font-weight: 600;">${vectorDatabaseStats.databaseType}</span>
            </div>
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Chunks Found:</span>
                <span style="color: var(--text-primary); font-weight: 600;">${searchResults.length}</span>
            </div>
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Avg Similarity:</span>
                <span style="color: var(--accent-light-blue); font-weight: 600;">${avgSimilarity}</span>
            </div>
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Best Match:</span>
                <span style="color: var(--success); font-weight: 600;">${highestSimilarity}</span>
            </div>
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Mode:</span>
                <span style="color: var(--success); font-weight: 600;">${responseData.metadata?.processing_mode || 'standard'}</span>
            </div>
            <div style="background: var(--bg-secondary); padding: 10px; border-radius: 6px;">
                <span style="color: var(--text-secondary); display: block;">Sources:</span>
                <span style="color: var(--text-primary); font-weight: 600;">${(responseData.citations || []).length}</span>
            </div>
        </div>
        
        ${searchResults.length > 0 ? `
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">
            <div style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 8px;">Top Search Results:</div>
            <div style="display: flex; flex-direction: column; gap: 6px;">
                ${searchResults.slice(0, 3).map((result, i) => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: var(--bg-secondary); border-radius: 4px; font-size: 0.8rem;">
                        <span style="color: var(--text-primary); font-weight: 500;">${result.filename || 'Unknown'}</span>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="color: var(--text-secondary);">Chunk ${result.chunkIndex || i}</span>
                            <span style="color: var(--accent-light-blue); font-weight: 600;">${(result.similarity || 0).toFixed(3)}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}
    `;
    
    return section;
}

// Batch vectorization handler
async function handleBatchVectorization() {
    const batchBtn = document.getElementById('batchVectorizeBtn');
    if (!batchBtn) return;
    
    const originalText = batchBtn.textContent;
    batchBtn.disabled = true;
    batchBtn.textContent = '🔄 Processing...';
    
    try {
        const response = await fetch('/api/vectorize/batch', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            batchBtn.textContent = '✅ Completed';
            batchBtn.style.background = '#10b981';
            
            // Update stats and display results
            await updateVectorDatabaseStats();
            displayBatchVectorizationResults(result);
            
        } else {
            batchBtn.textContent = '❌ Failed';
            batchBtn.style.background = '#ef4444';
            displaySystemNotification(`Batch vectorization failed: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Batch vectorization error:', error);
        batchBtn.textContent = '❌ Error';
        batchBtn.style.background = '#ef4444';
        displaySystemNotification('Network error during batch vectorization', 'error');
    } finally {
        setTimeout(() => {
            batchBtn.textContent = originalText;
            batchBtn.style.background = '';
            batchBtn.disabled = false;
        }, 3000);
    }
}

// Utility functions for enhanced vector database features
function getCurrentSimilarityThreshold() {
    const slider = document.getElementById('similarityThreshold');
    return slider ? parseFloat(slider.value) : 0.3;
}

function getAdvancedSearchEnabled() {
    const toggle = document.getElementById('advancedSearchEnabled');
    return toggle ? toggle.checked : false;
}

function createUploadProgressIndicator() {
    const progressContainer = document.createElement('div');
    progressContainer.className = 'upload-progress-indicator';
    progressContainer.style.cssText = `
        margin-top: 8px;
        padding: 8px;
        background: var(--bg-primary);
        border-radius: 4px;
        border: 1px solid var(--border-color);
    `;
    
    progressContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.8rem; color: var(--text-secondary);">Vectorizing with ${vectorDatabaseStats.databaseType}</span>
            <span style="font-size: 0.8rem; color: var(--accent-light-blue);">${vectorDatabaseStats.embeddingMethod}</span>
        </div>
        <div style="width: 100%; height: 4px; background: var(--bg-tertiary); border-radius: 2px; overflow: hidden;">
            <div style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--system-accent)); border-radius: 2px; animation: progressBar 3s ease-in-out infinite;"></div>
        </div>
    `;
    
    return progressContainer;
}

function displayVectorizationSuccessNotification(result, processingTime) {
    const notification = document.createElement('div');
    notification.className = 'vectorization-success-notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
    `;
    
    notification.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 5px;">✅ Vectorization Complete</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            ${result.filename} • ${result.chunksCreated} chunks • ${processingTime}ms
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

function displayBatchVectorizationResults(result) {
    console.log('Batch vectorization results:', result);
    displaySystemNotification(
        `Batch completed: ${result.summary?.successfulFiles || 0} files, ${result.summary?.totalChunks || 0} chunks`, 
        'success'
    );
}

function displaySystemNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `system-notification ${type}`;
    
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 12px 16px;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
        font-size: 0.9rem;
        max-width: 300px;
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Enhanced system status monitoring with vector database
async function checkEnhancedSystemStatus() {
    await Promise.all([
        checkOllamaStatus(),
        checkMCPServerStatus(),
        checkEmbeddingServiceStatus(),
        checkVectorDatabaseStatus()
    ]);
    
    // Update vector database stats
    await updateVectorDatabaseStats();
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

async function checkVectorDatabaseStatus() {
    try {
        const response = await fetch('/api/vectors/stats');
        updateStatusIndicator('vectordb', response.ok);
    } catch (error) {
        updateStatusIndicator('vectordb', false);
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

// Initialize enhanced features
function initializeEnhancedFeatures() {
    setupProcessingModeToggle();
    updateVectorDatabaseStats();
    setInterval(checkEnhancedSystemStatus, 15000);
    setInterval(updateVectorDatabaseStats, 30000);
    setInterval(updatePerformanceMetrics, 60000);
}

function initializePerformanceMonitoring() {
    // Initialize performance metrics tracking
    performanceMetrics = {
        lastSearchTime: 0,
        averageSearchTime: 0,
        searchCount: 0,
        lastVectorizationTime: 0,
        vectorizationCount: 0,
        cacheHitRate: 0
    };
}

function setupVectorDatabaseUI() {
    // Initialize vector database UI elements
    updateVectorDatabaseUI();
}

async function checkMigrationStatus() {
    // Check if migration from in-memory system is needed
    console.log('Checking vector database migration status...');
}

function updatePerformanceMetrics() {
    // Update performance metrics periodically
    updatePerformanceMetricsUI();
}

function setupEnhancedClearHistory() {
    document.getElementById('clearHistoryBtn').addEventListener('click', async function() {
        if (confirm('Clear all chat history? This action cannot be undone.')) {
            try {
                const response = await fetch('/api/chat/history', {
                    method: 'DELETE'
                });

                if (response.ok) {
                    document.getElementById('historyList').innerHTML = '<p>No chat history yet</p>';
                    document.getElementById('chatMessages').innerHTML = `
                        <div class="message assistant enhanced-message vector-enhanced">
                            <div class="message-header">
                                <span>Assistant</span>
                                <div class="message-metadata">
                                    <span class="confidence-indicator" style="color: #10b981">Confidence: 9/10 (system-ready)</span>
                                    <span class="vector-database-indicator" style="color: var(--system-accent);">📊 ${vectorDatabaseStats.databaseType}</span>
                                    <span class="timestamp">Ready</span>
                                </div>
                            </div>
                            <div class="message-content">Chat history cleared. Enhanced vector database pipeline is ready for new conversations with ${vectorDatabaseStats.databaseType} semantic search!</div>
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

// ALL REMAINING FUNCTIONS FROM ORIGINAL MAIN.JS
// Copy all remaining functions that weren't redefined above

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
        
        currentProcessingMode = mode;
    }

    // Set default mode
    setMode('fast');

    if (fastBtn) fastBtn.addEventListener('click', () => setMode('fast'));
    if (detailedBtn) detailedBtn.addEventListener('click', () => setMode('detailed'));
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
            await updateVectorDatabaseStats();
        } else {
            alert('Failed to delete file');
        }
    } catch (error) {
        console.error('File deletion error:', error);
        alert('Error occurred during file deletion');
    }
}

// Chat interface management
function handleMessageInput(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
        sendEnhancedMessage();
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
            const historyElement = createEnhancedHistoryListItem(historyItem, historyData.length - 1 - index);
            historyList.appendChild(historyElement);
        });
        
        chatHistory = historyData;
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

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
    
    if (historyItem.deliberation_summary || historyItem.metadata) {
        addEnhancedVectorizedMessageToInterface('assistant', historyItem);
    } else {
        addMessageToInterface('assistant', historyItem.response);
    }
}

// Conversation management
function startNewConversation() {
    currentConversationId = 'conv_' + Date.now();
    conversationState = 'new';
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    addMessageTo
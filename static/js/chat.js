// Weather RAG Chat Interface JavaScript - Clean Version
class WeatherChatInterface {
    constructor() {
        this.socket = io();
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.inputSuggestions = document.getElementById('input-suggestions');
        
        this.isConnected = false;
        this.isTyping = false;
        this.messageHistory = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupSocketEvents();
        this.checkSystemStatus();
        this.addWelcomeMessage();
    }
    
    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Show/hide suggestions
        this.messageInput.addEventListener('input', () => {
            this.toggleSuggestions();
        });
        
        this.messageInput.addEventListener('focus', () => {
            this.showSuggestions();
        });
        
        this.messageInput.addEventListener('blur', () => {
            setTimeout(() => this.hideSuggestions(), 200);
        });
        
        // Clear chat button
        document.getElementById('clear-chat').addEventListener('click', () => {
            this.clearChat();
        });
        
        // Export chat button
        document.getElementById('export-chat').addEventListener('click', () => {
            this.exportChat();
        });
        
        // Auto-scroll to bottom when new messages arrive
        const observer = new MutationObserver(() => {
            this.scrollToBottom();
        });
        observer.observe(this.chatMessages, { childList: true });
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.addSystemMessage('Connection lost. Attempting to reconnect...');
        });
        
        this.socket.on('message', (data) => {
            this.hideLoading();
            this.addMessage(data.type, data.content, data.timestamp);
        });
        
        this.socket.on('typing', () => {
            this.showTypingIndicator();
        });
        
        this.socket.on('stop_typing', () => {
            this.hideTypingIndicator();
        });
        
        this.socket.on('error', (error) => {
            this.hideLoading();
            this.addMessage('error', `Connection error: ${error}`, new Date().toISOString());
        });
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        if (!this.isConnected) {
            this.addSystemMessage('Not connected to server. Please refresh the page.');
            return;
        }
        
        // Clear input
        this.messageInput.value = '';
        this.hideSuggestions();
        
        // Add to history
        this.messageHistory.push(message);
        
        // Show loading for analysis commands
        if (message.startsWith('/analyze') || message.startsWith('/predict') || 
            message.toLowerCase().includes('analyze') || message.toLowerCase().includes('hurricane') ||
            message.toLowerCase().includes('tornado') || message.toLowerCase().includes('weather')) {
            this.showLoading();
        }
        
        // Send to server
        this.socket.emit('message', { message: message });
    }
    
    addMessage(type, content, timestamp) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const avatar = this.createAvatar(type);
        const messageContent = this.createMessageContent(content, timestamp, type);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
    }
    
    createAvatar(type) {
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        const icons = {
            'bot': '<i class="fas fa-robot"></i>',
            'user': '<i class="fas fa-user"></i>',
            'system': '<i class="fas fa-cog"></i>',
            'error': '<i class="fas fa-exclamation-triangle"></i>'
        };
        
        avatar.innerHTML = icons[type] || '<i class="fas fa-comment"></i>';
        return avatar;
    }
    
    createMessageContent(content, timestamp, type) {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const text = document.createElement('div');
        text.className = 'message-text';
        
        // Format content
        text.innerHTML = this.formatMessageContent(content);
        
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.formatTimestamp(timestamp);
        
        bubble.appendChild(text);
        contentDiv.appendChild(bubble);
        contentDiv.appendChild(timestampDiv);
        
        return contentDiv;
    }
    
    formatMessageContent(content) {
        // Convert markdown-like formatting to HTML
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>')            // Inline code
            .replace(/\n/g, '<br>')                            // Line breaks
            .replace(/• /g, '• ')                              // Bullet points
            .replace(/(\d+\. )/g, '<strong>$1</strong>');      // Numbered lists
        
        // Add risk level indicators
        formatted = formatted
            .replace(/Risk Level (\d+)\/10/g, (match, level) => {
                const riskClass = this.getRiskClass(parseInt(level));
                return `<span class="risk-indicator ${riskClass}">Risk Level ${level}/10</span>`;
            });
        
        return formatted;
    }
    
    getRiskClass(level) {
        if (level >= 8) return 'risk-extreme';
        if (level >= 6) return 'risk-high';
        if (level >= 4) return 'risk-moderate';
        return 'risk-low';
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    addSystemMessage(message) {
        this.addMessage('system', message, new Date().toISOString());
    }
    
    addWelcomeMessage() {
        // Add initial welcome message if chat is empty
        if (this.chatMessages.children.length === 0) {
            setTimeout(() => {
                this.addSystemMessage('Weather RAG System Ready! Ask me about extreme weather conditions or use /help for commands.');
            }, 500);
        }
    }
    
    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.showTypingIndicator();
    }
    
    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        this.hideTypingIndicator();
    }
    
    showTypingIndicator() {
        if (this.isTyping) return;
        
        this.isTyping = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-message';
        typingDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <span>AI is analyzing weather patterns</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        const typingMessage = this.chatMessages.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
    }
    
    showSuggestions() {
        if (this.messageInput.value.trim() === '') {
            this.inputSuggestions.style.display = 'flex';
        }
    }
    
    hideSuggestions() {
        this.inputSuggestions.style.display = 'none';
    }
    
    toggleSuggestions() {
        if (this.messageInput.value.trim() === '') {
            this.showSuggestions();
        } else {
            this.hideSuggestions();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.chatMessages.innerHTML = '';
            this.messageHistory = [];
            this.addWelcomeMessage();
        }
    }
    
    exportChat() {
        const messages = Array.from(this.chatMessages.querySelectorAll('.message')).map(msg => {
            const type = msg.classList[1];
            const content = msg.querySelector('.message-text').textContent;
            const timestamp = msg.querySelector('.message-timestamp').textContent;
            return `[${timestamp}] ${type.toUpperCase()}: ${content}`;
        }).join('\n\n');
        
        const blob = new Blob([messages], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `weather-chat-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    updateConnectionStatus(connected) {
        const ragLight = document.getElementById('rag-light');
        const weatherLight = document.getElementById('weather-light');
        
        if (connected) {
            ragLight.className = 'status-light';
            weatherLight.className = 'status-light';
        } else {
            ragLight.className = 'status-light error';
            weatherLight.className = 'status-light error';
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/health');
            const status = await response.json();
            
            const ragLight = document.getElementById('rag-light');
            const weatherLight = document.getElementById('weather-light');
            
            // Update RAG system status
            if (status.rag_system_ready) {
                ragLight.className = 'status-light';
            } else {
                ragLight.className = 'status-light warning';
            }
            
            // Weather API status
            if (status.weather_api_ready) {
                weatherLight.className = 'status-light';
            } else {
                weatherLight.className = 'status-light warning';
            }
            
        } catch (error) {
            console.error('Failed to check system status:', error);
            document.getElementById('rag-light').className = 'status-light error';
            document.getElementById('weather-light').className = 'status-light error';
        }
    }
}

// Global functions for HTML onclick events
function sendCommand(command) {
    if (window.chatInterface) {
        window.chatInterface.messageInput.value = command;
        window.chatInterface.sendMessage();
    }
}

function sendSuggestion(suggestion) {
    if (window.chatInterface) {
        window.chatInterface.messageInput.value = suggestion;
        window.chatInterface.messageInput.focus();
        window.chatInterface.hideSuggestions();
    }
}

function analyzeLocation(location) {
    if (window.chatInterface) {
        window.chatInterface.messageInput.value = `/analyze ${location}`;
        window.chatInterface.sendMessage();
    }
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new WeatherChatInterface();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            window.chatInterface.messageInput.focus();
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            window.chatInterface.messageInput.value = '';
            window.chatInterface.messageInput.blur();
            window.chatInterface.hideSuggestions();
        }
    });
    
    // Add helpful tooltips
    addTooltips();
    addKeyboardHints();
});

function addTooltips() {
    // Add title attributes for better UX
    document.getElementById('send-button').title = 'Send message (Enter)';
    document.getElementById('clear-chat').title = 'Clear chat history';
    document.getElementById('export-chat').title = 'Export chat to file';
    document.getElementById('message-input').title = 'Type your weather question or command';
}

function addKeyboardHints() {
    // Add keyboard shortcut hints
    const inputHelp = document.querySelector('.input-help small');
    if (inputHelp) {
        inputHelp.innerHTML += ' • Press Ctrl+K to focus input • Press Escape to clear';
    }
}

// Error handling for network issues
window.addEventListener('online', () => {
    if (window.chatInterface) {
        window.chatInterface.addSystemMessage('Connection restored');
        window.chatInterface.updateConnectionStatus(true);
    }
});

window.addEventListener('offline', () => {
    if (window.chatInterface) {
        window.chatInterface.addSystemMessage('No internet connection');
        window.chatInterface.updateConnectionStatus(false);
    }
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log(`Page load time: ${perfData.loadEventEnd - perfData.loadEventStart}ms`);
        }, 0);
    });
}
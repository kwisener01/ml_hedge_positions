/**
 * Server-Sent Events (SSE) Client
 * Manages real-time connection to backend
 */

class MonitorSSE {
    constructor() {
        this.eventSource = null;
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;
        this.currentDelay = this.reconnectDelay;
        this.reconnectAttempts = 0;
    }

    connect() {
        console.log('Connecting to SSE endpoint...');

        this.eventSource = new EventSource('/api/events');

        // Connection opened
        this.eventSource.onopen = () => {
            console.log('SSE connection established');
            this.reconnectAttempts = 0;
            this.currentDelay = this.reconnectDelay;
            this.updateConnectionStatus(true);
        };

        // Initial state
        this.eventSource.addEventListener('initial_state', (event) => {
            const data = JSON.parse(event.data);
            console.log('Received initial state:', data);
            this.handleInitialState(data);
        });

        // Quote updates
        this.eventSource.addEventListener('quote_update', (event) => {
            const data = JSON.parse(event.data);
            this.handleQuoteUpdate(data);
        });

        // Greek level updates
        this.eventSource.addEventListener('greek_update', (event) => {
            const data = JSON.parse(event.data);
            this.handleGreekUpdate(data);
        });

        // Strength updates
        this.eventSource.addEventListener('strength_update', (event) => {
            const data = JSON.parse(event.data);
            this.handleStrengthUpdate(data);
        });

        // Signal alerts
        this.eventSource.addEventListener('signal_alert', (event) => {
            const data = JSON.parse(event.data);
            this.handleSignalAlert(data);
        });

        // Monitoring status
        this.eventSource.addEventListener('monitoring_status', (event) => {
            const data = JSON.parse(event.data);
            console.log('Monitoring status:', data.active ? 'Active' : 'Inactive');
        });

        // Connection error
        this.eventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            this.updateConnectionStatus(false);
            this.eventSource.close();
            this.reconnect();
        };
    }

    reconnect() {
        this.reconnectAttempts++;
        console.log(`Reconnecting in ${this.currentDelay/1000}s (attempt ${this.reconnectAttempts})...`);

        setTimeout(() => {
            this.connect();
        }, this.currentDelay);

        // Exponential backoff
        this.currentDelay = Math.min(this.currentDelay * 2, this.maxReconnectDelay);
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.updateConnectionStatus(false);
    }

    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('#connection-status span:last-child');

        if (connected) {
            statusDot.classList.remove('status-dot-disconnected');
            statusDot.classList.add('status-dot-connected');
            statusText.textContent = 'LIVE';
        } else {
            statusDot.classList.remove('status-dot-connected');
            statusDot.classList.add('status-dot-disconnected');
            statusText.textContent = 'DISCONNECTED';
        }
    }

    handleInitialState(data) {
        // Update quote if available
        if (data.quote) {
            window.updateQuote(data.quote);
        }

        // Update Greeks if available
        if (data.greeks) {
            window.updateGreeks(data.greeks);
        }

        // Update strength if available
        if (data.strength) {
            window.updateStrength(data.strength, data.confidence, data.model_probability);
        }

        // Load recent signals
        if (data.recent_signals && data.recent_signals.length > 0) {
            window.loadRecentSignals(data.recent_signals);
        }

        // Update last update time
        if (data.last_update) {
            window.updateLastUpdate(data.last_update);
        }
    }

    handleQuoteUpdate(data) {
        window.updateQuote(data);
    }

    handleGreekUpdate(data) {
        window.updateGreeks(data);
    }

    handleStrengthUpdate(data) {
        window.updateStrength(data.strength, data.confidence, data.model_probability);
    }

    handleSignalAlert(data) {
        console.log('SIGNAL ALERT:', data);
        window.showAlertBanner(data);
        window.addSignalToTable(data);
        window.playAlertSound();
    }
}

// Initialize SSE connection when page loads
let monitorSSE;

document.addEventListener('DOMContentLoaded', () => {
    monitorSSE = new MonitorSSE();
    monitorSSE.connect();
});

// Disconnect on page unload
window.addEventListener('beforeunload', () => {
    if (monitorSSE) {
        monitorSSE.disconnect();
    }
});

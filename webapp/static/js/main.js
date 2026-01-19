/**
 * Main Dashboard UI Updates
 * Functions called by sse.js to update the UI
 */

// Update quote display
window.updateQuote = function(data) {
    const priceEl = document.getElementById('current-price');
    const spreadEl = document.getElementById('current-spread');

    if (data.price !== undefined) {
        priceEl.textContent = `$${data.price.toFixed(2)}`;
    }

    if (data.spread !== undefined) {
        spreadEl.textContent = `$${data.spread.toFixed(4)}`;
    }

    // Update last update time
    if (data.timestamp) {
        updateLastUpdate(data.timestamp);
    }
};

// Update Greek levels
window.updateGreeks = function(data) {
    const levels = {
        'hp-support': data.hp_support,
        'hp-resist': data.hp_resistance,
        'mhp-support': data.mhp_support,
        'mhp-resist': data.mhp_resistance,
        'hg-below': data.hg_below,
        'hg-above': data.hg_above,
        'gamma-flip': data.gamma_flip
    };

    for (const [id, value] of Object.entries(levels)) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value ? `$${value.toFixed(2)}` : '$---.--';
        }
    }
};

// Update strength meter and components
window.updateStrength = function(strengthData, confidence, modelProb) {
    // Strength meter
    const strengthValue = strengthData.total;
    const strengthBar = document.getElementById('strength-bar');
    const strengthValueEl = document.getElementById('strength-value');
    const strengthLabel = document.getElementById('strength-label');

    strengthValueEl.textContent = `${strengthValue.toFixed(1)}/100`;
    strengthBar.style.width = `${strengthValue}%`;

    // Update color based on strength
    strengthBar.classList.remove('strength-weak', 'strength-moderate', 'strength-strong');
    if (strengthValue < 15) {
        strengthBar.classList.add('strength-weak');
        strengthLabel.textContent = 'Weak';
    } else if (strengthValue <= 30) {
        strengthBar.classList.add('strength-moderate');
        strengthLabel.textContent = 'Moderate';
    } else {
        strengthBar.classList.add('strength-strong');
        strengthLabel.textContent = 'Strong';
    }

    // Level type and source
    document.getElementById('level-type').textContent = strengthData.level_type || '-';
    document.getElementById('level-source').textContent = strengthData.level_source || '-';

    // Components
    const components = strengthData.components || {};
    document.getElementById('comp-gamma').textContent = (components.gamma || 0).toFixed(1);
    document.getElementById('comp-vanna').textContent = (components.vanna || 0).toFixed(1);
    document.getElementById('comp-hp').textContent = (components.hp || 0).toFixed(1);
    document.getElementById('comp-mhp').textContent = (components.mhp || 0).toFixed(1);
    document.getElementById('comp-hg').textContent = (components.hg || 0).toFixed(1);
    document.getElementById('comp-overlap').textContent = (components.overlap || 0).toFixed(1);

    // Confidence
    const confidenceValueEl = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const modelProbEl = document.getElementById('model-prob');
    const statusText = document.getElementById('status-text');

    confidenceValueEl.textContent = `${(confidence * 100).toFixed(1)}%`;
    confidenceBar.style.width = `${confidence * 100}%`;
    modelProbEl.textContent = `Model P(UP): ${(modelProb * 100).toFixed(1)}%`;

    // Update status
    const isSetup = strengthValue >= 15 && strengthValue <= 30 &&
                    strengthData.level_type === 'support' &&
                    confidence > 0.55;

    if (isSetup) {
        statusText.textContent = 'SETUP DETECTED - Watch for entry!';
        statusText.classList.add('text-yellow-400');
        statusText.classList.remove('text-gray-400');
    } else {
        statusText.textContent = 'WATCHING (No signal)';
        statusText.classList.remove('text-yellow-400');
        statusText.classList.add('text-gray-400');
    }
};

// Show alert banner
window.showAlertBanner = function(data) {
    const banner = document.getElementById('alert-banner');

    // Populate alert data
    document.getElementById('alert-entry').textContent = `$${data.entry_price.toFixed(2)}`;
    document.getElementById('alert-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('alert-strength').textContent = data.strength_score.toFixed(1);
    document.getElementById('alert-level').textContent = data.level_source;
    document.getElementById('alert-stop').textContent = `$${data.suggested_stop.toFixed(2)}`;
    document.getElementById('alert-target').textContent = `$${data.suggested_target.toFixed(2)}`;

    // Show banner
    banner.classList.remove('hidden');
    banner.classList.add('fade-in');

    // Auto-hide after 30 seconds
    setTimeout(() => {
        banner.classList.add('hidden');
    }, 30000);
};

// Dismiss alert button
document.getElementById('dismiss-alert')?.addEventListener('click', () => {
    document.getElementById('alert-banner').classList.add('hidden');
});

// Add signal to table
window.addSignalToTable = function(data) {
    const table = document.getElementById('signals-table');

    // Remove "no signals" message if present
    if (table.querySelector('td[colspan]')) {
        table.innerHTML = '';
    }

    // Create new row
    const row = document.createElement('tr');
    row.classList.add('fade-in');

    const time = new Date(data.timestamp).toLocaleTimeString();

    row.innerHTML = `
        <td>${time}</td>
        <td><span class="px-2 py-1 bg-green-600 rounded text-xs">${data.signal_type}</span></td>
        <td class="font-mono">$${data.entry_price.toFixed(2)}</td>
        <td class="font-mono">${(data.confidence * 100).toFixed(1)}%</td>
        <td class="font-mono">${data.strength_score.toFixed(1)}</td>
        <td>${data.level_source}</td>
    `;

    // Insert at top
    table.insertBefore(row, table.firstChild);

    // Keep only last 50 signals
    while (table.children.length > 50) {
        table.removeChild(table.lastChild);
    }
};

// Load recent signals
window.loadRecentSignals = function(signals) {
    const table = document.getElementById('signals-table');
    table.innerHTML = '';

    if (!signals || signals.length === 0) {
        table.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-gray-500">No signals yet</td></tr>';
        return;
    }

    signals.forEach(signal => {
        addSignalToTable(signal);
    });
};

// Update last update time
window.updateLastUpdate = function(timestamp) {
    const el = document.getElementById('last-update');
    const date = new Date(timestamp);
    el.textContent = date.toLocaleTimeString();
};

// Play alert sound (optional)
window.playAlertSound = function() {
    // Simple beep using Web Audio API
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800; // Hz
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    } catch (e) {
        console.log('Audio not supported or blocked');
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized');
});

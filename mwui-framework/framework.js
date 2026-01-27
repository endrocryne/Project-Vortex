/**
 * UI Framework JavaScript
 * Core functionality for the reusable UI framework
 * Handles theming, tab navigation, settings management, and modal control
 */

class UIFramework {
    constructor(config = {}) {
        this.config = {
            appName: config.appName || 'My App',
            defaultTheme: config.defaultTheme || 'light',
            defaultStyle: config.defaultStyle || 'liquid-glass',
            defaultAccentColor: config.defaultAccentColor || null,
            enableSettings: config.enableSettings !== false,
            tabs: config.tabs || [],
            onTabChange: config.onTabChange || null,
            ...config
        };
        
        this.initialize();
    }

    /**
     * Initialize the framework
     */
    initialize() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    /**
     * Setup framework components
     */
    setup() {
        this.setupElements();
        this.setupTabNavigation();
        this.setupSettings();
        this.loadSettings();
        this.updateLiquidGlassUI();
    }

    /**
     * Get references to DOM elements
     */
    setupElements() {
        // Header elements
        this.titleElement = document.querySelector('.title');
        if (this.titleElement) {
            this.titleElement.textContent = this.config.appName;
        }

        // Settings elements
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettingsBtn = document.getElementById('closeSettings');
        this.saveSettingsBtn = document.getElementById('saveSettings');
        
        // Theme controls
        this.themeSelect = document.getElementById('themeSelect');
        this.styleSelect = document.getElementById('styleSelect');
        this.accentColorInput = document.getElementById('accentColorInput');
        this.fontSizeSlider = document.getElementById('fontSizeSlider');
        this.highContrastToggle = document.getElementById('highContrastToggle');

        // Tabs
        this.tabLinks = document.querySelectorAll('.tab-link');
        this.tabContents = document.querySelectorAll('.tab-content');
    }

    /**
     * Setup tab navigation system
     */
    setupTabNavigation() {
        // Create global openTab function for inline onclick handlers
        window.openTab = (evt, tabName) => {
            // Hide all tab contents
            this.tabContents.forEach(content => {
                content.style.display = 'none';
            });

            // Remove active class from all tabs
            this.tabLinks.forEach(link => {
                link.classList.remove('active');
            });

            // Show selected tab and mark as active
            const selectedTab = document.getElementById(tabName);
            if (selectedTab) {
                selectedTab.style.display = 'block';
            }
            
            if (evt && evt.currentTarget) {
                evt.currentTarget.classList.add('active');
            }

            // Call custom tab change handler if provided
            if (this.config.onTabChange) {
                this.config.onTabChange(tabName);
            }
        };

        // Initialize tab visibility on setup so first load doesn't show all contents stacked
        // Hide all tab contents, then show the active one (or first if none active)
        if (this.tabContents.length) {
            this.tabContents.forEach(content => content.style.display = 'none');

            const activeIndex = Array.from(this.tabLinks).findIndex(link => link.classList.contains('active'));
            if (activeIndex >= 0 && this.tabContents[activeIndex]) {
                this.tabContents[activeIndex].style.display = 'block';
            } else if (this.tabContents[0]) {
                this.tabContents[0].style.display = 'block';
                if (this.tabLinks[0]) this.tabLinks[0].classList.add('active');
            }
        }
    }

    /**
     * Setup settings modal and controls
     */
    setupSettings() {
        if (!this.config.enableSettings) return;

        // Open settings modal
        if (this.settingsBtn) {
            this.settingsBtn.addEventListener('click', () => {
                if (this.settingsModal) {
                    this.settingsModal.style.display = 'block';
                }
            });
        }

        // Close settings modal
        if (this.closeSettingsBtn) {
            this.closeSettingsBtn.addEventListener('click', () => {
                this.closeSettings();
            });
        }

        // Close modal on outside click
        window.addEventListener('click', (event) => {
            if (event.target === this.settingsModal) {
                this.closeSettings();
            }
        });

        // Save settings
        if (this.saveSettingsBtn) {
            this.saveSettingsBtn.addEventListener('click', () => {
                this.saveSettings();
            });
        }
    }

    /**
     * Close settings modal
     */
    closeSettings() {
        if (this.settingsModal) {
            this.settingsModal.style.display = 'none';
        }
    }

    /**
     * Save settings to localStorage
     */
    saveSettings() {
        // Save theme
        if (this.themeSelect) {
            const theme = this.themeSelect.value;
            localStorage.setItem('theme', theme);
            document.body.setAttribute('data-theme', theme);
        }

        // Save style
        if (this.styleSelect) {
            const style = this.styleSelect.value;
            localStorage.setItem('style', style);
            document.body.className = style;
        }

        // Save accent color
        if (this.accentColorInput) {
            const accentColor = this.accentColorInput.value;
            localStorage.setItem('accentColor', accentColor);
            this.setAccentColor(accentColor);
        }

        // Save font size
        if (this.fontSizeSlider) {
            const fontSize = this.fontSizeSlider.value;
            localStorage.setItem('fontSize', fontSize);
            document.documentElement.style.fontSize = `${fontSize}px`;
        }

        // Save high contrast setting
        if (this.highContrastToggle) {
            const highContrast = this.highContrastToggle.checked;
            localStorage.setItem('highContrast', highContrast);
            
            if (highContrast) {
                document.body.setAttribute('data-theme', 'high-contrast');
            } else if (this.themeSelect) {
                document.body.setAttribute('data-theme', this.themeSelect.value);
            }
        }

        this.updateLiquidGlassUI();
        this.closeSettings();
    }

    /**
     * Load settings from localStorage
     */
    loadSettings() {
        // Load theme
        const theme = localStorage.getItem('theme') || this.config.defaultTheme;
        if (this.themeSelect) {
            this.themeSelect.value = theme;
        }
        document.body.setAttribute('data-theme', theme);

        // Load style
        const style = localStorage.getItem('style') || this.config.defaultStyle;
        if (this.styleSelect) {
            this.styleSelect.value = style;
        }
        document.body.className = style;

        // Load accent color
        const defaultAccent = this.config.defaultAccentColor || 
            (theme === 'dark' ? '#bb86fc' : '#1976d2');
        const accentColor = localStorage.getItem('accentColor') || defaultAccent;
        if (this.accentColorInput) {
            this.accentColorInput.value = accentColor;
        }
        this.setAccentColor(accentColor);

        // Load font size
        const fontSize = localStorage.getItem('fontSize') || '16';
        if (this.fontSizeSlider) {
            this.fontSizeSlider.value = fontSize;
        }
        document.documentElement.style.fontSize = `${fontSize}px`;

        // Load high contrast
        const highContrast = localStorage.getItem('highContrast') === 'true';
        if (this.highContrastToggle) {
            this.highContrastToggle.checked = highContrast;
        }
        if (highContrast) {
            document.body.setAttribute('data-theme', 'high-contrast');
        }
    }

    /**
     * Update UI for liquid glass mode
     * Handles special positioning for liquid glass design
     */
    updateLiquidGlassUI() {
        const tabs = document.querySelector('.tabs');
        const controls = document.querySelector('.header .controls');
        
        if (!this.settingsBtn || !tabs || !controls) return;

        if (document.body.classList.contains('liquid-glass')) {
            // Move settings button to tabs in liquid glass mode
            if (!tabs.contains(this.settingsBtn)) {
                this.settingsBtn.classList.add('settings-tab-btn');
                // Give it the same visual treatment as other tab links so it's visible and accessible
                this.settingsBtn.classList.add('tab-link');
                this.settingsBtn.setAttribute('role', 'tab');
                this.settingsBtn.style.display = 'inline-flex';
                this.settingsBtn.style.alignItems = 'center';
                // Push it to the far right so it behaves like a settings control in the tab bar
                this.settingsBtn.style.marginLeft = 'auto';
                tabs.appendChild(this.settingsBtn);
            }
        } else {
            // Move settings button back to header controls
            if (!controls.contains(this.settingsBtn)) {
                this.settingsBtn.classList.remove('settings-tab-btn');
                // Remove tab-link related adjustments we added when it was moved into tabs
                this.settingsBtn.classList.remove('tab-link');
                this.settingsBtn.removeAttribute('role');
                this.settingsBtn.style.display = '';
                this.settingsBtn.style.alignItems = '';
                this.settingsBtn.style.marginLeft = '';
                controls.appendChild(this.settingsBtn);
            }
        }
    }

    /**
     * Set accent color and update CSS variables
     */
    setAccentColor(color) {
        const root = document.documentElement;
        root.style.setProperty('--primary', color);
        root.style.setProperty('--primary-variant', this.darkenColor(color, 15));
        
        const rgb = this.hexToRgb(color);
        if (rgb) {
            root.style.setProperty('--primary-rgb', `${rgb.r}, ${rgb.g}, ${rgb.b}`);
        }
    }

    /**
     * Darken a hex color by a percentage
     */
    darkenColor(hex, percent) {
        const p = percent / 100;
        let [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
        r = Math.floor(r * (1 - p));
        g = Math.floor(g * (1 - p));
        b = Math.floor(b * (1 - p));
        return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
    }

    /**
     * Convert hex color to RGB object
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    /**
     * Show a modal by ID
     */
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'block';
        }
    }

    /**
     * Hide a modal by ID
     */
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
        }
    }

    /**
     * Create a modal programmatically
     */
    createModal(config) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.id = config.id || 'dynamic-modal';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">${config.title || 'Modal'}</div>
                    <button class="close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    ${config.content || ''}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add close functionality
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => {
            this.hideModal(modal.id);
        });
        
        window.addEventListener('click', (event) => {
            if (event.target === modal) {
                this.hideModal(modal.id);
            }
        });
        
        return modal;
    }
}

// Export for use in modules or scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIFramework;
}

/**
 * UI Customization Module
 * Provides drag-and-drop, resize, and layout customization capabilities
 */

class UICustomizer {
    constructor(framework) {
        this.framework = framework;
        this.customizationMode = false;
        this.layoutConfig = this.loadLayout();
        this.interactLoaded = false;

        // Defaults
        this.gridSize = 8;
        // Load saved grid size if present
        const savedGrid = localStorage.getItem('uiGridSize');
        if (savedGrid) this.gridSize = Math.max(4, Number(savedGrid) || 8);
        this.snapRangePx = 10; // threshold for snaplines

        // Container overlays for snaplines and guides
        this.containers = new Map(); // container -> {overlay, snapLines}
        this._pending = new Map(); // container -> [elements waiting for visibility]
        this._containerObservers = new Map();

        this.setupCustomizationControls();
    }

    /* -------------------------
       Script loader for interact.js
       ------------------------- */
    loadInteract() {
        if (this.interactLoaded || window.interact) {
            this.interactLoaded = true;
            return Promise.resolve(window.interact);
        }

        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/interactjs/dist/interact.min.js';
            script.onload = () => {
                this.interactLoaded = true;
                resolve(window.interact);
            };
            script.onerror = () => reject(new Error('Failed to load interact.js'));
            document.head.appendChild(script);
        });
    }

    /**
     * Ensure customization controls exist in the settings modal (idempotent)
     */
    setupCustomizationControls() {
        const ensure = () => {
            const settingsModal = document.getElementById('settingsModal');
            if (!settingsModal) return;

            // Avoid duplication
            if (document.getElementById('customize-toolbar')) return;

            const saveButton = document.getElementById('saveSettings');
            if (!saveButton) return;

            const customizationSection = document.createElement('div');
            customizationSection.id = 'customize-toolbar';
            customizationSection.innerHTML = `
                <div class="form-group">
                    <label for="customizeLayoutToggle">Customize Layout:</label>
                    <label class="switch">
                        <input type="checkbox" id="customizeLayoutToggle">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="form-group">
                    <label for="gridSizeInput">Grid Size (px):</label>
                    <input type="number" id="gridSizeInput" min="4" max="64" value="8">
                </div>
                <div class="form-group">
                    <button type="button" id="resetLayoutBtn" style="width:100%;">Reset Layout</button>
                </div>
            `;

            saveButton.parentNode.insertBefore(customizationSection, saveButton);

            // Wire up listeners (guard against missing nodes)
            const toggle = document.getElementById('customizeLayoutToggle');
            const gridInput = document.getElementById('gridSizeInput');
            const resetBtn = document.getElementById('resetLayoutBtn');

            if (toggle) toggle.addEventListener('change', (e) => this.toggleCustomizationMode(e.target.checked));
        if (gridInput) gridInput.addEventListener('change', (e) => {
            this.gridSize = Math.max(4, Number(e.target.value) || 8);
            try { localStorage.setItem('uiGridSize', String(this.gridSize)); } catch (err) { /* ignore */ }
        });

            // Apply initial value
            if (gridInput) gridInput.value = this.gridSize;
        };

        // Try once now
        ensure();

        // Also ensure controls exist whenever settings modal is opened (in case modal content is re-rendered)
        const settingsBtn = this.framework && this.framework.settingsBtn ? this.framework.settingsBtn : document.getElementById('settingsBtn');
        if (settingsBtn && !settingsBtn._customizerBound) {
            settingsBtn.addEventListener('click', () => setTimeout(ensure, 0));
            settingsBtn._customizerBound = true;
        }

        // Watch for DOM changes under settingsModal to re-run ensure (handles re-renders)
        const settingsModal = document.getElementById('settingsModal');
        if (settingsModal && !settingsModal._customizerObserver) {
            const obs = new MutationObserver(() => ensure());
            obs.observe(settingsModal, { childList: true, subtree: true });
            settingsModal._customizerObserver = obs;
        }
    }

    /**
     * Toggle customization mode (async to load interact.js)
     */
    async toggleCustomizationMode(enabled) {
        this.customizationMode = enabled;
        if (enabled) {
            try {
                await this.loadInteract();
            } catch (err) {
                alert('Failed to load customization library. Check console for details.');
                console.error(err);
                return;
            }
            this.enableCustomization();
            this.showCustomizationHints();
        } else {
            this.disableCustomization();
            this.hideCustomizationHints();
        }
    }

    /**
     * Enable customization: make elements absolute within their container and attach interact.js
     */
    enableCustomization() {
        const elems = this.getCustomizableElements();
        if (elems.length === 0) return;

        // Create overlays and prepare containers
        elems.forEach(el => {
            // Support both framework conventions: .tab-content (old) and .tab-panel (new)
            const container = el.closest('.tab-content, .tab-panel') || document.body;
            if (!this.containers.has(container)) {
                this.prepareContainer(container);
            }
        });

        // Make each element absolute and attach interact but skip elements inside hidden tabs
        elems.forEach(el => {
            // Support both .tab-content and .tab-panel
            const container = el.closest('.tab-content, .tab-panel') || document.body;
            const containerStyle = window.getComputedStyle(container);

            if (container !== document.body && (container.offsetWidth === 0 || container.offsetHeight === 0 || containerStyle.display === 'none')) {
                // queue for processing when the tab becomes visible
                this._queueElementForContainer(container, el);
            } else {
                this.makeElementCustomizable(el);
            }
        });

        // Apply saved layout if present (positions are applied even for hidden tabs)
        this.applyLayout();
    }

    /**
     * Prepare container: ensure relative positioning and add overlay for snaplines
     */
    prepareContainer(container) {
        const computed = window.getComputedStyle(container);
        let origPositionChanged = false;
        if (computed.position === 'static') {
            container.style.position = 'relative';
            origPositionChanged = true;
        }

        // create overlay for snap lines
        const overlay = document.createElement('div');
        overlay.className = 'customization-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = 0;
        overlay.style.left = 0;
        overlay.style.right = 0;
        overlay.style.bottom = 0;
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = 9000;

        container.appendChild(overlay);

        this.containers.set(container, { overlay, snapLines: [], origPositionChanged });
    }

    /**
     * Convert element into an absolutely positioned, interact-enabled item
     */
    makeElementCustomizable(element) {
        if (element.dataset.customizable === 'true') return; // already setup

        // Support both .tab-content and .tab-panel as container roots
        // Prefer explicit panel containers, fall back to offsetParent (positioning ancestor), then body
        const container = element.closest('.tab-content, .tab-panel') || element.offsetParent || document.body;
        const containerStyle = window.getComputedStyle(container);

        // If the element lives inside a hidden container, defer processing
        if (container !== document.body && (container.offsetWidth === 0 || container.offsetHeight === 0 || containerStyle.display === 'none')) {
            this._queueElementForContainer(container, element);
            return;
        }

        element.dataset.customizable = 'true';
        element.classList.add('ui-customizable');

        // store original inline styles to restore later (if not already stored)
        if (element.dataset._origPosition === undefined) element.dataset._origPosition = element.style.position || '';
        if (element.dataset._origLeft === undefined) element.dataset._origLeft = element.style.left || '';
        if (element.dataset._origTop === undefined) element.dataset._origTop = element.style.top || '';
        if (element.dataset._origWidth === undefined) element.dataset._origWidth = element.style.width || '';
        if (element.dataset._origHeight === undefined) element.dataset._origHeight = element.style.height || '';

        const containerRect = container.getBoundingClientRect();
        const rect = element.getBoundingClientRect();

        // Convert to absolute positioning within container
        element.style.position = 'absolute';

        // If a saved position/size already exists (from applyLayout), preserve it instead of re-calculating
        const hasSavedStyle = element.style.left || element.style.top || element.style.width || element.style.height;
        if (!hasSavedStyle) {
            element.style.left = (rect.left - containerRect.left) + 'px';
            element.style.top = (rect.top - containerRect.top) + 'px';
            element.style.width = rect.width + 'px';
            element.style.height = rect.height + 'px';
        }

        element.style.zIndex = 1000;

        // Add handles and classes
        if (!element.querySelector('.ui-drag-handle')) {
            const dragHandle = document.createElement('div');
            dragHandle.className = 'ui-drag-handle';
            dragHandle.innerHTML = '⋮⋮';
            element.appendChild(dragHandle);
        }

        // Add resize handles (interact will use edges)
        if (!document.getElementById('customization-styles')) this.addCustomizationStyles();

        // Setup interact draggable + resizable
        const grid = this.gridSize;
        const snapRange = this.snapRangePx;
        const self = this;
        const containerRectForRestrict = container;

        window.interact(element).draggable({
            listeners: {
                move (event) { self.onDragMove(event); },
                end (event) { self.onDragEnd(event); }
            },
            modifiers: [
                window.interact.modifiers.snap({
                    targets: [window.interact.createSnapGrid({ x: grid, y: grid })],
                    range: snapRange,
                    relativePoints: [{ x: 0, y: 0 }]
                }),
                window.interact.modifiers.restrictRect({ restriction: containerRectForRestrict, endOnly: true })
            ],
            inertia: true
        }).resizable({
            edges: { left: true, right: true, bottom: true, top: true },
            listeners: {
                move (event) { self.onResizeMove(event); },
                end (event) { self.onResizeEnd(event); }
            },
            modifiers: [
                window.interact.modifiers.snapSize({ targets: [window.interact.createSnapGrid({ x: grid, y: grid })] }),
                window.interact.modifiers.restrictSize({ min: { width: 80, height: 40 }, max: { width: container.clientWidth, height: container.clientHeight } })
            ],
            inertia: true
        });

        // show element border
        element.classList.add('ui-customizable-active');
    }

    /**
     * Drag move handler - applies translation and shows snaplines
     */
    onDragMove(event) {
        const target = event.target;
        const dx = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx;
        const dy = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;

        target.style.webkitTransform = target.style.transform = 'translate(' + dx + 'px, ' + dy + 'px)';
        target.setAttribute('data-x', dx);
        target.setAttribute('data-y', dy);

        // When dragging ends, we'll commit translation to left/top
        this.updateSnapGuides(target);
    }

    onDragEnd(event) {
        const target = event.target;
        // translation applied during drag is available via data-x/y OR via transform on bounding rect
        const dx = parseFloat(target.getAttribute('data-x')) || 0;
        const dy = parseFloat(target.getAttribute('data-y')) || 0;

        // commit final position using bounding client rect to avoid issues when inline left/top were not set
        const container = target.closest('.tab-content, .tab-panel') || document.body;
        const containerRect = container.getBoundingClientRect();
        const rect = target.getBoundingClientRect();

        // left/top relative to container (rect already includes current translation)
        target.style.left = Math.round(rect.left - containerRect.left) + 'px';
        target.style.top  = Math.round(rect.top  - containerRect.top) + 'px';

        // clear transform and transient attributes
        target.style.transform = '';
        target.removeAttribute('data-x');
        target.removeAttribute('data-y');

        // hide snaplines after move
        this.clearSnapGuides(target);

        // Save layout after each drag
        this.saveLayout();

        // If there are pending elements for this container (dragging may have changed layout), process them
        if (this._pending && this._pending.has(container)) {
            this._processPendingForContainer(container);
        }
    }

    /**
     * Resize move handler - applies size and checks snaplines
     */
    onResizeMove(event) {
        const target = event.target;
        let x = parseFloat(target.getAttribute('data-x')) || 0;
        let y = parseFloat(target.getAttribute('data-y')) || 0;

        // update the element's style
        target.style.width  = event.rect.width + 'px';
        target.style.height = event.rect.height + 'px';

        // translate when resizing from top or left edges
        x += event.deltaRect.left;
        y += event.deltaRect.top;

        target.style.webkitTransform = target.style.transform = 'translate(' + x + 'px, ' + y + 'px)';
        target.setAttribute('data-x', x);
        target.setAttribute('data-y', y);

        this.updateSnapGuides(target);
    }

    onResizeEnd(event) {
        const target = event.target;

        // commit final position and size using bounding rect (handles translations applied during resize)
        const container = target.closest('.tab-content, .tab-panel') || document.body;
        const containerRect = container.getBoundingClientRect();
        const rect = target.getBoundingClientRect();

        target.style.left = Math.round(rect.left - containerRect.left) + 'px';
        target.style.top  = Math.round(rect.top  - containerRect.top) + 'px';

        // clear transient transforms/attributes
        target.style.transform = '';
        target.removeAttribute('data-x');
        target.removeAttribute('data-y');

        this.clearSnapGuides(target);
        this.saveLayout();

        // If there are pending elements for this container (resizing may have changed layout), process them
        if (this._pending && this._pending.has(container)) {
            this._processPendingForContainer(container);
        }
    }

    /**
     * Show snap guides when element aligns with grid or other elements
     */
    updateSnapGuides(target) {
        const container = target.closest('.tab-content, .tab-panel') || document.body;
        const info = this.containers.get(container);
        if (!info) return;

        // Clear existing lines
        this.clearSnapGuides(target);

        const overlay = info.overlay;
        const threshold = this.snapRangePx;

        const targetRect = target.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        // compute candidate lines (edges + centers) from other elements
        const others = Array.from(container.querySelectorAll('.ui-customizable')).filter(el => el !== target);

        // normalize rects relative to container
        const t = {
            left: targetRect.left - containerRect.left,
            right: targetRect.right - containerRect.left,
            top: targetRect.top - containerRect.top,
            bottom: targetRect.bottom - containerRect.top,
            centerX: (targetRect.left - containerRect.left + targetRect.right - containerRect.left) / 2,
            centerY: (targetRect.top - containerRect.top + targetRect.bottom - containerRect.top) / 2
        };

        // check for snaps with other elements
        others.forEach(o => {
            const r = o.getBoundingClientRect();
            const or = {
                left: r.left - containerRect.left,
                right: r.right - containerRect.left,
                top: r.top - containerRect.top,
                bottom: r.bottom - containerRect.top,
                centerX: (r.left - containerRect.left + r.right - containerRect.left) / 2,
                centerY: (r.top - containerRect.top + r.bottom - containerRect.top) / 2
            };

            // horizontal alignments (show vertical lines)
            [[t.left, or.left], [t.left, or.right], [t.right, or.left], [t.right, or.right], [t.centerX, or.centerX]].forEach(([a, b]) => {
                if (Math.abs(a - b) <= threshold) {
                    const x = Math.round(b);
                    this.drawVerticalGuide(overlay, x);
                }
            });

            // vertical alignments (show horizontal lines)
            [[t.top, or.top], [t.top, or.bottom], [t.bottom, or.top], [t.bottom, or.bottom], [t.centerY, or.centerY]].forEach(([a, b]) => {
                if (Math.abs(a - b) <= threshold) {
                    const y = Math.round(b);
                    this.drawHorizontalGuide(overlay, y);
                }
            });
        });
    }

    drawVerticalGuide(overlay, x) {
        const line = document.createElement('div');
        line.className = 'snapline snapline-vertical';
        line.style.position = 'absolute';
        line.style.top = '0';
        line.style.bottom = '0';
        line.style.left = x + 'px';
        line.style.width = '1px';
        line.style.background = 'rgba(255,0,0,0.6)';
        line.style.zIndex = 9200;
        overlay.appendChild(line);
        overlay._lastLine = overlay._lastLine || [];
        overlay._lastLine.push(line);
    }

    drawHorizontalGuide(overlay, y) {
        const line = document.createElement('div');
        line.className = 'snapline snapline-horizontal';
        line.style.position = 'absolute';
        line.style.left = '0';
        line.style.right = '0';
        line.style.top = y + 'px';
        line.style.height = '1px';
        line.style.background = 'rgba(255,0,0,0.6)';
        line.style.zIndex = 9200;
        overlay.appendChild(line);
        overlay._lastLine = overlay._lastLine || [];
        overlay._lastLine.push(line);
    }

    clearSnapGuides(target) {
        const container = target.closest('.tab-content, .tab-panel') || document.body;
        const info = this.containers.get(container);
        if (!info || !info.overlay) return;
        const overlay = info.overlay;
        if (overlay._lastLine) {
            overlay._lastLine.forEach(l => l.remove());
            overlay._lastLine = [];
        }
    }

    /**
     * Queue elements that live in hidden containers so they are processed when the tab becomes visible
     */
    _queueElementForContainer(container, element) {
        if (!this._pending.has(container)) this._pending.set(container, new Set());
        this._pending.get(container).add(element);
        this._observeContainerVisibility(container);
    }

    /**
     * Observe container visibility changes and process pending elements when visible
     */
    _observeContainerVisibility(container) {
        if (this._containerObservers.has(container)) return; // already observing

        const checkAndProcess = () => {
            const style = window.getComputedStyle(container);
            if (style.display !== 'none' && container.offsetWidth > 0 && container.offsetHeight > 0) {
                this._processPendingForContainer(container);
            }
        };

        // Use MutationObserver to detect style/class changes that may reveal the tab
        const obs = new MutationObserver(() => checkAndProcess());
        obs.observe(container, { attributes: true, attributeFilter: ['style', 'class'] });

        // Store and also run an initial check in case visibility changed already
        this._containerObservers.set(container, obs);
        setTimeout(checkAndProcess, 50);
    }

    _processPendingForContainer(container) {
        if (!this._pending.has(container)) return;
        const set = Array.from(this._pending.get(container));
        set.forEach(el => {
            if (!document.body.contains(el)) return; // element gone
            this.makeElementCustomizable(el);
        });
        this._pending.delete(container);

        // disconnect observer if present
        const obs = this._containerObservers.get(container);
        if (obs) {
            obs.disconnect();
            this._containerObservers.delete(container);
        }
    }

    /**
     * Add customization styles (handles, overlays)
     */
    addCustomizationStyles() {
        const style = document.createElement('style');
        style.id = 'customization-styles';
        style.textContent = `
            .ui-customizable { outline: 2px dashed transparent; transition: outline-color 0.12s; }
            .ui-customizable.ui-customizable-active { outline-color: var(--primary); }
            .ui-drag-handle { position: absolute; top: 6px; right: 6px; background: var(--primary); color: var(--on-primary); padding: 4px 6px; border-radius: 6px; cursor: move; z-index: 9100; }
            .snapline-vertical { pointer-events: none; }
            .snapline-horizontal { pointer-events: none; }
            .customization-overlay { pointer-events: none; }
        `;
        document.head.appendChild(style);
    }

    /**
     * Save layout configuration to localStorage
     */
    saveLayout() {
        const layout = { items: [] };
        document.querySelectorAll('.ui-customizable').forEach((el) => {
            const id = el.id || el.dataset.customId || this._ensureId(el);
            layout.items.push({ id, left: el.style.left, top: el.style.top, width: el.style.width, height: el.style.height });
        });
        localStorage.setItem('uiLayout', JSON.stringify(layout));
        // alert('Layout saved!');
    }

    /**
     * Ensure an element has a stable id used for layout persistence
     */
    _ensureId(el) {
        if (!el.id) {
            el.id = 'custom-' + Math.random().toString(36).slice(2, 9);
        }
        return el.id;
    }

    /**
     * Load layout
     */
    loadLayout() {
        const saved = localStorage.getItem('uiLayout');
        return saved ? JSON.parse(saved) : { items: [] };
    }

    /**
     * Apply saved layout by positioning elements
     */
    applyLayout() {
        if (!this.layoutConfig || !this.layoutConfig.items) return;
        this.layoutConfig.items = this.layoutConfig.items || [];

        this.layoutConfig.items.forEach(item => {
            const el = document.getElementById(item.id);
            if (el) {
                // Avoid applying layout to high-level containers that contain other customized items
                // This prevents a saved container position from moving all children at once.
                if (el.matches && (el.matches('.tab-panel') || el.matches('.tab-content') || el.matches('.dashboard-grid'))) {
                    // if it contains child elements like cards or form-groups, skip applying its saved position
                    if (el.querySelector('.card, .form-group, .dashboard-grid > *')) {
                        return;
                    }
                }

                // Only apply saved layout if coordinates or size are present. Otherwise skip so we don't
                // force elements (especially containers) into absolute mode and cause jumps.
                if (!(item.left || item.top || item.width || item.height)) {
                    return;
                }

                // ensure container is positioned so absolute coordinates are relative
                // Support both .tab-content and .tab-panel panels
                const container = el.closest('.tab-content, .tab-panel') || el.offsetParent || document.body;
                const computed = window.getComputedStyle(container);
                if (computed.position === 'static') {
                    container.style.position = 'relative';
                }

                // store original inline styles for reset (if not already stored)
                if (el.dataset._origPosition === undefined) el.dataset._origPosition = el.style.position || '';
                if (el.dataset._origLeft === undefined) el.dataset._origLeft = el.style.left || '';
                if (el.dataset._origTop === undefined) el.dataset._origTop = el.style.top || '';
                if (el.dataset._origWidth === undefined) el.dataset._origWidth = el.style.width || '';
                if (el.dataset._origHeight === undefined) el.dataset._origHeight = el.style.height || '';

                // Apply absolute positioning and saved sizes/positions without enabling interact handles
                el.style.position = 'absolute';
                if (item.left) el.style.left = item.left;
                if (item.top) el.style.top = item.top;
                if (item.width) el.style.width = item.width;
                if (item.height) el.style.height = item.height;
            }
        });
    }

    /**
     * Reset layout to default
     */
    resetLayout() {
        if (!confirm('Reset layout to default? This will clear all customizations.')) return;
        localStorage.removeItem('uiLayout');
        // restore original inline styles where available (look for any element that has stored original values)
        document.querySelectorAll('[data-_orig-position], [data-_orig-left], [data-_orig-top], [data-_orig-width], [data-_orig-height]').forEach(el => {
            if (el.dataset._origPosition) el.style.position = el.dataset._origPosition;
            el.style.left = el.dataset._origLeft || '';
            el.style.top = el.dataset._origTop || '';
            el.style.width = el.dataset._origWidth || '';
            el.style.height = el.dataset._origHeight || '';
            // cleanup helper dataset attributes
            delete el.dataset._origPosition;
            delete el.dataset._origLeft;
            delete el.dataset._origTop;
            delete el.dataset._origWidth;
            delete el.dataset._origHeight;
        });
        alert('Layout reset to default!');
        window.location.reload();
    }

    /**
     * Returns true when the given element contains any already-selected customizable elements
     */
    hasCustomizableChild(element, customizableElements) {
        return Array.from(customizableElements).some(el => element.contains(el));
    }

    /**
     * Get elements that can be customized
     */
    getCustomizableElements() {
        // Prefer specific components first (cards, form-groups) so we don't accidentally
        // select a broad container (tab-panel) and block movement of inner items.
        const selectors = [
            '.card',
            '.form-group',
            '.dashboard-grid > *',
            '.chart-placeholder',
            '.data-table',
            '.notification-list',
            '.timer-display',
            '.stopwatch-display',
            '.tab-content > *:not(h1):not(h2):not(h3)',
            '.tab-panel > *:not(h1):not(h2):not(h3)'
        ];

        const elements = [];
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                // Skip elements that live inside modals (settings or other dialogs)
                if (el.closest('.modal')) return;

                // Skip if a parent is already selected (we don't want to duplicate) or
                // if the element contains a previously selected child (prefer child specificity)
                if (!elements.includes(el) && !this.hasCustomizableParent(el, elements) && !this.hasCustomizableChild(el, elements)) {
                    elements.push(el);
                }
            });
        });

        return elements;
    }
    hasCustomizableParent(element, customizableElements) {
        let parent = element.parentElement;
        while (parent) {
            if (customizableElements.includes(parent)) return true;
            parent = parent.parentElement;
        }
        return false;
    }

    /**
     * Disable customization: destroy interact instances and cleanup overlays
     */
    disableCustomization() {
        // Auto-save current layout before removing helper UI so it persists
        this.saveLayout();

        // destroy interact instances and keep the modified inline styles so layout persists
        document.querySelectorAll('.ui-customizable').forEach(el => {
            try {
                if (window.interact) window.interact(el).unset();
            } catch(e) { /* ignore */ }

            // remove handles and helpers
            const dragHandle = el.querySelector('.ui-drag-handle'); if (dragHandle) dragHandle.remove();

            // keep original values in dataset for reset, but do NOT restore them here so customized layout remains
            // (this allows auto-save of current inline styles)

            el.classList.remove('ui-customizable');
            delete el.dataset.customizable;
            el.classList.remove('ui-customizable-active');
        });

        // remove overlays but keep container positioning (relative) so absolute children remain positioned correctly
        this.containers.forEach((info, container) => {
            if (info.overlay && info.overlay.parentNode) info.overlay.parentNode.removeChild(info.overlay);
            // do NOT revert container.style.position here
        });
        this.containers.clear();

        // disconnect any pending observers and clear pending queues
        this._containerObservers.forEach((obs, container) => {
            try { obs.disconnect(); } catch (e) {}
        });
        this._containerObservers.clear();
        if (this._pending) this._pending.clear();
    }

    /**
     * Show hints overlay while in customization mode
     */
    showCustomizationHints() {
        const hint = document.createElement('div');
        hint.id = 'customization-hint';
        hint.innerHTML = `
            <div style="position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background: var(--primary); color: var(--on-primary); padding: 1rem 2rem; border-radius: var(--border-radius); box-shadow: 0 4px 8px var(--shadow); z-index: 3000; text-align: center;">
                <strong>Customization Mode Active</strong><br>
                <small>Drag elements to rearrange • Resize from edges • Grid snap & alignment guides active</small>
            </div>
        `;
        document.body.appendChild(hint);
    }

    hideCustomizationHints() {
        const hint = document.getElementById('customization-hint'); if (hint) hint.remove();
    }
}

// Auto-initialize when UIFramework is available
if (typeof UIFramework !== 'undefined') {
    const originalInit = UIFramework.prototype.setup;
    UIFramework.prototype.setup = function() {
        originalInit.call(this);
        this.customizer = new UICustomizer(this);
        this.customizer.applyLayout();
    };
}

// Export for use in modules

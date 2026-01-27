# UI Framework Guide

## Best Practices & Usage Instructions

This document provides comprehensive guidance on using the UI Framework to build consistent, beautiful, and accessible web applications.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Design Systems](#design-systems)
4. [Core Components](#core-components)
5. [Theming & Customization](#theming--customization)
6. [Layout Patterns](#layout-patterns)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Accessibility](#accessibility)
10. [Performance](#performance)
11. [UI Customization](#ui-customization)

---

## Introduction

This UI framework is a lightweight, reusable design system that supports three distinct design philosophies:

- **Material 3**: Google's modern design language with smooth animations and elevation
- **Fluent Design**: Microsoft's clean, efficient design with minimal transitions
- **Liquid Glass**: A modern glassmorphism aesthetic with floating elements and backdrop blur

The framework is built with vanilla HTML, CSS, and JavaScript for maximum compatibility and minimal dependencies.

---

## Getting Started

### Basic Setup

1. **Create your HTML file** with the following structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your App Name</title>
    <link rel="stylesheet" href="path/to/framework.css">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
</head>
<body class="liquid-glass" data-theme="light">
    <!-- Your content here -->
    
    <script src="path/to/framework.js"></script>
    <script>
        const app = new UIFramework({
            appName: 'Your App Name',
            defaultTheme: 'light',
            defaultStyle: 'liquid-glass'
        });
    </script>
</body>
</html>
```

2. **Initialize the framework** in your JavaScript:

```javascript
const app = new UIFramework({
    appName: 'My Application',
    defaultTheme: 'light',
    defaultStyle: 'liquid-glass',
    onTabChange: (tabName) => {
        console.log('Switched to:', tabName);
    }
});
```

---

## Design Systems

### Material 3

**When to use**: Modern applications that need smooth, delightful animations and clear visual hierarchy.

**Characteristics**:
- Rounded buttons (20px radius)
- Smooth elevation changes on hover
- Fast animations (0.15s)
- Clear shadows for depth

**CSS Class**: `material3`

```html
<body class="material3" data-theme="light">
```

### Fluent Design

**When to use**: Professional applications, enterprise tools, or when you need maximum performance.

**Characteristics**:
- Sharp corners (2-4px radius)
- No animations (instant feedback)
- Clean, minimal aesthetic
- High information density

**CSS Class**: `fluent`

```html
<body class="fluent" data-theme="light">
```

### Liquid Glass

**When to use**: Modern, premium applications where aesthetics are important.

**Characteristics**:
- Rounded, pill-shaped elements (50px+ radius)
- Backdrop blur effects
- Semi-transparent surfaces
- Floating navigation elements
- Smooth, fluid animations (0.4-0.6s)

**CSS Class**: `liquid-glass`

```html
<body class="liquid-glass" data-theme="light">
```

---

## Core Components

### Header

The header contains your app title and main controls.

```html
<div class="header">
    <div class="title">App Name</div>
    <div class="controls">
        <button id="settingsBtn">⚙️ Settings</button>
    </div>
</div>
```

**Best Practices**:
- Keep the title short and descriptive
- Limit controls to 2-3 primary actions
- Use icons with text for clarity

### Tab Navigation

Tabs organize your app into logical sections.

```html
<div class="tabs">
    <button class="tab-link active" onclick="openTab(event, 'home')">
        <i class="material-icons">home</i> Home
    </button>
    <button class="tab-link" onclick="openTab(event, 'explore')">
        <i class="material-icons">explore</i> Explore
    </button>
</div>
```

**Best Practices**:
- Use 3-5 tabs maximum for optimal usability
- Always include icons with labels
- Mark the current tab with the `active` class
- Use descriptive, single-word labels when possible

### Tab Content

Each tab has corresponding content:

```html
<div id="home" class="tab-content" style="display: block;">
    <h2>Home</h2>
    <!-- Your content -->
</div>

<div id="explore" class="tab-content">
    <h2>Explore</h2>
    <!-- Your content -->
</div>
```

**Best Practices**:
- Only show one tab at a time
- Use semantic headings (h2, h3) for structure
- Keep content focused and organized

### Buttons

Buttons are the primary interaction element:

```html
<button>Default Button</button>
<button disabled>Disabled Button</button>
```

**Best Practices**:
- Use clear, action-oriented labels ("Save", "Delete", not "OK")
- Disable buttons when actions are unavailable
- Use icons to enhance recognition
- Group related buttons together

### Modals

Modals present focused tasks or information:

```html
<div class="modal" id="myModal">
    <div class="modal-content">
        <div class="modal-header">
            <div class="modal-title">Modal Title</div>
            <button class="close-btn">&times;</button>
        </div>
        <div class="form-group">
            <!-- Modal content -->
        </div>
        <button>Action Button</button>
    </div>
</div>
```

**Best Practices**:
- Use modals for important, focused tasks
- Always provide a clear way to close
- Keep modal content concise
- Use for confirmations, settings, or data entry

### Form Elements

Forms collect user input:

```html
<div class="form-group">
    <label for="myInput">Input Label:</label>
    <input type="text" id="myInput" placeholder="Enter value">
</div>

<div class="form-group">
    <label for="mySelect">Select Option:</label>
    <select id="mySelect">
        <option value="1">Option 1</option>
        <option value="2">Option 2</option>
    </select>
</div>

<div class="form-group toggle-group">
    <label for="myToggle">Enable Feature:</label>
    <label class="switch">
        <input type="checkbox" id="myToggle">
        <span class="slider"></span>
    </label>
</div>
```

**Best Practices**:
- Always use labels with form inputs
- Provide helpful placeholder text
- Use appropriate input types (text, number, date, etc.)
- Group related form fields together
- Provide validation feedback

---

## Theming & Customization

### Themes

The framework supports three built-in themes:

1. **Light Theme** (`data-theme="light"`)
   - White/light backgrounds
   - Dark text
   - Default for most applications

2. **Dark Theme** (`data-theme="dark"`)
   - Dark backgrounds (#121212)
   - Light text
   - Reduced eye strain in low-light

3. **High Contrast** (`data-theme="high-contrast"`)
   - Maximum contrast (black/white)
   - Accessibility focused
   - WCAG AAA compliant

### Accent Colors

Customize the primary color:

```javascript
app.setAccentColor('#1976d2');
```

Or let users choose:

```html
<input type="color" id="accentColorInput" value="#1976d2">
```

### CSS Variables

All colors are controlled via CSS variables:

```css
:root {
    --primary: #1976d2;
    --primary-variant: #1565c0;
    --secondary: #03dac6;
    --background: #ffffff;
    --surface: #f5f5f5;
    --error: #b00020;
    --success: #4caf50;
    --warning: #ff9800;
}
```

**Best Practices**:
- Use CSS variables for consistent theming
- Test your app in all three themes
- Ensure sufficient color contrast (WCAG AA minimum)
- Consider color blindness when choosing colors

---

## Layout Patterns

### Grid Layout

Use CSS Grid for card-based layouts:

```html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
    <div class="card">Card 1</div>
    <div class="card">Card 2</div>
    <div class="card">Card 3</div>
</div>
```

### Flex Layout

Use Flexbox for linear layouts:

```html
<div style="display: flex; gap: 1rem; align-items: center;">
    <button>Action 1</button>
    <button>Action 2</button>
    <button>Action 3</button>
</div>
```

### Responsive Design

The framework is mobile-first:

```css
/* Mobile: Full width */
.card { width: 100%; }

/* Tablet: 2 columns */
@media (min-width: 768px) {
    .card { width: calc(50% - 0.5rem); }
}

/* Desktop: 3+ columns */
@media (min-width: 1024px) {
    .card { width: calc(33.333% - 0.667rem); }
}
```

---

## Best Practices

### 1. Structure Your App

```
your-app/
├── index.html          # Main HTML file
├── app.js             # Your app logic
├── styles.css         # Custom styles (extends framework)
├── framework/
│   ├── framework.css
│   └── framework.js
└── assets/
    ├── images/
    └── icons/
```

### 2. Semantic HTML

Use appropriate HTML elements:

```html
<!-- Good -->
<button onclick="save()">Save</button>
<nav class="tabs">...</nav>
<main class="tab-content">...</main>

<!-- Avoid -->
<div onclick="save()">Save</div>
<div class="tabs">...</div>
```

### 3. Progressive Enhancement

Start with functional HTML, then enhance with CSS and JavaScript:

1. HTML structure works without CSS
2. CSS makes it beautiful
3. JavaScript adds interactivity

### 4. Accessibility First

- Use semantic HTML
- Provide text alternatives for icons
- Ensure keyboard navigation works
- Test with screen readers
- Maintain color contrast ratios

### 5. Performance

- Load framework files once (from CDN or local)
- Minimize custom CSS
- Use CSS variables instead of inline styles
- Lazy load heavy content
- Debounce expensive operations

### 6. State Management

Store user preferences:

```javascript
// Save
localStorage.setItem('userPreference', value);

// Load
const preference = localStorage.getItem('userPreference');

// Remove
localStorage.removeItem('userPreference');
```

### 7. Error Handling

Always handle errors gracefully:

```javascript
try {
    // Your code
} catch (error) {
    console.error('Error:', error);
    alert('Something went wrong. Please try again.');
}
```

---

## Examples

### Simple Counter App

```html
<div class="tab-content">
    <h2>Counter</h2>
    <div style="text-align: center; font-size: 3rem; margin: 2rem;">
        <span id="counter">0</span>
    </div>
    <div style="display: flex; gap: 1rem; justify-content: center;">
        <button onclick="decrement()">-</button>
        <button onclick="reset()">Reset</button>
        <button onclick="increment()">+</button>
    </div>
</div>

<script>
let count = 0;
function increment() {
    count++;
    document.getElementById('counter').textContent = count;
}
function decrement() {
    count--;
    document.getElementById('counter').textContent = count;
}
function reset() {
    count = 0;
    document.getElementById('counter').textContent = count;
}
</script>
```

### Data List with Search

```html
<div class="form-group">
    <input type="text" id="search" placeholder="Search..." oninput="filterList()">
</div>
<ul id="itemList">
    <li>Item 1</li>
    <li>Item 2</li>
    <li>Item 3</li>
</ul>

<script>
function filterList() {
    const query = document.getElementById('search').value.toLowerCase();
    const items = document.querySelectorAll('#itemList li');
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(query) ? '' : 'none';
    });
}
</script>
```

---

## Accessibility

### Keyboard Navigation

- **Tab**: Navigate through interactive elements
- **Enter/Space**: Activate buttons
- **Escape**: Close modals
- **Arrow Keys**: Navigate within components

### Screen Reader Support

Use ARIA labels and roles:

```html
<button aria-label="Close settings">×</button>
<div role="dialog" aria-labelledby="modalTitle">
    <h2 id="modalTitle">Settings</h2>
</div>
```

### Color Contrast

- Normal text: 4.5:1 minimum (WCAG AA)
- Large text: 3:1 minimum
- Use high contrast theme for maximum accessibility

### Focus Indicators

Ensure visible focus states:

```css
button:focus {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}
```

---

## Performance

### Loading Performance

1. **Minimize HTTP requests**: Bundle CSS and JS
2. **Use CDNs**: For external libraries like Material Icons
3. **Lazy load**: Heavy content and images
4. **Cache**: Set appropriate cache headers

### Runtime Performance

1. **Debounce**: Expensive operations like search
2. **Throttle**: Scroll and resize handlers
3. **Virtual scrolling**: For large lists
4. **Use CSS**: For animations instead of JavaScript

### Memory Management

1. **Remove event listeners**: When elements are destroyed
2. **Clear intervals**: When components unmount
3. **Limit localStorage**: Keep under 5MB per domain

---

## Common Patterns

### Loading States

```html
<button id="submitBtn" onclick="submitForm()">
    Submit
</button>

<script>
async function submitForm() {
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.textContent = 'Loading...';
    
    try {
        await performAction();
        btn.textContent = 'Success!';
    } catch (error) {
        btn.textContent = 'Error!';
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = 'Submit';
        }, 2000);
    }
}
</script>
```

### Confirmation Dialogs

```javascript
function deleteItem(id) {
    if (confirm('Are you sure you want to delete this item?')) {
        // Perform deletion
        console.log('Deleted item:', id);
    }
}
```

### Toast Notifications

```javascript
function showToast(message) {
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--surface);
        padding: 1rem 2rem;
        border-radius: var(--border-radius);
        box-shadow: 0 4px 8px var(--shadow);
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
```

---

## Troubleshooting

### Modal Won't Open

**Problem**: Settings modal doesn't show when button is clicked.

**Solution**: Ensure you've initialized the framework:
```javascript
const app = new UIFramework({ /* config */ });
```

### Tabs Not Switching

**Problem**: Clicking tabs doesn't change content.

**Solution**: Ensure `openTab` function is called correctly:
```html
<button class="tab-link" onclick="openTab(event, 'tabName')">
```

### Liquid Glass Style Not Working

**Problem**: Blur effects don't appear in Liquid Glass mode.

**Solution**: Some browsers don't support `backdrop-filter`. Use fallbacks:
```css
background: rgba(255, 255, 255, 0.1);
backdrop-filter: blur(8px);
/* Fallback for unsupported browsers */
@supports not (backdrop-filter: blur(8px)) {
    background: rgba(255, 255, 255, 0.9);
}
```

### Colors Not Updating

**Problem**: Accent color changes don't apply.

**Solution**: Call `setAccentColor()` after changing:
```javascript
app.setAccentColor('#1976d2');
```

---

## Migration Guide

### From Bootstrap

Replace Bootstrap classes with framework equivalents:

```html
<!-- Bootstrap -->
<button class="btn btn-primary">Click</button>

<!-- This Framework -->
<button>Click</button>
```

### From Tailwind

Use semantic HTML and CSS classes:

```html
<!-- Tailwind -->
<div class="flex gap-4 p-4 bg-white rounded-lg shadow">

<!-- This Framework -->
<div style="display: flex; gap: 1rem; padding: 1rem; 
     background-color: var(--surface); 
     border-radius: var(--border-radius);
     box-shadow: 0 2px 4px var(--shadow);">
```

---

## Contributing

To extend the framework:

1. **Add new CSS variables** in `:root`
2. **Create reusable classes** that use variables
3. **Test in all three design systems**
4. **Document your additions** in this guide
5. **Ensure accessibility compliance**

---

## UI Customization

### Drag-and-Drop Layout Customization

The framework includes a powerful customization system that allows users to rearrange and resize UI elements.

#### Enabling Customization

1. **Include the customizer module** in your HTML:
```html
<script src="framework.js"></script>
<script src="customizer.js"></script>
```

2. **Access via Settings**: Users can toggle customization mode in Settings > Customize Layout

#### Features

**Drag to Reorder**
- Click and drag any customizable element
- Drop it in a new position to rearrange
- Works across different containers

**Resize Elements**
- Use the resize handle (⋰) in the bottom-right corner
- Drag to adjust width and height
- Minimum sizes enforced for usability

**Layout Persistence**
- Custom layouts are automatically saved to localStorage
- Restored on page load
- Persists across sessions

**Reset to Default**
- Users can reset to the original layout anytime
- Clears all customizations
- Reloads the page with defaults

#### Customizable Elements

The following elements are automatically customizable:
- Cards (`.card`)
- Form groups (`.form-group`)
- Chart placeholders (`.chart-placeholder`)
- Data tables (`.data-table`)
- Notification lists (`.notification-list`)
- Dashboard grid items
- Timer/stopwatch displays

#### Usage Example

```javascript
// Initialize framework with customization
const app = new UIFramework({
    appName: 'My App',
    defaultTheme: 'light',
    defaultStyle: 'liquid-glass'
});

// Customizer is automatically initialized
// Access it via: app.customizer
```

#### Customization API

```javascript
// Enable customization mode programmatically
app.customizer.toggleCustomizationMode(true);

// Disable customization mode
app.customizer.toggleCustomizationMode(false);

// Layouts auto-save when moving/resizing elements; manual save is not required
// (a manual save API remains available if needed)
// Programmatic manual save (optional):
app.customizer.saveLayout();

// Load saved layout (applies stored positions on load)
app.customizer.applyLayout();

// Reset to default (clears saved layout)
app.customizer.resetLayout();
```

#### Best Practices

1. **Test with customization**: Ensure your app works well with rearranged elements
2. **Provide defaults**: Always have a sensible default layout
3. **Consider mobile**: Customization works best on desktop/tablet
4. **Document for users**: Tell users about the customization feature
5. **Backup layouts**: Consider exporting/importing custom layouts

#### Styling Customizable Elements

When customization mode is active, elements receive special styling:

```css
.ui-customizable {
    position: relative;
    cursor: move;
    border: 2px dashed transparent;
}

.ui-customizable:hover {
    border-color: var(--primary);
}
```

You can override these styles in your custom CSS if needed.

---

## Resources

- [Material Design Guidelines](https://m3.material.io/)
- [Fluent Design System](https://www.microsoft.com/design/fluent/)
- [Web Content Accessibility Guidelines (WCAG)](https://www.w3.org/WAI/WCAG21/quickref/)
- [MDN Web Docs](https://developer.mozilla.org/)

---

## License

This framework is part of the DMHS AI Time Tracker project and follows the same license.

---

## Summary

This UI framework provides:

✅ Three beautiful design systems (Material 3, Fluent, Liquid Glass)  
✅ Light, Dark, and High Contrast themes  
✅ Responsive, mobile-first layouts  
✅ Accessible by default  
✅ Zero dependencies (except Material Icons)  
✅ Easy to learn and use  
✅ Highly customizable  
✅ Performance optimized  

**Happy building! 🚀**

# UI Customization Quick Reference

## Overview

The UI Framework includes a powerful drag-and-drop customization system that allows users to personalize their layout by rearranging and resizing elements.

## Enabling Customization

### For Developers

1. Include both framework files:
```html
<script src="framework.js"></script>
<script src="customizer.js"></script>
```

2. Initialize the framework:
```javascript
const app = new UIFramework({
    appName: 'My App',
    defaultStyle: 'liquid-glass'
});
// Customizer is automatically available
```

### For End Users

1. Click the **Settings** button (⚙️)
2. Toggle **Customize Layout** switch
3. Click outside or close settings to start customizing

## Using Customization Mode

### Rearranging Elements

1. **Drag Handle**: Look for the handle icon (⋮⋮) in the top-right corner
2. **Click and Drag**: Click anywhere on a customizable element
3. **Drop**: Release to place in new position
4. **Visual Feedback**: Elements show a dashed border on hover

### Resizing Elements

1. **Resize Handle**: Look for the resize icon (⋰) in the bottom-right corner
2. **Click and Drag**: Click the resize handle and drag
3. **Minimum Size**: Elements enforce minimum dimensions for usability

### Saving & Resetting

- **Auto-Save**: Layout saves automatically when you disable customization mode
- **Persistent**: Layout is saved to browser localStorage
- **Reset**: Go to Settings > Reset Layout to Default

## Visual Indicators

When customization mode is active:

- **Notification Banner**: Shows at the top with instructions
- **Dashed Borders**: Appear when hovering over elements
- **Drag Handle (⋮⋮)**: Top-right corner of customizable elements
- **Resize Handle (⋰)**: Bottom-right corner of customizable elements

## Customizable Elements

The following elements are automatically customizable:

- Dashboard cards (`.card`)
- Form sections (`.form-group`)
- Charts and visualizations (`.chart-placeholder`)
- Data tables (`.data-table`)
- Notification panels (`.notification-list`)
- Timer displays
- Grid items in dashboard layouts

## Tips & Best Practices

### For Developers

1. **Test thoroughly**: Ensure app functionality works regardless of element order
2. **Provide good defaults**: Start with a sensible layout
3. **Consider mobile**: Customization works best on larger screens
4. **Document the feature**: Tell users about customization capabilities

### For End Users

1. **Experiment freely**: You can always reset to defaults
2. **Desktop recommended**: Customization is easier on desktop/tablet
3. **Disable when done**: Turn off customization mode to prevent accidental changes
4. **Refresh if needed**: Reload the page to see your saved layout

## Programmatic API

### Toggle Customization Mode

```javascript
// Enable customization
app.customizer.toggleCustomizationMode(true);

// Disable customization
app.customizer.toggleCustomizationMode(false);
```

### Layout Management

```javascript
// Layouts auto-save on change (no manual save required).
// Manual save API still available if needed:
app.customizer.saveLayout();

// Apply saved layout
app.customizer.applyLayout();

// Reset to default
app.customizer.resetLayout();
```

### Check Customization State

```javascript
// Check if customization is active
if (app.customizer.customizationMode) {
    console.log('Customization is active');
}
```

## Keyboard Shortcuts

Currently, customization is mouse/touch-driven. Future versions may include:
- Arrow keys for precise positioning
- Ctrl+Z for undo
- Keyboard-accessible resize controls

## Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (touch support)

**Note**: Requires localStorage support for persistence.

## Troubleshooting

### Layout Not Saving

**Problem**: Custom layout doesn't persist after reload.

**Solution**: 
- Check if localStorage is enabled
- Ensure you disabled customization mode (this triggers save)
- Check browser console for errors

### Elements Not Draggable

**Problem**: Can't drag elements.

**Solution**:
- Ensure customization mode is enabled
- Look for the drag handle (⋮⋮)
- Try refreshing the page
- Check that elements have the `.ui-customizable` class

### Resize Not Working

**Problem**: Resize handle doesn't appear or work.

**Solution**:
- Enable customization mode
- Look for resize handle (⋰) in bottom-right
- Ensure element has explicit dimensions
- Try clicking directly on the handle

### Reset Not Working

**Problem**: Reset button doesn't restore default layout.

**Solution**:
- Click "Reset Layout to Default" button
- Confirm the dialog
- Page will reload with defaults
- If persists, clear localStorage manually

## Security & Privacy

- Layouts are stored **locally only** (localStorage)
- No data is sent to external servers
- Clearing browser data will reset layouts
- Safe to use in privacy-focused environments

## Performance Considerations

- Minimal overhead when disabled
- Efficient DOM manipulation
- No impact on page load time
- Small file size (15KB)

## Future Enhancements

Planned features for future versions:

- Export/import layouts
- Preset layout templates
- Grid snapping
- Undo/redo functionality
- Keyboard navigation
- Layout sharing (with user consent)
- Multi-device sync (optional)

## Examples

### Basic Dashboard Customization

1. Open the dashboard example
2. Enable customization mode
3. Rearrange cards to prioritize important metrics
4. Resize charts for better visibility
5. Save and use your personalized layout

### Workflow Optimization

1. Identify your most-used features
2. Move them to prominent positions
3. Resize less important elements smaller
4. Create a workflow-optimized layout
5. Enjoy improved productivity!

## Support

For issues or questions:
- Check the FRAMEWORK-GUIDE.md for detailed documentation
- Review examples in `/examples/` directory
- Test in different browsers
- Clear cache and localStorage if problems persist

---

**Happy Customizing! 🎨**

# Extension Development Guide for Project Vortex

## Overview

The Vortex Extension System allows you to add custom functionality to all Vortex Desktop apps: **PlotVisual**, **HexaKinetic**, **HexaVisual**, and **Mission Control**.

Extensions can add custom graphs, data sources, fault models, 3D overlays, dashboard widgets, and more — all through a unified framework with per-app hooks.

---

## Quick Start

### 1. Create your extension directory

```
my_extension/
  manifest.json     <- Required metadata
  plugin.py         <- Your extension code (entry_point)
```

### 2. Write the manifest

```json
{
  "id": "my_extension",
  "name": "My Custom Extension",
  "version": "1.0.0",
  "extension_type": "plotvisual",
  "entry_point": "plugin",
  "author": "Your Name",
  "description": "A short description of what this extension does",
  "tags": ["graphs", "analysis"],
  "dependencies": []
}
```

### 3. Write the extension class

```python
# plugin.py
from extensions.base import VortexExtension

class MyExtension(VortexExtension):
    @property
    def name(self):
        return "My Custom Extension"

    @property
    def version(self):
        return "1.0.0"

    @property
    def extension_type(self):
        return "plotvisual"

    def activate(self, app_context):
        print(f"[{self.name}] Activated!")

    def deactivate(self):
        print(f"[{self.name}] Deactivated")
```

### 4. Install it

- **From PlotVisual**: Extensions page → Develop tab → "Install from Directory"
- **From any app**: Extensions menu → "Install from File..."
- **Manual**: Copy your folder to `extensions/installed/my_extension/`
- **Package**: Use `extensions.packaging.pack_extension()` to create a `.vortexext` file

---

## Extension Types

| Type | Target App | Hook Base Class | Key Capabilities |
|------|-----------|----------------|-----------------|
| `plotvisual` | PlotVisual | `PlotVisualExtension` | Custom graphs, data sources, export formats |
| `hexakinetic` | HexaKinetic | `HexaKineticExtension` | Fault models, motor profiles, controllers, atmosphere models |
| `hexavisual` | HexaVisual | `HexaVisualExtension` | 3D overlays, camera modes, render effects, HUD widgets |
| `mission_control` | Mission Control | `MissionControlExtension` | Dashboard widgets, serial protocols, commands |
| `universal` | All apps | `VortexExtension` | Works everywhere (use for utilities/data) |

---

## Architecture

```
extensions/
  __init__.py          # Package init
  base.py              # VortexExtension ABC (base for all extensions)
  manifest.py          # ExtensionManifest dataclass + validation
  manager.py           # ExtensionManager - lifecycle, install, uninstall, enable/disable
  registry.py          # ExtensionRegistry - per-app dispatch + LegacyPluginAdapter
  packaging.py         # .vortexext archive creation/extraction
  catalog.py           # ExtensionCatalog - bundled catalog of available extensions
  catalog.json         # Catalog data
  state.json           # Persisted enable/disable state
  installed/           # User-installed extensions go here
  hooks/
    plotvisual.py      # PlotVisualExtension + GraphDefinition
    hexakinetic.py     # HexaKineticExtension hooks
    hexavisual.py      # HexaVisualExtension hooks
    mission_control.py # MissionControlExtension hooks
```

### Lifecycle

1. **Discovery**: `ExtensionManager.discover()` scans `extensions/installed/` and legacy `plugins/` directories
2. **Registration**: `ExtensionRegistry` filters extensions by app type
3. **Activation**: `registry.activate_all(app_context)` calls `ext.activate(app_context)` on each
4. **Runtime**: App calls extension hooks as needed (render graphs, inject faults, etc.)
5. **Deactivation**: `registry.deactivate_all()` on app close

### State Persistence

Enable/disable state is stored in `extensions/state.json`:
```json
{
  "enabled": {
    "claude_graphs": true,
    "my_extension": false
  }
}
```

---

## PlotVisual Extensions (Detailed)

PlotVisual extensions can register custom graphs that appear in the Graphs sidebar.

### Hook: `PlotVisualExtension`

```python
from extensions.hooks.plotvisual import PlotVisualExtension, GraphDefinition

class MyGraphs(PlotVisualExtension):
    @property
    def name(self):
        return "My Graphs Pack"

    @property
    def version(self):
        return "1.0.0"

    def activate(self, app_context):
        self._app = app_context

    def deactivate(self):
        self._app = None

    def register_graphs(self):
        """Return a list of GraphDefinition objects."""
        return [
            GraphDefinition(
                key='my_scatter',
                label='My Scatter Plot',
                description='Custom scatter plot with extra analysis',
                icon='📊',
                category='Custom'
            ),
            GraphDefinition(
                key='my_histogram',
                label='My Histogram',
                description='Custom histogram with stats overlay',
                icon='📈',
                category='Custom'
            ),
        ]

    def render_graph(self, graph_key, data_store, figure, **kwargs):
        """Render a specific graph into the provided matplotlib Figure."""
        ax = figure.add_subplot(111)

        if graph_key == 'my_scatter':
            data = data_store.get('legacy')
            if data is not None and 'Landing Velocity' in data.columns:
                ax.scatter(data['Total Fault Intensity'], data['Landing Velocity'],
                          c='#00bcd4', alpha=0.7, s=40)
                ax.set_xlabel('Fault Intensity')
                ax.set_ylabel('Landing Velocity (m/s)')
                ax.set_title('My Custom Scatter Plot')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)

        elif graph_key == 'my_histogram':
            data = data_store.get('legacy')
            if data is not None and 'Landing Distance' in data.columns:
                ax.hist(data['Landing Distance'], bins=30, color='#ff9800', alpha=0.8)
                ax.set_xlabel('Landing Distance (m)')
                ax.set_ylabel('Count')
                ax.set_title('My Custom Histogram')

    def generate_sample_data(self, data_store):
        """Optional: Generate sample data for preview."""
        pass
```

### GraphDefinition Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | str | Yes | Unique identifier for this graph |
| `label` | str | Yes | Display name in the sidebar |
| `description` | str | No | Tooltip text |
| `icon` | str | No | Emoji icon (default: 🧩) |
| `category` | str | No | Group heading in sidebar (default: 'Extensions') |

### Data Store

Extensions receive a `VortexDataStore` instance with keyed datasets:

```python
data_store.get('legacy')         # Main DataFrame
data_store.get('trajectory')     # Trajectory data
data_store.get('optimization')   # Optimization results
data_store.has('legacy')         # Check if data exists
data_store.set('my_data', df)    # Store custom data
```

---

## HexaKinetic Extensions (Detailed)

HexaKinetic extensions can add custom fault models, motor profiles, controllers, and atmosphere models.

### Hook: `HexaKineticExtension`

```python
from extensions.hooks.hexakinetic import HexaKineticExtension

class MyFaults(HexaKineticExtension):
    @property
    def name(self):
        return "Advanced Fault Models"

    @property
    def version(self):
        return "1.0.0"

    def activate(self, app_context):
        self._app = app_context

    def deactivate(self):
        self._app = None

    def register_fault_models(self):
        """Return dict of {id: FaultModelDefinition}."""
        return {
            'sensor_drift': {
                'name': 'Sensor Drift',
                'description': 'Gradual IMU sensor drift over time',
                'parameters': {
                    'drift_rate': {'type': 'float', 'default': 0.01, 'min': 0, 'max': 1},
                    'axis': {'type': 'choice', 'options': ['pitch', 'yaw', 'roll'], 'default': 'pitch'},
                },
            }
        }

    def apply_fault(self, fault_id, state_vector, params, sim_time):
        """Apply a fault to the state vector and return the modified state."""
        if fault_id == 'sensor_drift':
            # Modify the state vector based on drift
            drift = params.get('drift_rate', 0.01) * sim_time
            # ... apply drift logic ...
        return state_vector

    def register_motor_profiles(self):
        """Return dict of custom motor profiles."""
        return {}

    def register_controllers(self):
        """Return dict of custom controller implementations."""
        return {}

    def register_atmosphere_models(self):
        """Return dict of custom atmosphere models."""
        return {}

    def pre_simulation_hook(self, config):
        """Called before simulation starts. Can modify config."""
        pass

    def post_simulation_hook(self, results):
        """Called after simulation completes. Can post-process results."""
        pass
```

---

## HexaVisual Extensions (Detailed)

HexaVisual extensions can add 3D overlays, custom camera modes, and render effects.

### Hook: `HexaVisualExtension`

```python
from extensions.hooks.hexavisual import HexaVisualExtension

class MyOverlays(HexaVisualExtension):
    @property
    def name(self):
        return "Trajectory Overlays"

    @property
    def version(self):
        return "1.0.0"

    def activate(self, app_context):
        self._app = app_context

    def deactivate(self):
        self._app = None

    def register_overlays(self):
        """Return dict of 3D overlay definitions."""
        return {
            'velocity_vectors': {
                'name': 'Velocity Vectors',
                'description': 'Show velocity arrows along trajectory',
            }
        }

    def render_overlay(self, overlay_id, plotter, data, frame_idx):
        """Render a 3D overlay into the pyvista plotter."""
        if overlay_id == 'velocity_vectors':
            # Add velocity arrow actors to the plotter
            pass

    def register_camera_modes(self):
        """Return dict of custom camera modes."""
        return {}

    def update_camera(self, mode_id, plotter, position, orientation, dt):
        """Update camera for a custom mode."""
        pass

    def register_hud_widgets(self):
        """Return dict of HUD widget definitions."""
        return {}

    def update_hud(self, widget_id, frame_data):
        """Return updated text/value for a HUD widget."""
        return ""
```

---

## Mission Control Extensions (Detailed)

Mission Control extensions can add dashboard widgets, serial protocols, and commands.

### Hook: `MissionControlExtension`

```python
from extensions.hooks.mission_control import MissionControlExtension

class MyDashboard(MissionControlExtension):
    @property
    def name(self):
        return "Telemetry Dashboard"

    @property
    def version(self):
        return "1.0.0"

    def activate(self, app_context):
        self._app = app_context

    def deactivate(self):
        self._app = None

    def register_dashboard_widgets(self):
        """Return list of dashboard widget definitions."""
        return [
            {
                'id': 'altitude_gauge',
                'name': 'Altitude Gauge',
                'type': 'gauge',
                'description': 'Real-time altitude indicator',
            }
        ]

    def update_widget(self, widget_id, telemetry_data):
        """Return updated data for a dashboard widget."""
        if widget_id == 'altitude_gauge':
            return {'value': telemetry_data.get('altitude', 0)}
        return {}

    def register_serial_protocols(self):
        """Return dict of custom serial protocol parsers."""
        return {}

    def parse_telemetry(self, protocol_id, raw_bytes):
        """Parse raw serial data into telemetry dict."""
        return {}

    def register_commands(self):
        """Return dict of custom commands that can be sent to the vehicle."""
        return {}

    def execute_command(self, command_id, serial_bridge, params):
        """Execute a custom command via the serial bridge."""
        pass
```

---

## Manifest Reference

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (lowercase, underscores) |
| `name` | string | Human-readable display name |
| `version` | string | Semantic version (e.g., "1.0.0") |
| `extension_type` | string | One of: plotvisual, hexakinetic, hexavisual, mission_control, universal |
| `entry_point` | string | Python module name (without .py) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `author` | string | Extension author name |
| `description` | string | Short description |
| `icon` | string | Emoji icon for display |
| `tags` | string[] | Searchable tags |
| `dependencies` | string[] | Required Python packages |

### Example

```json
{
  "id": "advanced_graphs",
  "name": "Advanced Graphs Pack",
  "version": "2.1.0",
  "extension_type": "plotvisual",
  "entry_point": "graphs",
  "author": "Vortex Community",
  "description": "Additional graph types including 3D surface plots, correlation matrices, and time series analysis",
  "icon": "📊",
  "tags": ["graphs", "analysis", "statistics"],
  "dependencies": ["scipy"]
}
```

---

## Packaging & Distribution

### Creating a .vortexext package

```python
from extensions.packaging import pack_extension

# Pack a directory into a distributable .vortexext file
pack_extension('path/to/my_extension/', 'my_extension.vortexext')
```

Or from PlotVisual: Extensions → Develop tab → "Pack Extension"

### .vortexext format

A `.vortexext` file is a standard ZIP archive containing:
- `manifest.json` (required, at root level)
- Extension Python files
- Any additional assets

### Installing

- **Drag & drop**: Drop a `.vortexext` file onto the PlotVisual window
- **UI**: Extensions → Install from File
- **API**: `ExtensionManager().install_from_vortexext('path.vortexext')`
- **Manual**: Extract to `extensions/installed/<extension_id>/`

---

## Settings Schema

Extensions can expose settings that apps will render as a settings UI:

```python
class MyExtension(VortexExtension):
    def get_settings_schema(self):
        return {
            'color_scheme': {
                'type': 'choice',
                'label': 'Color Scheme',
                'options': ['viridis', 'plasma', 'inferno', 'magma'],
                'default': 'viridis',
            },
            'show_labels': {
                'type': 'bool',
                'label': 'Show Data Labels',
                'default': True,
            },
            'max_points': {
                'type': 'int',
                'label': 'Maximum Points',
                'min': 100,
                'max': 10000,
                'default': 1000,
            },
        }

    def get_setting(self, key):
        return self._settings.get(key, self.get_settings_schema()[key]['default'])

    def set_setting(self, key, value):
        self._settings[key] = value
```

---

## Legacy Plugin Compatibility

The existing `plugins/` directory and `PlotVisualPlugin` base class are still supported. The extension system automatically wraps legacy plugins via `LegacyPluginAdapter`:

```python
# Old-style plugin (still works)
from plugins.base import PlotVisualPlugin

class MyPlugin(PlotVisualPlugin):
    def name(self):
        return "Legacy Plugin"

    def register_graphs(self):
        return [...]

    def render_graph(self, key, data_store, figure, **kwargs):
        ...
```

Legacy plugins are automatically discovered from `plugins/*/` directories that contain `plugin.py` or have a `manifest.json`.

---

## Catalog System

The bundled catalog (`extensions/catalog.json`) lists available extensions. To add your extension to the catalog, submit a PR with an entry:

```json
{
  "id": "my_extension",
  "name": "My Extension",
  "version": "1.0.0",
  "extension_type": "plotvisual",
  "description": "Description for the store",
  "author": "Your Name",
  "icon": "📊",
  "featured": false,
  "bundled": false,
  "download_url": "https://github.com/you/my-ext/releases/latest",
  "tags": ["graphs"]
}
```

---

## Best Practices

1. **Keep it focused**: One extension should do one thing well
2. **Handle errors gracefully**: Wrap rendering code in try/except to avoid crashing the host app
3. **Use the data store**: Don't load files directly — use `VortexDataStore` for consistency
4. **Test with sample data**: Implement `generate_sample_data()` so users can preview your graphs
5. **Document your settings**: Use `get_settings_schema()` so the UI can render settings automatically
6. **Version your manifest**: Bump version on every release for update detection
7. **Don't modify host state**: Extensions should not modify the host app's internal state directly

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| Extension not showing up | Check `manifest.json` exists and has valid `id`, `extension_type` |
| Import error on activate | Ensure `entry_point` matches your module name (without `.py`) |
| Graph not rendering | Check that `render_graph()` writes to the Figure, not `plt` directly |
| Extension crashes app | Wrap extension code in try/except; check the terminal for stack traces |
| Can't uninstall | Bundled extensions (in `plugins/`) cannot be uninstalled, only disabled |

---

## File Reference

| File | Purpose |
|------|---------|
| `extensions/base.py` | `VortexExtension` ABC — subclass this for universal extensions |
| `extensions/hooks/plotvisual.py` | `PlotVisualExtension` + `GraphDefinition` |
| `extensions/hooks/hexakinetic.py` | `HexaKineticExtension` |
| `extensions/hooks/hexavisual.py` | `HexaVisualExtension` |
| `extensions/hooks/mission_control.py` | `MissionControlExtension` |
| `extensions/manager.py` | `ExtensionManager` — install, uninstall, enable, disable |
| `extensions/registry.py` | `ExtensionRegistry` — per-app filtering and dispatch |
| `extensions/packaging.py` | `pack_extension()`, `unpack_extension()` |
| `extensions/catalog.py` | `ExtensionCatalog` — store browsing |
| `extensions/manifest.py` | `ExtensionManifest` dataclass |

---

*Part of Project Vortex — © 2026 HexaKinetic Systems*

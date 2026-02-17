"""
Vortex Extension Framework
===========================

Universal extension system for all Vortex Desktop applications.

Extension types:
    - plotvisual    : Custom graphs, data sources, export formats for PlotVisual
    - hexakinetic   : Fault models, motor profiles, controllers for HexaKinetic
    - hexavisual    : 3D overlays, camera modes, HUD widgets for HexaVisual
    - mission_control : Dashboard widgets, serial protocols for Vortex Mission Control
    - universal     : Extensions that provide hooks for multiple apps

Directory layout:
    extensions/
        __init__.py          # This file
        base.py              # VortexExtension ABC and hook base classes
        manifest.py          # Manifest schema parsing / validation
        manager.py           # ExtensionManager (install/uninstall/enable/disable)
        catalog.py           # Bundled catalog loader
        packaging.py         # .vortexext pack/unpack utilities
        registry.py          # Per-app extension registry + hook dispatch
        state.json           # Runtime state (enabled/disabled, install dates)
        catalog.json         # Bundled catalog of known extensions
        installed/           # Extracted user-installed extensions
        hooks/
            __init__.py
            plotvisual.py    # PlotVisual-specific hook types
            hexakinetic.py   # HexaKinetic-specific hook types
            hexavisual.py    # HexaVisual-specific hook types
            mission_control.py  # Mission Control-specific hook types
"""

from extensions.base import VortexExtension, ExtensionType
from extensions.manifest import ExtensionManifest, load_manifest
from extensions.manager import ExtensionManager
from extensions.registry import ExtensionRegistry

__all__ = [
    'VortexExtension',
    'ExtensionType',
    'ExtensionManifest',
    'load_manifest',
    'ExtensionManager',
    'ExtensionRegistry',
]

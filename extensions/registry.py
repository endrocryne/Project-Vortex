"""
Extension Registry — per-app hook dispatch and legacy plugin adapter.

Each app creates an ExtensionRegistry scoped to its type.
The registry loads extensions via ExtensionManager and provides
type-safe hook access.
"""

import traceback
from typing import List, Optional, Any, Type

from extensions.base import VortexExtension, ExtensionType
from extensions.manager import ExtensionManager
from extensions.manifest import ExtensionManifest


class LegacyPluginAdapter(VortexExtension):
    """
    Wraps a legacy PlotVisualPlugin (from plugins/) as a VortexExtension.

    This allows the old plugin system to work seamlessly with the new
    extension framework without requiring any changes to existing plugins.
    """

    def __init__(self, legacy_plugin):
        self._plugin = legacy_plugin

    @property
    def name(self) -> str:
        return self._plugin.name

    @property
    def version(self) -> str:
        return self._plugin.version

    @property
    def description(self) -> str:
        return getattr(self._plugin, 'description', '') or ''

    @property
    def author(self) -> str:
        return getattr(self._plugin, 'author', '') or ''

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.PLOTVISUAL

    @property
    def legacy_plugin(self):
        """Access the wrapped PlotVisualPlugin instance."""
        return self._plugin

    def activate(self, app_context=None):
        pass

    def deactivate(self):
        pass


class ExtensionRegistry:
    """
    Per-app extension registry.

    Each Vortex app creates one of these scoped to its type (e.g. 'plotvisual').
    The registry discovers and loads extensions that target that app,
    and provides methods to query hooks by type.

    Usage:
        registry = ExtensionRegistry('plotvisual')
        registry.discover()

        # Get all graph hooks
        for ext in registry.get_extensions():
            if hasattr(ext, 'register_graphs'):
                graphs = ext.register_graphs()
    """

    def __init__(self, app_type: str, manager: Optional[ExtensionManager] = None):
        """
        Args:
            app_type: One of 'plotvisual', 'hexakinetic', 'hexavisual', 'mission_control'
            manager: Optional shared ExtensionManager instance.
                     If None, creates its own.
        """
        self.app_type = app_type
        self.manager = manager or ExtensionManager()
        self._extensions: List[VortexExtension] = []
        self._manifests: List[ExtensionManifest] = []

    def discover(self) -> List[ExtensionManifest]:
        """
        Discover and load all extensions for this app type.

        Calls manager.discover() to scan directories, then loads and
        activates extensions matching this app type.

        Returns:
            List of manifests for loaded extensions.
        """
        all_manifests = self.manager.discover()
        self._manifests = self.manager.list_for_app(self.app_type)
        self._extensions = []

        for manifest in self._manifests:
            try:
                instance = self.manager.get_instance(manifest.id)
                if instance:
                    self._extensions.append(instance)
            except Exception as e:
                print(f"[ExtensionRegistry:{self.app_type}] "
                      f"Failed to load '{manifest.id}': {e}")
                traceback.print_exc()

        return self._manifests

    def activate_all(self, app_context: Any = None):
        """
        Activate all loaded extensions, passing the app context.

        Args:
            app_context: The application instance (e.g. PlotVisualWindow)
        """
        for ext in self._extensions:
            try:
                ext.activate(app_context)
            except Exception as e:
                print(f"[ExtensionRegistry:{self.app_type}] "
                      f"Failed to activate '{ext.name}': {e}")
                traceback.print_exc()

    def deactivate_all(self):
        """Deactivate all loaded extensions."""
        for ext in self._extensions:
            try:
                ext.deactivate()
            except Exception:
                pass
        self._extensions.clear()

    def get_extensions(self) -> List[VortexExtension]:
        """Get all loaded and activated extension instances."""
        return list(self._extensions)

    def get_extensions_of_type(self, cls: Type) -> List[Any]:
        """
        Get extensions that are instances of a specific class.

        Useful for querying specific hook types:
            graphs = registry.get_extensions_of_type(PlotVisualExtension)
        """
        return [ext for ext in self._extensions if isinstance(ext, cls)]

    def get_manifests(self) -> List[ExtensionManifest]:
        """Get manifests for all loaded extensions."""
        return list(self._manifests)

    def get_legacy_plugins(self) -> list:
        """
        Get legacy PlotVisualPlugin instances (unwrapped from adapters).

        For backward compatibility with PlotVisual's existing plugin rendering.
        """
        plugins = []
        for ext in self._extensions:
            if isinstance(ext, LegacyPluginAdapter):
                plugins.append(ext.legacy_plugin)
        return plugins

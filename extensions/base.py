"""
Vortex Extension Base Classes
==============================

Defines the universal VortexExtension ABC that all extensions must subclass,
plus the ExtensionType enum and base hook classes.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional


class ExtensionType(Enum):
    """Supported extension target applications."""
    PLOTVISUAL = "plotvisual"
    HEXAKINETIC = "hexakinetic"
    HEXAVISUAL = "hexavisual"
    MISSION_CONTROL = "mission_control"
    UNIVERSAL = "universal"


class VortexExtension(ABC):
    """
    Base class for all Vortex extensions.

    Every extension must subclass this and implement the required properties
    and methods. Extensions declare which app(s) they target via
    extension_type and provide hooks through activate().

    Lifecycle:
        1. Extension is discovered from extensions/installed/ or plugins/
        2. manifest.json is read for metadata
        3. The extension class is instantiated
        4. activate(app_context) is called when the target app loads
        5. deactivate() is called on shutdown or disable
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable extension name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g. '1.0.0')."""
        ...

    @property
    def description(self) -> str:
        """Longer description of what this extension does."""
        return ""

    @property
    def author(self) -> str:
        """Extension author name."""
        return ""

    @property
    def url(self) -> str:
        """Homepage or repository URL."""
        return ""

    @property
    def min_vortex_version(self) -> str:
        """Minimum Vortex Desktop version required."""
        return "0.0.0"

    @property
    @abstractmethod
    def extension_type(self) -> ExtensionType:
        """Which app this extension targets."""
        ...

    def activate(self, app_context: Optional[Any] = None) -> None:
        """
        Called when the extension is loaded by its target app.

        Args:
            app_context: The application instance (e.g. PlotVisualWindow,
                         SimulationGUI) — allows the extension to register
                         callbacks, add UI elements, etc.
        """
        pass

    def deactivate(self) -> None:
        """Called on shutdown or when the extension is disabled at runtime."""
        pass

    def get_settings_schema(self) -> List[Dict[str, Any]]:
        """
        Optional: Return a list of setting definitions for this extension.

        Each item is a dict with:
            - key: str          (unique setting key)
            - label: str        (display label)
            - type: str         (one of 'bool', 'int', 'float', 'str', 'choice', 'color')
            - default: Any      (default value)
            - options: list     (for 'choice' type, list of allowed values)
            - description: str  (tooltip text)

        These are rendered automatically in the Settings > Extensions panel.
        """
        return []

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a persisted setting value for this extension.

        Settings are stored in extensions_state.json under the extension's ID.
        This is set up by the ExtensionManager — extensions can just call this.
        """
        if hasattr(self, '_settings') and self._settings:
            return self._settings.get(key, default)
        return default

    def set_setting(self, key: str, value: Any) -> None:
        """
        Persist a setting value for this extension.

        Changes are written to extensions_state.json by the ExtensionManager.
        """
        if not hasattr(self, '_settings'):
            self._settings = {}
        self._settings[key] = value
        # Manager will pick this up on next save cycle
        if hasattr(self, '_manager_save_callback') and self._manager_save_callback:
            self._manager_save_callback()

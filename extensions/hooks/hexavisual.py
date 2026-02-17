"""
HexaVisual Extension Hooks
============================

Hook types for HexaVisual (3D flight visualizer) extensions.
Extensions targeting HexaVisual should subclass HexaVisualExtension.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from extensions.base import VortexExtension, ExtensionType


class OverlayDefinition:
    """Defines a 3D overlay that can be toggled in the visualizer."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "🔲", default_visible: bool = True):
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.default_visible = default_visible


class CameraModeDefinition:
    """Defines a custom camera mode for the 3D viewer."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "📷"):
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon


class HUDWidgetDefinition:
    """Defines a HUD overlay widget."""

    def __init__(self, key: str, label: str, description: str = "",
                 position: str = "top-right",
                 width: int = 200, height: int = 100):
        """
        Args:
            position: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        """
        self.key = key
        self.label = label
        self.description = description
        self.position = position
        self.width = width
        self.height = height


class HexaVisualExtension(VortexExtension):
    """
    Base class for HexaVisual extensions.

    Provides hooks for:
        - 3D overlays (waypoints, geofences, wind vectors, etc.)
        - Custom camera modes
        - Render effects
        - HUD widgets
    """

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.HEXAVISUAL

    # --- Overlay hooks ---

    def register_overlays(self) -> List[OverlayDefinition]:
        """Register 3D overlays for the visualization scene."""
        return []

    def render_overlay(self, overlay_key: str, plotter: Any,
                       trajectory_data: Any, time_index: int,
                       params: Dict[str, Any]) -> None:
        """
        Render a 3D overlay into the PyVista plotter.

        Args:
            overlay_key: The overlay key from register_overlays()
            plotter: pyvista QtInteractor instance
            trajectory_data: Trajectory DataFrame
            time_index: Current playback time index
            params: Overlay configuration parameters
        """
        pass

    def remove_overlay(self, overlay_key: str, plotter: Any) -> None:
        """Remove an overlay's actors from the plotter."""
        pass

    # --- Camera mode hooks ---

    def register_camera_modes(self) -> List[CameraModeDefinition]:
        """Register custom camera modes."""
        return []

    def update_camera(self, mode_key: str, plotter: Any,
                      position: np.ndarray, velocity: np.ndarray,
                      orientation: np.ndarray, time_index: int) -> None:
        """
        Update camera position/orientation for a custom camera mode.

        Args:
            mode_key: The camera mode key
            plotter: pyvista QtInteractor instance
            position: Current rocket [x, y, z] position
            velocity: Current rocket [vx, vy, vz] velocity
            orientation: Current rocket quaternion [qw, qx, qy, qz]
            time_index: Current playback time index
        """
        pass

    # --- HUD widget hooks ---

    def register_hud_widgets(self) -> List[HUDWidgetDefinition]:
        """Register HUD overlay widgets."""
        return []

    def render_hud_widget(self, widget_key: str, painter: Any,
                          rect: Any, state: Dict[str, Any]) -> None:
        """
        Render a HUD widget using QPainter.

        Args:
            widget_key: The HUD widget key
            painter: QPainter instance
            rect: QRect defining the widget area
            state: Current visualization state dict
        """
        pass

    # --- Render effect hooks ---

    def register_render_effects(self) -> List[Dict[str, str]]:
        """
        Register custom rendering effects.

        Returns list of dicts with: key, label, description.
        """
        return []

    def apply_render_effect(self, effect_key: str, plotter: Any,
                            params: Dict[str, Any]) -> None:
        """Apply a custom rendering effect to the plotter."""
        pass

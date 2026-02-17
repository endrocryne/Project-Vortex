"""
Mission Control Extension Hooks
=================================

Hook types for Vortex Mission Control extensions.
Extensions targeting Mission Control should subclass MissionControlExtension.
"""

from typing import List, Dict, Any, Optional

from extensions.base import VortexExtension, ExtensionType


class DashboardWidgetDefinition:
    """Defines a custom dashboard widget for Mission Control."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "📊", width: str = "1fr",
                 min_height: str = "200px"):
        """
        Args:
            key: Unique widget identifier
            label: Display name
            width: CSS grid width (e.g. '1fr', '2fr', '300px')
            min_height: CSS min-height
        """
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.width = width
        self.min_height = min_height


class SerialProtocolDefinition:
    """Defines a custom serial data protocol parser."""

    def __init__(self, key: str, label: str, description: str = "",
                 baud_rate: int = 115200):
        self.key = key
        self.label = label
        self.description = description
        self.baud_rate = baud_rate


class CommandDefinition:
    """Defines a custom command that can be sent to the rocket."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "▶", parameters: Optional[List[Dict[str, Any]]] = None,
                 confirm_required: bool = False):
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.parameters = parameters or []
        self.confirm_required = confirm_required


class MissionControlExtension(VortexExtension):
    """
    Base class for Vortex Mission Control extensions.

    Provides hooks for:
        - Dashboard widgets (rendered as HTML injected into MWUI)
        - Serial protocol parsers
        - Custom commands
        - Telemetry processing
    """

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.MISSION_CONTROL

    # --- Dashboard widget hooks ---

    def register_dashboard_widgets(self) -> List[DashboardWidgetDefinition]:
        """Register custom dashboard widgets."""
        return []

    def render_widget_html(self, widget_key: str,
                           telemetry: Dict[str, Any]) -> str:
        """
        Generate HTML content for a dashboard widget.

        Args:
            widget_key: The widget key from register_dashboard_widgets()
            telemetry: Current telemetry data dict

        Returns:
            HTML string to inject into the dashboard
        """
        return "<div>No data</div>"

    def get_widget_css(self, widget_key: str) -> str:
        """Return CSS styles for a dashboard widget."""
        return ""

    def get_widget_js(self, widget_key: str) -> str:
        """Return JavaScript code for a dashboard widget."""
        return ""

    # --- Serial protocol hooks ---

    def register_serial_protocols(self) -> List[SerialProtocolDefinition]:
        """Register custom serial data protocol parsers."""
        return []

    def parse_serial_data(self, protocol_key: str,
                          raw_data: bytes) -> Dict[str, Any]:
        """
        Parse raw serial data using a custom protocol.

        Args:
            protocol_key: The protocol key
            raw_data: Raw bytes from serial port

        Returns:
            Parsed telemetry data dict
        """
        return {}

    # --- Command hooks ---

    def register_commands(self) -> List[CommandDefinition]:
        """Register custom commands sendable to the rocket."""
        return []

    def format_command(self, command_key: str,
                       params: Dict[str, Any]) -> bytes:
        """
        Format a command for serial transmission.

        Args:
            command_key: The command key
            params: Command parameters

        Returns:
            Bytes to send over serial
        """
        return b""

    # --- Telemetry processing hooks ---

    def process_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming telemetry data (add computed fields, filtering, etc.).

        Args:
            telemetry: Raw telemetry dict

        Returns:
            Processed telemetry dict (can add or modify fields)
        """
        return telemetry

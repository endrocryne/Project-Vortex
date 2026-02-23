"""
PlotVisual Extension Hooks
===========================

Hook types for PlotVisual extensions. Extensions targeting PlotVisual
should subclass PlotVisualExtension and implement the desired hooks.
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional
from matplotlib.figure import Figure

from extensions.base import VortexExtension, ExtensionType


class GraphDefinition:
    """Defines a single graph that an extension can render."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "", category: str = "General",
                 required_data_types: Optional[List[str]] = None):
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.category = category
        self.required_data_types = required_data_types or []


class PlotVisualExtension(VortexExtension):
    """
    Base class for PlotVisual extensions.

    Provides hooks for:
        - Custom graph types (register_graphs + render_graph)
        - Sample data generation
        - Custom filter columns
        - Data source registration
        - Export format registration
    """

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.PLOTVISUAL

    # --- Graph hooks ---

    def register_graphs(self) -> List[GraphDefinition]:
        """
        Return a list of GraphDefinition objects declaring what graphs
        this extension can render. Called once during extension loading.
        """
        return []

    def render_graph(self, graph_key: str, data_store: Any, fig: Figure, **kwargs) -> None:
        """
        Render the specified graph onto the provided matplotlib Figure.

        Args:
            graph_key: The key from GraphDefinition identifying which graph
            data_store: The VortexDataStore containing all loaded data
            fig: A matplotlib Figure to draw on
            **kwargs: Additional options (e.g. yscale, filters)
        """
        pass

    def generate_sample_data(self, data_store: Any) -> None:
        """Optional: populate the data_store with sample data for this extension's graphs."""
        pass

    def get_filter_columns(self) -> List[str]:
        """Optional: return additional column names for the filter panel."""
        return []

    # --- Data source hooks ---

    def register_data_sources(self) -> List[Dict[str, str]]:
        """
        Optional: Register custom data source loaders.

        Returns a list of dicts, each with:
            - key: str          (unique identifier)
            - label: str        (menu item text)
            - description: str  (tooltip)
            - file_types: str   (e.g. "HDF5 files (*.h5)")

        When selected, load_data_source(key, filepath) will be called.
        """
        return []

    def load_data_source(self, key: str, filepath: str, data_store: Any) -> bool:
        """
        Load data from a custom source into the data store.

        Args:
            key: The data source key from register_data_sources()
            filepath: User-selected file path
            data_store: The VortexDataStore to populate

        Returns:
            True if data was loaded successfully
        """
        return False

    # --- Export hooks ---

    def register_export_formats(self) -> List[Dict[str, str]]:
        """
        Optional: Register custom export formats.

        Returns a list of dicts, each with:
            - key: str          (unique identifier)
            - label: str        (menu item text)
            - extension: str    (file extension, e.g. '.html')
        """
        return []

    def export_graph(self, key: str, fig: Figure, filepath: str) -> bool:
        """
        Export the current graph in a custom format.

        Args:
            key: The export format key from register_export_formats()
            fig: The current matplotlib Figure
            filepath: User-selected output path

        Returns:
            True if exported successfully
        """
        return False

    # --- Sidebar panel hook ---

    def inject_sidebar_panel(self, parent_layout: Any, data_store: Any,
                              embed_fn: Any, status_fn: Any,
                              results_dir: str) -> None:
        """
        Optional: inject a custom QWidget block into the Graphs sidebar.

        Called by PlotVisual after the built-in graph buttons are added.
        The extension adds whatever Qt widgets it needs directly to
        *parent_layout* (a QVBoxLayout).  This lets extensions ship their
        own compact panels (e.g. data-loader groups, quick-action buttons)
        that live in the sidebar without requiring changes to PlotVisual.

        Args:
            parent_layout: The QVBoxLayout of the Graphs left panel.
            data_store:    The VortexDataStore (read/write).
            embed_fn:      PlotVisual._embed_figure -- call with a Figure to display it.
            status_fn:     PlotVisual.status_bar.showMessage -- call with a str.
            results_dir:   Path to the current results directory (for file dialogs).
        """
        pass

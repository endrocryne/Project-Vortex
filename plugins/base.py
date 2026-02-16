"""
Base plugin interface for PlotVisual.
All plugins must subclass PlotVisualPlugin and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from matplotlib.figure import Figure


class GraphDefinition:
    """Defines a single graph that a plugin can render."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "", category: str = "General",
                 required_data_types: Optional[List[str]] = None):
        """
        Args:
            key: Unique identifier for this graph (e.g. 'cliff_plot')
            label: Human-readable button label (e.g. 'Success vs Ignition Altitude')
            description: Tooltip / longer description
            icon: Emoji or short prefix for the button
            category: Grouping category for the UI
            required_data_types: List of VortexDataStore types this graph needs
                                 (e.g. ['optimization', 'trajectory', 'monte_carlo'])
                                 If None, works with any data.
        """
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.category = category
        self.required_data_types = required_data_types or []


class PlotVisualPlugin(ABC):
    """
    Base class for PlotVisual plugins.

    Lifecycle:
        1. Plugin is instantiated by the plugin loader
        2. register_graphs() is called to declare available graphs
        3. When user clicks a graph button, render_graph() is called
        4. Plugin can also contribute sample data via generate_sample_data()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (e.g. 'Claude Graphs')"""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string"""
        ...

    @property
    def description(self) -> str:
        """Optional longer description"""
        return ""

    @property
    def author(self) -> str:
        """Plugin author"""
        return ""

    @abstractmethod
    def register_graphs(self) -> List[GraphDefinition]:
        """
        Return a list of GraphDefinition objects declaring what this plugin can render.
        Called once during plugin loading.
        """
        ...

    @abstractmethod
    def render_graph(self, graph_key: str, data_store: 'VortexDataStore', fig: Figure, **kwargs) -> None:
        """
        Render the specified graph onto the provided matplotlib Figure.

        Args:
            graph_key: The key from GraphDefinition identifying which graph to render
            data_store: The VortexDataStore containing all loaded data
            fig: A matplotlib Figure to draw on (already created, plugin adds subplots)
            **kwargs: Additional options (e.g. yscale, filters)
        """
        ...

    def generate_sample_data(self, data_store: 'VortexDataStore') -> None:
        """
        Optional: populate the data_store with realistic sample data for this plugin's graphs.
        Called when user requests sample data generation.

        Args:
            data_store: The VortexDataStore to populate
        """
        pass

    def get_filter_columns(self) -> List[str]:
        """
        Optional: return additional column names this plugin wants in the filter panel.
        """
        return []

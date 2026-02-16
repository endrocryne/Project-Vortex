"""
Plugin loader — discovers and loads PlotVisual plugins from the plugins/ directory.
"""

import importlib
import os
import sys
import traceback
from typing import List, Dict
from plugins.base import PlotVisualPlugin


def discover_plugins(plugins_dir: str = None) -> List[PlotVisualPlugin]:
    """
    Scan the plugins directory for valid PlotVisualPlugin subclasses.

    Each plugin lives in a subdirectory of plugins/ and must have a __init__.py
    or plugin.py that defines a class inheriting from PlotVisualPlugin.

    Returns:
        List of instantiated plugin objects.
    """
    if plugins_dir is None:
        plugins_dir = os.path.dirname(__file__)

    loaded: List[PlotVisualPlugin] = []

    # Ensure plugins dir is importable
    parent_dir = os.path.dirname(plugins_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    for entry in sorted(os.listdir(plugins_dir)):
        entry_path = os.path.join(plugins_dir, entry)

        # Skip non-directories and internal files
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith('_') or entry.startswith('.'):
            continue

        # Try to import the plugin module
        module_name = f"plugins.{entry}"

        # Prefer plugin.py, fall back to __init__.py
        plugin_py = os.path.join(entry_path, 'plugin.py')
        init_py = os.path.join(entry_path, '__init__.py')

        target_module = None
        if os.path.exists(plugin_py):
            target_module = f"plugins.{entry}.plugin"
        elif os.path.exists(init_py):
            target_module = f"plugins.{entry}"
        else:
            continue

        try:
            mod = importlib.import_module(target_module)

            # Find all PlotVisualPlugin subclasses in the module
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type) and
                        issubclass(attr, PlotVisualPlugin) and
                        attr is not PlotVisualPlugin):
                    instance = attr()
                    loaded.append(instance)
                    print(f"[PluginLoader] Loaded: {instance.name} v{instance.version}")
                    break  # One plugin class per directory

        except Exception as e:
            print(f"[PluginLoader] Failed to load plugin '{entry}': {e}")
            traceback.print_exc()

    return loaded

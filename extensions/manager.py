"""
Extension Manager — install, uninstall, enable, disable extensions.

Manages the lifecycle of extensions and persists state to extensions_state.json.
"""

import json
import os
import shutil
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any

from extensions.base import VortexExtension, ExtensionType
from extensions.manifest import ExtensionManifest, load_manifest


# State file lives next to extensions/__init__.py
_STATE_FILE = os.path.join(os.path.dirname(__file__), 'state.json')


class ExtensionManager:
    """
    Central manager for all Vortex extensions.

    Responsibilities:
        - Discover installed extensions (extensions/installed/ + legacy plugins/)
        - Install new extensions from .vortexext files or directories
        - Uninstall extensions (remove from installed/)
        - Enable / disable extensions (toggle in state.json)
        - Persist extension settings
        - Provide extension instances to app registries
    """

    def __init__(self):
        self._installed_dir = os.path.join(os.path.dirname(__file__), 'installed')
        self._state: Dict[str, dict] = {}
        self._manifests: Dict[str, ExtensionManifest] = {}
        self._instances: Dict[str, VortexExtension] = {}
        os.makedirs(self._installed_dir, exist_ok=True)
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load extension state from state.json."""
        if os.path.exists(_STATE_FILE):
            try:
                with open(_STATE_FILE, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {}
        else:
            self._state = {}

    def _save_state(self):
        """Persist extension state to state.json (atomic write)."""
        tmp = _STATE_FILE + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2)
            # Atomic replace
            if os.path.exists(_STATE_FILE):
                os.replace(tmp, _STATE_FILE)
            else:
                os.rename(tmp, _STATE_FILE)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> List[ExtensionManifest]:
        """
        Scan installed extensions directory and legacy plugins/ for manifests.
        Returns list of all discovered manifests.
        """
        self._manifests.clear()

        # 1. Scan extensions/installed/
        if os.path.isdir(self._installed_dir):
            for entry in sorted(os.listdir(self._installed_dir)):
                ext_dir = os.path.join(self._installed_dir, entry)
                if not os.path.isdir(ext_dir):
                    continue
                if entry.startswith('_') or entry.startswith('.'):
                    continue
                try:
                    manifest = load_manifest(ext_dir, bundled=False)
                    self._manifests[manifest.id] = manifest
                    # Ensure state entry exists
                    if manifest.id not in self._state:
                        self._state[manifest.id] = {
                            'enabled': True,
                            'installed_at': datetime.now().isoformat(),
                            'version': manifest.version,
                            'settings': {},
                        }
                except Exception as e:
                    print(f"[ExtensionManager] Failed to load manifest from '{entry}': {e}")

        # 2. Scan legacy plugins/ directory for backward compatibility
        plugins_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plugins')
        if os.path.isdir(plugins_dir):
            for entry in sorted(os.listdir(plugins_dir)):
                entry_path = os.path.join(plugins_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                if entry.startswith('_') or entry.startswith('.'):
                    continue

                # Check for manifest.json first (new-style plugin)
                manifest_path = os.path.join(entry_path, 'manifest.json')
                if os.path.exists(manifest_path):
                    try:
                        manifest = load_manifest(entry_path, bundled=True)
                        if manifest.id not in self._manifests:
                            self._manifests[manifest.id] = manifest
                            if manifest.id not in self._state:
                                self._state[manifest.id] = {
                                    'enabled': True,
                                    'installed_at': 'bundled',
                                    'version': manifest.version,
                                    'settings': {},
                                }
                    except Exception as e:
                        print(f"[ExtensionManager] Failed to load plugin manifest '{entry}': {e}")
                else:
                    # Legacy plugin without manifest — create synthetic manifest
                    plugin_py = os.path.join(entry_path, 'plugin.py')
                    init_py = os.path.join(entry_path, '__init__.py')
                    if os.path.exists(plugin_py) or os.path.exists(init_py):
                        ext_id = entry
                        if ext_id not in self._manifests:
                            manifest = ExtensionManifest(
                                id=ext_id,
                                name=entry.replace('_', ' ').title(),
                                version='0.0.0',
                                extension_type='plotvisual',
                                entry_point='plugin' if os.path.exists(plugin_py) else '__init__',
                                description=f'Legacy plugin: {entry}',
                                source_path=entry_path,
                                bundled=True,
                            )
                            self._manifests[ext_id] = manifest
                            if ext_id not in self._state:
                                self._state[ext_id] = {
                                    'enabled': True,
                                    'installed_at': 'bundled',
                                    'version': '0.0.0',
                                    'settings': {},
                                }

        self._save_state()
        return list(self._manifests.values())

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install_from_directory(self, source_dir: str) -> ExtensionManifest:
        """
        Install an extension from a directory (copy into installed/).

        Args:
            source_dir: Path to directory containing manifest.json + code

        Returns:
            The installed extension's manifest

        Raises:
            FileNotFoundError: If manifest.json is missing
            ValueError: If manifest is invalid
        """
        manifest = load_manifest(source_dir)
        target_dir = os.path.join(self._installed_dir, manifest.id)

        # Remove existing if present
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir)

        manifest.source_path = target_dir
        manifest.bundled = False
        self._manifests[manifest.id] = manifest
        self._state[manifest.id] = {
            'enabled': True,
            'installed_at': datetime.now().isoformat(),
            'version': manifest.version,
            'settings': {},
        }
        self._save_state()

        print(f"[ExtensionManager] Installed: {manifest.name} v{manifest.version}")
        return manifest

    def install_from_vortexext(self, vortexext_path: str) -> ExtensionManifest:
        """
        Install an extension from a .vortexext file (zip archive).

        Args:
            vortexext_path: Path to the .vortexext file

        Returns:
            The installed extension's manifest
        """
        from extensions.packaging import unpack_extension

        ext_dir = unpack_extension(vortexext_path, self._installed_dir)
        manifest = load_manifest(ext_dir)
        manifest.source_path = ext_dir
        manifest.bundled = False
        self._manifests[manifest.id] = manifest
        self._state[manifest.id] = {
            'enabled': True,
            'installed_at': datetime.now().isoformat(),
            'version': manifest.version,
            'settings': {},
        }
        self._save_state()

        print(f"[ExtensionManager] Installed from .vortexext: {manifest.name} v{manifest.version}")
        return manifest

    # ------------------------------------------------------------------
    # Uninstall
    # ------------------------------------------------------------------

    def uninstall(self, ext_id: str) -> bool:
        """
        Uninstall an extension by removing its directory from installed/.

        Bundled extensions cannot be uninstalled (they can only be disabled).

        Returns:
            True if uninstalled, False if bundled or not found
        """
        manifest = self._manifests.get(ext_id)
        if not manifest:
            return False

        if manifest.bundled:
            print(f"[ExtensionManager] Cannot uninstall bundled extension: {ext_id}")
            return False

        target_dir = os.path.join(self._installed_dir, ext_id)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Deactivate instance if loaded
        instance = self._instances.pop(ext_id, None)
        if instance:
            try:
                instance.deactivate()
            except Exception:
                pass

        self._manifests.pop(ext_id, None)
        self._state.pop(ext_id, None)
        self._save_state()

        print(f"[ExtensionManager] Uninstalled: {ext_id}")
        return True

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(self, ext_id: str) -> bool:
        """Enable an extension. Returns True if state changed."""
        if ext_id in self._state:
            if not self._state[ext_id].get('enabled', True):
                self._state[ext_id]['enabled'] = True
                self._save_state()
                return True
        return False

    def disable(self, ext_id: str) -> bool:
        """Disable an extension. Returns True if state changed."""
        if ext_id in self._state:
            if self._state[ext_id].get('enabled', True):
                self._state[ext_id]['enabled'] = False
                self._save_state()

                # Deactivate instance if loaded
                instance = self._instances.pop(ext_id, None)
                if instance:
                    try:
                        instance.deactivate()
                    except Exception:
                        pass
                return True
        return False

    def is_enabled(self, ext_id: str) -> bool:
        """Check if an extension is enabled."""
        return self._state.get(ext_id, {}).get('enabled', True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_installed(self) -> List[ExtensionManifest]:
        """Return all discovered extension manifests."""
        return list(self._manifests.values())

    def list_enabled(self) -> List[ExtensionManifest]:
        """Return manifests of enabled extensions only."""
        return [m for m in self._manifests.values() if self.is_enabled(m.id)]

    def list_for_app(self, app_type: str) -> List[ExtensionManifest]:
        """Return enabled extension manifests that target the given app type."""
        return [
            m for m in self._manifests.values()
            if self.is_enabled(m.id) and (
                m.extension_type == app_type or m.extension_type == 'universal'
            )
        ]

    def get_manifest(self, ext_id: str) -> Optional[ExtensionManifest]:
        """Get manifest for a specific extension."""
        return self._manifests.get(ext_id)

    def get_state(self, ext_id: str) -> dict:
        """Get the full state dict for an extension (enabled, settings, etc.)."""
        return self._state.get(ext_id, {})

    def get_settings(self, ext_id: str) -> dict:
        """Get persisted settings for an extension."""
        return self._state.get(ext_id, {}).get('settings', {})

    def save_settings(self, ext_id: str, settings: dict):
        """Persist settings for an extension."""
        if ext_id in self._state:
            self._state[ext_id]['settings'] = settings
            self._save_state()

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def get_instance(self, ext_id: str) -> Optional[VortexExtension]:
        """Get or create the VortexExtension instance for an extension."""
        if ext_id in self._instances:
            return self._instances[ext_id]

        manifest = self._manifests.get(ext_id)
        if not manifest or not self.is_enabled(ext_id):
            return None

        try:
            instance = self._load_extension_class(manifest)
            if instance:
                # Inject settings
                instance._settings = self.get_settings(ext_id)
                instance._manager_save_callback = lambda: self.save_settings(ext_id, instance._settings)
                self._instances[ext_id] = instance
                return instance
        except Exception as e:
            print(f"[ExtensionManager] Failed to instantiate '{ext_id}': {e}")
            traceback.print_exc()
        return None

    def _load_extension_class(self, manifest: ExtensionManifest) -> Optional[VortexExtension]:
        """
        Import and instantiate the extension class from its source directory.
        """
        import importlib
        import sys

        source_path = manifest.source_path
        if not source_path or not os.path.isdir(source_path):
            return None

        parent_dir = os.path.dirname(source_path)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        dir_name = os.path.basename(source_path)
        module_name = f"{dir_name}.{manifest.entry_point}"

        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            # Try with full parent context
            grandparent = os.path.basename(parent_dir)
            module_name = f"{grandparent}.{dir_name}.{manifest.entry_point}"
            try:
                mod = importlib.import_module(module_name)
            except ImportError:
                # Last resort: direct path import
                import importlib.util
                entry_file = os.path.join(source_path, manifest.entry_point + '.py')
                if not os.path.exists(entry_file):
                    entry_file = os.path.join(source_path, manifest.entry_point, '__init__.py')
                if not os.path.exists(entry_file):
                    return None
                spec = importlib.util.spec_from_file_location(module_name, entry_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

        # Find VortexExtension subclass (or legacy PlotVisualPlugin subclass)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and attr is not VortexExtension:
                if issubclass(attr, VortexExtension):
                    return attr()
                # Check for legacy PlotVisualPlugin
                try:
                    from plugins.base import PlotVisualPlugin
                    if issubclass(attr, PlotVisualPlugin) and attr is not PlotVisualPlugin:
                        # Wrap legacy plugin in a VortexExtension adapter
                        from extensions.registry import LegacyPluginAdapter
                        return LegacyPluginAdapter(attr())
                except ImportError:
                    pass

        return None

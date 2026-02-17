"""
Extension Manifest — schema parsing and validation for manifest.json files.

Every extension directory must contain a manifest.json with metadata about
the extension: its ID, name, version, type, entry point, etc.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtensionManifest:
    """Parsed and validated extension manifest."""

    id: str                             # Unique identifier (e.g. 'claude_graphs')
    name: str                           # Human-readable name
    version: str                        # Semantic version string
    extension_type: str                 # 'plotvisual', 'hexakinetic', 'hexavisual', 'mission_control', 'universal'
    entry_point: str                    # Python module path relative to extension dir (e.g. 'plugin')

    author: str = ""                    # Author name
    description: str = ""               # What this extension does
    min_vortex_version: str = "0.0.0"   # Minimum Vortex Desktop version
    url: str = ""                       # Homepage / repository URL
    icon: str = ""                      # Path to icon file within extension dir, or emoji
    dependencies: List[str] = field(default_factory=list)   # pip package names
    tags: List[str] = field(default_factory=list)           # searchable tags
    license: str = ""                   # License identifier

    # Set at load time (not from JSON)
    source_path: str = ""               # Absolute path to the extension directory
    bundled: bool = False               # True if shipped with Vortex (not user-installed)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (for catalog / state files)."""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'extension_type': self.extension_type,
            'entry_point': self.entry_point,
            'author': self.author,
            'description': self.description,
            'min_vortex_version': self.min_vortex_version,
            'url': self.url,
            'icon': self.icon,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'license': self.license,
        }

    @staticmethod
    def from_dict(data: dict, source_path: str = "", bundled: bool = False) -> 'ExtensionManifest':
        """Create a manifest from a parsed JSON dict."""
        required_fields = ['id', 'name', 'version', 'extension_type', 'entry_point']
        for f in required_fields:
            if f not in data:
                raise ValueError(f"Manifest missing required field: '{f}'")

        valid_types = ('plotvisual', 'hexakinetic', 'hexavisual', 'mission_control', 'universal')
        if data['extension_type'] not in valid_types:
            raise ValueError(
                f"Invalid extension_type '{data['extension_type']}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )

        return ExtensionManifest(
            id=data['id'],
            name=data['name'],
            version=data['version'],
            extension_type=data['extension_type'],
            entry_point=data['entry_point'],
            author=data.get('author', ''),
            description=data.get('description', ''),
            min_vortex_version=data.get('min_vortex_version', '0.0.0'),
            url=data.get('url', ''),
            icon=data.get('icon', ''),
            dependencies=data.get('dependencies', []),
            tags=data.get('tags', []),
            license=data.get('license', ''),
            source_path=source_path,
            bundled=bundled,
        )


def load_manifest(ext_dir: str, bundled: bool = False) -> ExtensionManifest:
    """
    Load and validate a manifest.json from an extension directory.

    Args:
        ext_dir: Absolute path to the extension directory containing manifest.json
        bundled: Whether this extension is shipped with Vortex

    Returns:
        Parsed ExtensionManifest

    Raises:
        FileNotFoundError: If manifest.json doesn't exist
        ValueError: If manifest is invalid
        json.JSONDecodeError: If manifest is malformed JSON
    """
    manifest_path = os.path.join(ext_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json found in {ext_dir}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return ExtensionManifest.from_dict(data, source_path=ext_dir, bundled=bundled)


def validate_manifest_dict(data: dict) -> List[str]:
    """
    Validate a manifest dict and return a list of error messages.
    Returns empty list if valid.
    """
    errors = []

    required_fields = ['id', 'name', 'version', 'extension_type', 'entry_point']
    for f in required_fields:
        if f not in data:
            errors.append(f"Missing required field: '{f}'")

    if 'id' in data:
        ext_id = data['id']
        if not ext_id.replace('_', '').replace('-', '').isalnum():
            errors.append(f"Invalid id '{ext_id}': must be alphanumeric with underscores/hyphens only")

    if 'version' in data:
        parts = data['version'].split('.')
        if len(parts) < 2 or not all(p.isdigit() for p in parts):
            errors.append(f"Invalid version '{data['version']}': use semantic versioning (e.g. '1.0.0')")

    valid_types = ('plotvisual', 'hexakinetic', 'hexavisual', 'mission_control', 'universal')
    if 'extension_type' in data and data['extension_type'] not in valid_types:
        errors.append(f"Invalid extension_type '{data['extension_type']}'. Must be one of: {', '.join(valid_types)}")

    if 'dependencies' in data and not isinstance(data['dependencies'], list):
        errors.append("'dependencies' must be a list of package names")

    if 'tags' in data and not isinstance(data['tags'], list):
        errors.append("'tags' must be a list of strings")

    return errors

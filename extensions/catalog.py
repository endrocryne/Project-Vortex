"""
Extension Catalog — bundled catalog of known Vortex extensions.

The catalog is a curated list of extensions with metadata and download info.
It supports both bundled extensions (shipped with Vortex) and external ones
(with download URLs for future marketplace support).
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CatalogEntry:
    """A single entry in the extension catalog."""
    id: str
    name: str
    version: str
    extension_type: str
    author: str = ""
    description: str = ""
    icon: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "General"
    download_url: Optional[str] = None
    bundled: bool = False
    featured: bool = False

    @staticmethod
    def from_dict(data: dict) -> 'CatalogEntry':
        return CatalogEntry(
            id=data['id'],
            name=data['name'],
            version=data.get('version', '0.0.0'),
            extension_type=data.get('extension_type', 'plotvisual'),
            author=data.get('author', ''),
            description=data.get('description', ''),
            icon=data.get('icon', ''),
            tags=data.get('tags', []),
            category=data.get('category', 'General'),
            download_url=data.get('download_url'),
            bundled=data.get('bundled', False),
            featured=data.get('featured', False),
        )


class ExtensionCatalog:
    """
    Manages the bundled extension catalog.

    Reads catalog.json and provides query methods for the extension store UI.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), 'catalog.json')
        self._catalog_path = catalog_path
        self._entries: List[CatalogEntry] = []
        self._load()

    def _load(self):
        """Load the catalog from JSON."""
        if not os.path.exists(self._catalog_path):
            self._entries = []
            return

        try:
            with open(self._catalog_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._entries = []
            for item in data.get('extensions', []):
                try:
                    self._entries.append(CatalogEntry.from_dict(item))
                except Exception as e:
                    print(f"[Catalog] Failed to parse entry: {e}")

        except Exception as e:
            print(f"[Catalog] Failed to load catalog: {e}")
            self._entries = []

    def get_all(self) -> List[CatalogEntry]:
        """Get all catalog entries."""
        return list(self._entries)

    def get_featured(self) -> List[CatalogEntry]:
        """Get featured extensions."""
        return [e for e in self._entries if e.featured]

    def get_by_type(self, extension_type: str) -> List[CatalogEntry]:
        """Get extensions for a specific app type."""
        return [e for e in self._entries if e.extension_type == extension_type]

    def get_by_category(self, category: str) -> List[CatalogEntry]:
        """Get extensions in a specific category."""
        return [e for e in self._entries if e.category == category]

    def get_by_id(self, ext_id: str) -> Optional[CatalogEntry]:
        """Get a specific catalog entry by ID."""
        for e in self._entries:
            if e.id == ext_id:
                return e
        return None

    def search(self, query: str) -> List[CatalogEntry]:
        """Search catalog by name, description, tags, or author."""
        q = query.lower()
        results = []
        for e in self._entries:
            if (q in e.name.lower() or
                q in e.description.lower() or
                q in e.author.lower() or
                any(q in t.lower() for t in e.tags)):
                results.append(e)
        return results

    def categories(self) -> List[str]:
        """Get all unique categories in the catalog."""
        cats = set()
        for e in self._entries:
            cats.add(e.category)
        return sorted(cats)

    def is_installed(self, ext_id: str, manager) -> bool:
        """Check if a catalog entry is already installed."""
        return manager.get_manifest(ext_id) is not None

    def has_update(self, ext_id: str, manager) -> bool:
        """Check if an installed extension has a newer version in the catalog."""
        catalog_entry = self.get_by_id(ext_id)
        if not catalog_entry:
            return False
        manifest = manager.get_manifest(ext_id)
        if not manifest:
            return False
        return self._version_gt(catalog_entry.version, manifest.version)

    @staticmethod
    def _version_gt(v1: str, v2: str) -> bool:
        """Check if v1 > v2 (semantic versioning)."""
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            while len(parts1) < len(parts2):
                parts1.append(0)
            while len(parts2) < len(parts1):
                parts2.append(0)
            return parts1 > parts2
        except Exception:
            return False

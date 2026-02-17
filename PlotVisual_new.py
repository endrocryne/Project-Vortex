"""
PlotVisual v2.0 - Advanced Visualization Suite for Project Vortex
==================================================================

PyQt5-based graph analysis tool with:
- Sidebar navigation (Home, Graphs, Extensions, Settings)
- Light/Dark theming with accent color customization
- Extension manager with bundled catalog and .vortexext install
- Built-in graphs: Velocity vs Intensity, Success Heatmap, Landing Accuracy,
  Landing Top-Down, Monte Carlo Distributions
- Plugin system for extensible graph types
- Interactive filtering, export, and preferences persistence
"""

import sys
import os
import json
import traceback
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

# Set matplotlib backend before any pyplot import
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QFrame, QPushButton, QLabel,
    QFileDialog, QMessageBox, QScrollArea, QSplitter, QSlider,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QGroupBox, QTabWidget, QStatusBar, QSizePolicy, QAction,
    QToolBar, QMenu, QMenuBar, QListWidget, QListWidgetItem,
    QProgressBar, QSpacerItem, QTextEdit
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QMimeData, QUrl
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QDragEnterEvent, QDropEvent

# Vortex imports
from vortex_theme import (
    apply_theme, get_matplotlib_style, DARK, LIGHT,
    ACCENT_BLUE, ACCENT_PRESETS
)
from plugins.data_store import VortexDataStore
from extensions.manager import ExtensionManager
from extensions.registry import ExtensionRegistry
from extensions.catalog import ExtensionCatalog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_VERSION = "2.0.0"
APP_TITLE = "PlotVisual"
PREFS_FILE = os.path.join(os.path.dirname(__file__), 'plotvisual_prefs.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

SIDEBAR_ICONS = {
    'home': '\U0001F3E0',       # 🏠
    'graphs': '\U0001F4CA',     # 📊
    'extensions': '\U0001F9E9', # 🧩
    'settings': '\U00002699',   # ⚙
}


# ===================================================================
# Preferences Manager
# ===================================================================

class PrefsManager:
    """Load, save, and access plotvisual_prefs.json."""

    DEFAULTS = {
        'schema_version': 2,
        'general': {
            'remember_last_source': True,
            'last_file': '',
            'last_folder': '',
            'recent_sources': [],
        },
        'plot': {
            'default_view': 'velocity',
            'yscale_velocity': 'Log',
            'marker_size': 50,
            'point_alpha': 0.6,
            'hist_bins': 30,
            'export_dpi': 300,
            'show_grid': True,
        },
        'appearance': {
            'theme': 'Dark',
            'accent_color': '#007acc',
            'font_scale': 13,
        },
        'extensions': {
            'auto_enable': True,
            'ext_directory': 'extensions/installed',
        },
    }

    def __init__(self, path: str = PREFS_FILE):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        """Load prefs from JSON, merging with defaults."""
        self.data = json.loads(json.dumps(self.DEFAULTS))
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                self._deep_merge(self.data, saved)
            except Exception as e:
                print(f"[Prefs] Failed to load {self.path}: {e}")

    def save(self):
        """Persist prefs to JSON."""
        try:
            tmpf = self.path + '.tmp'
            with open(tmpf, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            if os.path.exists(self.path):
                os.replace(tmpf, self.path)
            else:
                os.rename(tmpf, self.path)
        except Exception as e:
            print(f"[Prefs] Failed to save: {e}")

    def get(self, section: str, key: str, default=None):
        return self.data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value):
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value

    def add_recent(self, filepath: str):
        recents = self.data.get('general', {}).get('recent_sources', [])
        if filepath in recents:
            recents.remove(filepath)
        recents.insert(0, filepath)
        self.data['general']['recent_sources'] = recents[:10]
        self.data['general']['last_file'] = filepath
        self.save()

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                PrefsManager._deep_merge(base[k], v)
            else:
                base[k] = v


# ===================================================================
# Matplotlib Canvas Widget
# ===================================================================

class MplCanvas(FigureCanvasQTAgg):
    """Embeddable matplotlib canvas with Vortex theme awareness."""

    def __init__(self, parent=None, width=10, height=7, dpi=100, theme='dark'):
        style = get_matplotlib_style(theme)
        with plt.style.context(style):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ===================================================================
# Sidebar Button
# ===================================================================

class SidebarButton(QPushButton):
    """Icon button for the sidebar navigation."""

    def __init__(self, icon_text: str, tooltip: str, parent=None):
        super().__init__(icon_text, parent)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setProperty('class', 'sidebar-btn')
        self.setFixedSize(44, 44)
        font = self.font()
        font.setPointSize(16)
        self.setFont(font)


# ===================================================================
# Extension Card Widget
# ===================================================================

class ExtensionCard(QFrame):
    """Card widget displaying an extension in the manager."""

    toggled = pyqtSignal(str, bool)    # ext_id, enabled
    uninstall_clicked = pyqtSignal(str)  # ext_id

    def __init__(self, manifest, is_enabled: bool, is_bundled: bool, parent=None):
        super().__init__(parent)
        self.ext_id = manifest.id
        self.setProperty('class', 'card')
        self.setMinimumHeight(90)
        self.setMaximumHeight(120)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Icon
        icon_lbl = QLabel(manifest.icon or '\U0001F9E9')
        icon_lbl.setFont(QFont('Segoe UI Emoji', 24))
        icon_lbl.setFixedWidth(48)
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_row = QHBoxLayout()
        name_lbl = QLabel(manifest.name)
        name_lbl.setProperty('class', 'subheading')
        name_row.addWidget(name_lbl)

        ver_lbl = QLabel(f'v{manifest.version}')
        ver_lbl.setProperty('class', 'muted')
        name_row.addWidget(ver_lbl)

        if is_bundled:
            badge = QLabel('BUNDLED')
            badge.setStyleSheet(
                'background-color: #2a4a2a; color: #4caf50; '
                'border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: bold;'
            )
            name_row.addWidget(badge)

        name_row.addStretch()
        info_layout.addLayout(name_row)

        desc_lbl = QLabel(manifest.description[:120] if manifest.description else 'No description')
        desc_lbl.setProperty('class', 'muted')
        desc_lbl.setWordWrap(True)
        info_layout.addWidget(desc_lbl)

        author_lbl = QLabel(f'by {manifest.author}' if manifest.author else '')
        author_lbl.setProperty('class', 'muted')
        info_layout.addWidget(author_lbl)

        layout.addLayout(info_layout, stretch=1)

        # Controls
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(6)

        self.toggle_cb = QCheckBox('Enabled')
        self.toggle_cb.setChecked(is_enabled)
        self.toggle_cb.toggled.connect(lambda checked: self.toggled.emit(self.ext_id, checked))
        ctrl_layout.addWidget(self.toggle_cb)

        if not is_bundled:
            uninstall_btn = QPushButton('Uninstall')
            uninstall_btn.setProperty('class', 'danger')
            uninstall_btn.setFixedWidth(80)
            uninstall_btn.clicked.connect(lambda: self.uninstall_clicked.emit(self.ext_id))
            ctrl_layout.addWidget(uninstall_btn)

        layout.addLayout(ctrl_layout)


# ===================================================================
# Catalog Card Widget
# ===================================================================

class CatalogCard(QFrame):
    """Card widget for an extension in the Store tab."""

    install_clicked = pyqtSignal(str)  # ext_id

    def __init__(self, entry, is_installed: bool, parent=None):
        super().__init__(parent)
        self.ext_id = entry.id
        self.setProperty('class', 'card')
        self.setMinimumHeight(80)
        self.setMaximumHeight(110)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        icon_lbl = QLabel(entry.icon or '\U0001F4E6')
        icon_lbl.setFont(QFont('Segoe UI Emoji', 22))
        icon_lbl.setFixedWidth(44)
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_row = QHBoxLayout()
        name_lbl = QLabel(entry.name)
        name_lbl.setProperty('class', 'subheading')
        name_row.addWidget(name_lbl)

        ver_lbl = QLabel(f'v{entry.version}')
        ver_lbl.setProperty('class', 'muted')
        name_row.addWidget(ver_lbl)

        type_lbl = QLabel(entry.extension_type)
        type_lbl.setStyleSheet(
            'background-color: rgba(0,122,204,0.15); color: #007acc; '
            'border-radius: 4px; padding: 2px 6px; font-size: 10px;'
        )
        name_row.addWidget(type_lbl)

        if entry.featured:
            star = QLabel('\u2B50')
            star.setToolTip('Featured')
            name_row.addWidget(star)

        name_row.addStretch()
        info_layout.addLayout(name_row)

        desc_lbl = QLabel(entry.description[:120] if entry.description else '')
        desc_lbl.setProperty('class', 'muted')
        desc_lbl.setWordWrap(True)
        info_layout.addWidget(desc_lbl)

        layout.addLayout(info_layout, stretch=1)

        if is_installed:
            installed_lbl = QLabel('\u2705 Installed')
            installed_lbl.setStyleSheet('color: #4caf50; font-weight: bold;')
            layout.addWidget(installed_lbl)
        elif entry.bundled:
            bundled_lbl = QLabel('Built-in')
            bundled_lbl.setProperty('class', 'muted')
            layout.addWidget(bundled_lbl)
        else:
            install_btn = QPushButton('Install')
            install_btn.setProperty('class', 'primary')
            install_btn.setFixedWidth(80)
            install_btn.setEnabled(entry.download_url is not None)
            if not entry.download_url:
                install_btn.setToolTip('No download URL available')
            install_btn.clicked.connect(lambda: self.install_clicked.emit(self.ext_id))
            layout.addWidget(install_btn)


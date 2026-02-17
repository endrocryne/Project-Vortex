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
PREFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plotvisual_prefs.json')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

SIDEBAR_ICONS = {
    'home': '\U0001F3E0',       # house
    'graphs': '\U0001F4CA',     # chart
    'extensions': '\U0001F9E9', # puzzle
    'settings': '\u2699',       # gear
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

    def __init__(self, path=PREFS_FILE):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        self.data = json.loads(json.dumps(self.DEFAULTS))
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                self._deep_merge(self.data, saved)
            except Exception as e:
                print(f"[Prefs] Failed to load {self.path}: {e}")

    def save(self):
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

    def get(self, section, key, default=None):
        return self.data.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value

    def add_recent(self, filepath):
        recents = self.data.get('general', {}).get('recent_sources', [])
        if filepath in recents:
            recents.remove(filepath)
        recents.insert(0, filepath)
        self.data['general']['recent_sources'] = recents[:10]
        self.data['general']['last_file'] = filepath
        self.save()

    @staticmethod
    def _deep_merge(base, override):
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
    def __init__(self, icon_text, tooltip, parent=None):
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
    toggled = pyqtSignal(str, bool)
    uninstall_clicked = pyqtSignal(str)

    def __init__(self, manifest, is_enabled, is_bundled, parent=None):
        super().__init__(parent)
        self.ext_id = manifest.id
        self.setProperty('class', 'card')
        self.setMinimumHeight(90)
        self.setMaximumHeight(120)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        icon_lbl = QLabel(manifest.icon or '\U0001F9E9')
        icon_lbl.setFont(QFont('Segoe UI Emoji', 24))
        icon_lbl.setFixedWidth(48)
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

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
    install_clicked = pyqtSignal(str)

    def __init__(self, entry, is_installed, parent=None):
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


# ===================================================================
# Graph Button Widget
# ===================================================================

class GraphButton(QPushButton):
    """Clickable graph button for the left panel."""

    def __init__(self, icon_text, label, description='', parent=None):
        text = f'{icon_text}  {label}'
        super().__init__(text, parent)
        self.graph_key = label
        if description:
            self.setToolTip(description)
        self.setFixedHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 12px;
                border-radius: 6px;
            }
        """)


# ===================================================================
# Main Window
# ===================================================================

class PlotVisualWindow(QMainWindow):
    """PlotVisual v2.0 main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'{APP_TITLE} - Vortex Desktop')
        self.setMinimumSize(1200, 750)
        self.resize(1440, 900)

        # Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'hexakinetic_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Core state
        self.prefs = PrefsManager()
        self.data_store = VortexDataStore()
        self.data = None
        self.filtered_data = None
        self.current_canvas = None
        self.current_figure = None
        self.last_view = None

        # Extension system
        self.ext_manager = ExtensionManager()
        self.ext_registry = ExtensionRegistry('plotvisual', self.ext_manager)
        self.ext_catalog = ExtensionCatalog()
        self.plugin_graphs = {}

        # Theme state
        self._theme_mode = self.prefs.get('appearance', 'theme', 'Dark').lower()
        self._accent_color = self.prefs.get('appearance', 'accent_color', ACCENT_BLUE)

        # Build UI
        self._build_ui()
        self._load_extensions()
        self._apply_theme()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready \u2014 Load data to begin analysis')

    # ------------------------------------------------------------------
    # Extension loading
    # ------------------------------------------------------------------

    def _load_extensions(self):
        """Discover and load all PlotVisual extensions."""
        try:
            self.ext_registry.discover()
            self.ext_registry.activate_all(self)

            # Collect graph definitions from extensions
            for ext in self.ext_registry.get_extensions():
                # Try new-style PlotVisualExtension
                if hasattr(ext, 'register_graphs'):
                    for gdef in ext.register_graphs():
                        self.plugin_graphs[gdef.key] = (ext, gdef)

                # Also check legacy plugins wrapped in adapter
                if hasattr(ext, 'legacy_plugin'):
                    lp = ext.legacy_plugin
                    if hasattr(lp, 'register_graphs'):
                        for gdef in lp.register_graphs():
                            self.plugin_graphs[gdef.key] = (lp, gdef)

            if self.plugin_graphs:
                print(f'[PlotVisual] Loaded {len(self.plugin_graphs)} extension graphs')
                self._populate_extension_graphs()

        except Exception as e:
            print(f'[PlotVisual] Extension loading error: {e}')
            traceback.print_exc()

    def _populate_extension_graphs(self):
        """Add extension graph buttons to the graphs sidebar."""
        if not hasattr(self, '_ext_graph_container'):
            return

        # Clear existing extension buttons
        layout = self._ext_graph_container.layout()
        if layout is None:
            layout = QVBoxLayout(self._ext_graph_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

        # Group by category
        categories = {}
        for key, (ext, gdef) in self.plugin_graphs.items():
            cat = getattr(gdef, 'category', 'Extensions')
            categories.setdefault(cat, []).append((key, gdef))

        for cat, items in categories.items():
            cat_label = QLabel(cat)
            cat_label.setProperty('class', 'muted')
            cat_label.setStyleSheet('font-weight: bold; margin-top: 8px; padding-left: 4px;')
            layout.addWidget(cat_label)

            for key, gdef in items:
                icon = getattr(gdef, 'icon', '\U0001F9E9')
                label = getattr(gdef, 'label', key)
                desc = getattr(gdef, 'description', '')
                btn = GraphButton(icon, label, desc)
                btn.graph_key = key
                btn.clicked.connect(partial(self._render_plugin_graph, key))
                layout.addWidget(btn)

    # ------------------------------------------------------------------
    # Theme management
    # ------------------------------------------------------------------

    def _apply_theme(self):
        """Apply theme based on current prefs."""
        app = QApplication.instance()
        if app:
            apply_theme(app, mode=self._theme_mode, accent=self._accent_color)
        # Update any existing matplotlib canvas
        if self.last_view:
            self._refresh_current_graph()

    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        if self._theme_mode == 'dark':
            self._theme_mode = 'light'
        else:
            self._theme_mode = 'dark'
        self.prefs.set('appearance', 'theme', self._theme_mode.capitalize())
        self.prefs.save()
        self._apply_theme()
        self._update_theme_button()

    def _update_theme_button(self):
        if hasattr(self, 'theme_btn'):
            if self._theme_mode == 'dark':
                self.theme_btn.setText('\u2600')
                self.theme_btn.setToolTip('Switch to Light theme')
            else:
                self.theme_btn.setText('\U0001F319')
                self.theme_btn.setToolTip('Switch to Dark theme')

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the complete UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self._build_sidebar(main_layout)

        # Content stack
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, stretch=1)

        # Build pages
        self._build_home_page()
        self._build_graphs_page()
        self._build_extensions_page()
        self._build_settings_page()

        # Default to home
        self._navigate_to(0)

    def _build_sidebar(self, parent_layout):
        """Build the left icon sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(52)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 12, 4, 12)
        sidebar_layout.setSpacing(4)

        # App icon/title
        app_icon = QLabel('\U0001F680')
        app_icon.setFont(QFont('Segoe UI Emoji', 18))
        app_icon.setAlignment(Qt.AlignCenter)
        app_icon.setToolTip(f'{APP_TITLE} v{APP_VERSION}')
        sidebar_layout.addWidget(app_icon)
        sidebar_layout.addSpacing(16)

        # Navigation buttons
        self.sidebar_buttons = []
        for idx, (key, icon) in enumerate(SIDEBAR_ICONS.items()):
            btn = SidebarButton(icon, key.capitalize())
            btn.clicked.connect(partial(self._navigate_to, idx))
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons.append(btn)

        sidebar_layout.addStretch()

        # Theme toggle at bottom
        self.theme_btn = QPushButton('\u2600' if self._theme_mode == 'dark' else '\U0001F319')
        self.theme_btn.setProperty('class', 'sidebar-btn')
        self.theme_btn.setFixedSize(44, 44)
        self.theme_btn.setToolTip('Toggle theme')
        font = self.theme_btn.font()
        font.setPointSize(14)
        self.theme_btn.setFont(font)
        self.theme_btn.clicked.connect(self._toggle_theme)
        sidebar_layout.addWidget(self.theme_btn)

        parent_layout.addWidget(sidebar)

    def _navigate_to(self, page_index):
        """Switch to a different page in the content stack."""
        self.content_stack.setCurrentIndex(page_index)
        for i, btn in enumerate(self.sidebar_buttons):
            btn.setChecked(i == page_index)

    # ------------------------------------------------------------------
    # HOME PAGE
    # ------------------------------------------------------------------

    def _build_home_page(self):
        """Build the Home/Welcome page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Header
        header = QLabel(f'\U0001F680 {APP_TITLE}')
        header.setProperty('class', 'heading')
        header.setFont(QFont('Segoe UI', 28, QFont.Bold))
        layout.addWidget(header)

        subtitle = QLabel('Advanced Visualization Suite for Project Vortex')
        subtitle.setProperty('class', 'subheading')
        layout.addWidget(subtitle)
        layout.addSpacing(10)

        # Quick actions card
        actions_card = QFrame()
        actions_card.setProperty('class', 'card')
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(20, 16, 20, 16)
        actions_title = QLabel('Quick Actions')
        actions_title.setProperty('class', 'subheading')
        actions_layout.addWidget(actions_title)

        btn_row = QHBoxLayout()
        load_btn = QPushButton('\U0001F4C2  Load CSV')
        load_btn.setProperty('class', 'primary')
        load_btn.clicked.connect(self._load_data)
        btn_row.addWidget(load_btn)

        sample_btn = QPushButton('\U0001F4CA  Load Sample Data')
        sample_btn.clicked.connect(self._load_sample_data)
        btn_row.addWidget(sample_btn)

        traj_btn = QPushButton('\U0001F3AF  Load Trajectory')
        traj_btn.clicked.connect(self._load_trajectory)
        btn_row.addWidget(traj_btn)

        dir_btn = QPushButton('\U0001F4C1  Load Results Dir')
        dir_btn.clicked.connect(self._load_results_dir)
        btn_row.addWidget(dir_btn)

        btn_row.addStretch()
        actions_layout.addLayout(btn_row)
        layout.addWidget(actions_card)

        # Data summary card
        self.data_summary_card = QFrame()
        self.data_summary_card.setProperty('class', 'card')
        summary_layout = QVBoxLayout(self.data_summary_card)
        summary_layout.setContentsMargins(20, 16, 20, 16)
        summary_title = QLabel('Data Summary')
        summary_title.setProperty('class', 'subheading')
        summary_layout.addWidget(summary_title)
        self.data_summary_label = QLabel('No data loaded yet. Use the quick actions above to load data.')
        self.data_summary_label.setProperty('class', 'muted')
        self.data_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.data_summary_label)
        layout.addWidget(self.data_summary_card)

        # Recent files card
        recents_card = QFrame()
        recents_card.setProperty('class', 'card')
        recents_layout = QVBoxLayout(recents_card)
        recents_layout.setContentsMargins(20, 16, 20, 16)
        recents_title = QLabel('Recent Files')
        recents_title.setProperty('class', 'subheading')
        recents_layout.addWidget(recents_title)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(150)
        self.recent_list.itemDoubleClicked.connect(self._open_recent_file)
        recents_layout.addWidget(self.recent_list)
        self._refresh_recent_list()
        layout.addWidget(recents_card)

        # Extension info
        ext_count = len(self.ext_manager.discover())
        ext_info = QLabel(f'\U0001F9E9 {ext_count} extension(s) available  |  Version {APP_VERSION}')
        ext_info.setProperty('class', 'muted')
        ext_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(ext_info)

        layout.addStretch()
        self.content_stack.addWidget(page)

    def _refresh_recent_list(self):
        self.recent_list.clear()
        recents = self.prefs.get('general', 'recent_sources', [])
        for fp in recents:
            if os.path.exists(fp):
                item = QListWidgetItem(os.path.basename(fp))
                item.setToolTip(fp)
                item.setData(Qt.UserRole, fp)
                self.recent_list.addItem(item)

    def _open_recent_file(self, item):
        fp = item.data(Qt.UserRole)
        if fp and os.path.exists(fp):
            self._load_csv_file(fp)

    # ------------------------------------------------------------------
    # GRAPHS PAGE
    # ------------------------------------------------------------------

    def _build_graphs_page(self):
        """Build the Graphs workspace page."""
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        # Left panel: graph list + filters
        left_panel = QFrame()
        left_panel.setObjectName('left-panel')
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        graphs_title = QLabel('Graphs')
        graphs_title.setProperty('class', 'heading')
        graphs_title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        left_layout.addWidget(graphs_title)

        # Built-in graphs
        builtin_label = QLabel('Built-in')
        builtin_label.setProperty('class', 'muted')
        builtin_label.setStyleSheet('font-weight: bold; padding-left: 4px;')
        left_layout.addWidget(builtin_label)

        builtin_graphs = [
            ('\U0001F3AF', 'Velocity vs Intensity', 'Landing velocity scatter plot by fault intensity'),
            ('\U0001F4CA', 'Success Rate Heatmap', 'Parameter sweep success rate heatmaps'),
            ('\U0001F4CF', 'Landing Accuracy', 'Distance-based accuracy distribution and scatter'),
            ('\U0001F3AF', 'Landing Top-Down', 'Top-down view of landing positions'),
            ('\U0001F4C8', 'Monte Carlo', 'Multi-parameter distribution analysis'),
        ]

        self._graph_buttons = {}
        for icon, label, desc in builtin_graphs:
            btn = GraphButton(icon, label, desc)
            btn.clicked.connect(partial(self._on_builtin_graph, label))
            left_layout.addWidget(btn)
            self._graph_buttons[label] = btn

        # Extension graphs container (populated after extensions load)
        self._ext_graph_container = QWidget()
        left_layout.addWidget(self._ext_graph_container)

        left_layout.addStretch()

        # Filters section
        filter_group = QGroupBox('Filters')
        filter_layout = QVBoxLayout(filter_group)

        self._yscale_combo = QComboBox()
        self._yscale_combo.addItems(['Log', 'Linear'])
        self._yscale_combo.setCurrentText(self.prefs.get('plot', 'yscale_velocity', 'Log'))
        self._yscale_combo.currentTextChanged.connect(self._on_yscale_changed)
        yscale_row = QHBoxLayout()
        yscale_row.addWidget(QLabel('Y-Scale:'))
        yscale_row.addWidget(self._yscale_combo)
        filter_layout.addLayout(yscale_row)

        self._type_filter = QComboBox()
        self._type_filter.addItems(['All', 'ML', 'Optimization'])
        self._type_filter.currentTextChanged.connect(self._apply_filters)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel('Type:'))
        type_row.addWidget(self._type_filter)
        filter_layout.addLayout(type_row)

        self._success_filter = QComboBox()
        self._success_filter.addItems(['All', 'True', 'False'])
        self._success_filter.currentTextChanged.connect(self._apply_filters)
        success_row = QHBoxLayout()
        success_row.addWidget(QLabel('Success:'))
        success_row.addWidget(self._success_filter)
        filter_layout.addLayout(success_row)

        self._filter_info = QLabel('')
        self._filter_info.setProperty('class', 'muted')
        filter_layout.addWidget(self._filter_info)

        left_layout.addWidget(filter_group)

        page_layout.addWidget(left_panel)

        # Center: graph canvas
        center_frame = QFrame()
        center_frame.setObjectName('graph-container')
        self._graph_layout = QVBoxLayout(center_frame)
        self._graph_layout.setContentsMargins(8, 8, 8, 8)

        # Welcome placeholder
        self._graph_welcome = QLabel(
            '\U0001F4CA\n\nSelect a graph from the left panel\nor load data first'
        )
        self._graph_welcome.setAlignment(Qt.AlignCenter)
        self._graph_welcome.setProperty('class', 'muted')
        self._graph_welcome.setFont(QFont('Segoe UI', 14))
        self._graph_layout.addWidget(self._graph_welcome)

        page_layout.addWidget(center_frame, stretch=1)

        # Right panel: graph options
        right_panel = QFrame()
        right_panel.setObjectName('right-panel')
        right_panel.setFixedWidth(220)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        opts_title = QLabel('Graph Options')
        opts_title.setProperty('class', 'subheading')
        right_layout.addWidget(opts_title)

        # Marker size slider
        right_layout.addWidget(QLabel('Marker Size'))
        self._marker_slider = QSlider(Qt.Horizontal)
        self._marker_slider.setRange(10, 200)
        self._marker_slider.setValue(self.prefs.get('plot', 'marker_size', 50))
        self._marker_slider.valueChanged.connect(self._on_marker_size_changed)
        right_layout.addWidget(self._marker_slider)
        self._marker_label = QLabel(str(self._marker_slider.value()))
        self._marker_label.setProperty('class', 'muted')
        right_layout.addWidget(self._marker_label)

        # Alpha slider
        right_layout.addWidget(QLabel('Point Alpha'))
        self._alpha_slider = QSlider(Qt.Horizontal)
        self._alpha_slider.setRange(10, 100)
        self._alpha_slider.setValue(int(self.prefs.get('plot', 'point_alpha', 0.6) * 100))
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        right_layout.addWidget(self._alpha_slider)
        self._alpha_label = QLabel(f'{self._alpha_slider.value() / 100:.2f}')
        self._alpha_label.setProperty('class', 'muted')
        right_layout.addWidget(self._alpha_label)

        # Grid checkbox
        self._grid_cb = QCheckBox('Show Grid')
        self._grid_cb.setChecked(self.prefs.get('plot', 'show_grid', True))
        self._grid_cb.toggled.connect(self._on_grid_changed)
        right_layout.addWidget(self._grid_cb)

        right_layout.addSpacing(16)

        # Export button
        export_btn = QPushButton('\U0001F4BE  Export Plot')
        export_btn.setProperty('class', 'primary')
        export_btn.clicked.connect(self._export_plot)
        right_layout.addWidget(export_btn)

        right_layout.addStretch()
        page_layout.addWidget(right_panel)

        self.content_stack.addWidget(page)

    def _on_builtin_graph(self, graph_label):
        """Handle built-in graph selection."""
        dispatch = {
            'Velocity vs Intensity': self._plot_velocity_vs_intensity,
            'Success Rate Heatmap': self._plot_success_heatmap,
            'Landing Accuracy': self._plot_landing_accuracy,
            'Landing Top-Down': self._plot_landing_topdown,
            'Monte Carlo': self._plot_monte_carlo,
        }
        func = dispatch.get(graph_label)
        if func:
            func()

    # ------------------------------------------------------------------
    # EXTENSIONS PAGE
    # ------------------------------------------------------------------

    def _build_extensions_page(self):
        """Build the Extensions manager page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 20, 30, 20)

        header = QLabel('\U0001F9E9  Extensions')
        header.setProperty('class', 'heading')
        header.setFont(QFont('Segoe UI', 20, QFont.Bold))
        layout.addWidget(header)

        # Tab widget
        self.ext_tabs = QTabWidget()
        layout.addWidget(self.ext_tabs)

        # Installed tab
        self._build_installed_tab()
        # Store tab
        self._build_store_tab()
        # Develop tab
        self._build_develop_tab()

        self.content_stack.addWidget(page)

    def _build_installed_tab(self):
        """Build the Installed extensions tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)

        # Scroll area for extension cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self._installed_list_layout = QVBoxLayout(scroll_content)
        self._installed_list_layout.setSpacing(8)
        self._installed_list_layout.setContentsMargins(0, 0, 8, 0)
        self._installed_list_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._refresh_installed_list()
        self.ext_tabs.addTab(tab, 'Installed')

    def _refresh_installed_list(self):
        """Refresh the installed extensions list."""
        layout = self._installed_list_layout
        # Clear existing cards (keep the stretch)
        while layout.count() > 1:
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        manifests = self.ext_manager.list_installed()
        if not manifests:
            empty = QLabel('No extensions installed yet.\nVisit the Store tab or install from file.')
            empty.setProperty('class', 'muted')
            empty.setAlignment(Qt.AlignCenter)
            layout.insertWidget(0, empty)
            return

        for manifest in manifests:
            is_enabled = self.ext_manager.is_enabled(manifest.id)
            card = ExtensionCard(manifest, is_enabled, manifest.bundled)
            card.toggled.connect(self._on_ext_toggled)
            card.uninstall_clicked.connect(self._on_ext_uninstall)
            layout.insertWidget(layout.count() - 1, card)

    def _on_ext_toggled(self, ext_id, enabled):
        if enabled:
            self.ext_manager.enable(ext_id)
        else:
            self.ext_manager.disable(ext_id)
        self.status_bar.showMessage(
            f'Extension "{ext_id}" {"enabled" if enabled else "disabled"}. Restart for full effect.'
        )

    def _on_ext_uninstall(self, ext_id):
        reply = QMessageBox.question(
            self, 'Confirm Uninstall',
            f'Uninstall extension "{ext_id}"?',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.ext_manager.uninstall(ext_id):
                self._refresh_installed_list()
                self.status_bar.showMessage(f'Extension "{ext_id}" uninstalled')
            else:
                QMessageBox.warning(self, 'Error', f'Could not uninstall "{ext_id}".')

    def _build_store_tab(self):
        """Build the Store catalog tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 12, 8, 8)

        # Filter bar
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel('Filter:'))
        self._store_type_filter = QComboBox()
        self._store_type_filter.addItems(['All', 'plotvisual', 'hexakinetic', 'hexavisual', 'mission_control'])
        self._store_type_filter.currentTextChanged.connect(self._refresh_store_list)
        filter_row.addWidget(self._store_type_filter)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self._store_list_layout = QVBoxLayout(scroll_content)
        self._store_list_layout.setSpacing(8)
        self._store_list_layout.setContentsMargins(0, 0, 8, 0)
        self._store_list_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._refresh_store_list()
        self.ext_tabs.addTab(tab, 'Store')

    def _refresh_store_list(self):
        layout = self._store_list_layout
        while layout.count() > 1:
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        type_filter = (
            self._store_type_filter.currentText()
            if hasattr(self, '_store_type_filter')
            else 'All'
        )
        entries = self.ext_catalog.get_all()

        installed_ids = {m.id for m in self.ext_manager.list_installed()}

        for entry in entries:
            if type_filter != 'All' and entry.extension_type != type_filter:
                continue
            is_installed = entry.id in installed_ids
            card = CatalogCard(entry, is_installed)
            card.install_clicked.connect(self._on_catalog_install)
            layout.insertWidget(layout.count() - 1, card)

    def _on_catalog_install(self, ext_id):
        entry = self.ext_catalog.get_by_id(ext_id)
        if not entry:
            return
        if entry.download_url:
            QMessageBox.information(
                self, 'Download Required',
                f'Download the extension from:\n{entry.download_url}\n\n'
                'Then install via the Develop tab.'
            )

    def _build_develop_tab(self):
        """Build the Develop tab for extension developers."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Install from file
        install_card = QFrame()
        install_card.setProperty('class', 'card')
        ic_layout = QVBoxLayout(install_card)
        ic_layout.setContentsMargins(20, 16, 20, 16)
        ic_layout.addWidget(QLabel('Install Extension from File'))
        ic_desc = QLabel('Install a .vortexext archive or select an extension directory.')
        ic_desc.setProperty('class', 'muted')
        ic_desc.setWordWrap(True)
        ic_layout.addWidget(ic_desc)

        btn_row = QHBoxLayout()
        install_file_btn = QPushButton('\U0001F4E6  Install from .vortexext')
        install_file_btn.setProperty('class', 'primary')
        install_file_btn.clicked.connect(self._install_from_file)
        btn_row.addWidget(install_file_btn)

        install_dir_btn = QPushButton('\U0001F4C1  Install from Directory')
        install_dir_btn.clicked.connect(self._install_from_dir)
        btn_row.addWidget(install_dir_btn)
        btn_row.addStretch()
        ic_layout.addLayout(btn_row)
        layout.addWidget(install_card)

        # Pack extension tool
        pack_card = QFrame()
        pack_card.setProperty('class', 'card')
        pc_layout = QVBoxLayout(pack_card)
        pc_layout.setContentsMargins(20, 16, 20, 16)
        pc_layout.addWidget(QLabel('Package Extension'))
        pc_desc = QLabel(
            'Pack an extension directory into a distributable .vortexext file.\n'
            'The directory must contain a manifest.json.'
        )
        pc_desc.setProperty('class', 'muted')
        pc_desc.setWordWrap(True)
        pc_layout.addWidget(pc_desc)

        pack_btn = QPushButton('\U0001F4E6  Pack Extension')
        pack_btn.clicked.connect(self._pack_extension)
        pc_layout.addWidget(pack_btn)
        layout.addWidget(pack_card)

        # Developer info
        info_card = QFrame()
        info_card.setProperty('class', 'card')
        info_layout_c = QVBoxLayout(info_card)
        info_layout_c.setContentsMargins(20, 16, 20, 16)
        info_layout_c.addWidget(QLabel('Creating Extensions'))
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(250)
        info_text.setPlainText(
            'Extension Structure:\n'
            '  my_extension/\n'
            '    manifest.json     <- Required: id, name, version, extension_type, entry_point\n'
            '    plugin.py         <- Entry point with your extension class\n'
            '\n'
            'manifest.json example:\n'
            '  {\n'
            '    "id": "my_extension",\n'
            '    "name": "My Custom Graphs",\n'
            '    "version": "1.0.0",\n'
            '    "extension_type": "plotvisual",\n'
            '    "entry_point": "plugin",\n'
            '    "author": "Your Name",\n'
            '    "description": "Custom visualization graphs"\n'
            '  }\n'
            '\n'
            'Extension Types:\n'
            '  plotvisual      - Graphs, data sources, export formats\n'
            '  hexakinetic     - Fault models, motor profiles, controllers\n'
            '  hexavisual      - 3D overlays, camera modes, effects\n'
            '  mission_control - Dashboard widgets, serial protocols\n'
            '  universal       - Works with all apps\n'
            '\n'
            'For PlotVisual extensions, subclass PlotVisualExtension from\n'
            'extensions.hooks.plotvisual and implement register_graphs() and render_graph().\n'
            '\n'
            'See EXTENSION_DEVELOPMENT.md for the full guide.'
        )
        info_layout_c.addWidget(info_text)
        layout.addWidget(info_card)

        layout.addStretch()
        self.ext_tabs.addTab(tab, 'Develop')

    def _install_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Install Extension',
            '', 'Vortex Extensions (*.vortexext);;ZIP files (*.zip);;All files (*.*)'
        )
        if not filepath:
            return
        try:
            manifest = self.ext_manager.install_from_vortexext(filepath)
            self._refresh_installed_list()
            self._refresh_store_list()
            QMessageBox.information(
                self, 'Installed',
                f'Successfully installed:\n{manifest.name} v{manifest.version}\n\n'
                'Restart PlotVisual for the extension to take effect.'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Install Failed', f'Error: {str(e)}')

    def _install_from_dir(self):
        dirpath = QFileDialog.getExistingDirectory(
            self, 'Select Extension Directory'
        )
        if not dirpath:
            return
        try:
            manifest = self.ext_manager.install_from_directory(dirpath)
            self._refresh_installed_list()
            self._refresh_store_list()
            QMessageBox.information(
                self, 'Installed',
                f'Successfully installed:\n{manifest.name} v{manifest.version}\n\n'
                'Restart PlotVisual for the extension to take effect.'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Install Failed', f'Error: {str(e)}')

    def _pack_extension(self):
        dirpath = QFileDialog.getExistingDirectory(
            self, 'Select Extension Directory to Pack'
        )
        if not dirpath:
            return
        try:
            from extensions.packaging import pack_extension
            output, _ = QFileDialog.getSaveFileName(
                self, 'Save .vortexext file',
                os.path.basename(dirpath) + '.vortexext',
                'Vortex Extensions (*.vortexext)'
            )
            if output:
                result = pack_extension(dirpath, output)
                QMessageBox.information(
                    self, 'Packed',
                    f'Extension packed successfully:\n{result}'
                )
        except Exception as e:
            QMessageBox.critical(self, 'Pack Failed', f'Error: {str(e)}')

    # ------------------------------------------------------------------
    # SETTINGS PAGE
    # ------------------------------------------------------------------

    def _build_settings_page(self):
        """Build the Settings page."""
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(16)

        header = QLabel('\u2699  Settings')
        header.setProperty('class', 'heading')
        header.setFont(QFont('Segoe UI', 20, QFont.Bold))
        layout.addWidget(header)

        # Appearance section
        appearance_group = QGroupBox('Appearance')
        app_layout = QVBoxLayout(appearance_group)

        # Theme toggle
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel('Theme:'))
        self._settings_theme_combo = QComboBox()
        self._settings_theme_combo.addItems(['Dark', 'Light'])
        self._settings_theme_combo.setCurrentText(self._theme_mode.capitalize())
        self._settings_theme_combo.currentTextChanged.connect(self._on_settings_theme_changed)
        theme_row.addWidget(self._settings_theme_combo)
        theme_row.addStretch()
        app_layout.addLayout(theme_row)

        # Accent color
        accent_row = QHBoxLayout()
        accent_row.addWidget(QLabel('Accent Color:'))
        self._accent_combo = QComboBox()
        self._accent_combo.addItems(list(ACCENT_PRESETS.keys()))
        # Select current accent
        current_name = 'Blue'
        for name, color in ACCENT_PRESETS.items():
            if color == self._accent_color:
                current_name = name
                break
        self._accent_combo.setCurrentText(current_name)
        self._accent_combo.currentTextChanged.connect(self._on_accent_changed)
        accent_row.addWidget(self._accent_combo)
        accent_row.addStretch()
        app_layout.addLayout(accent_row)

        # Font scale
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel('Font Size:'))
        self._font_spin = QSpinBox()
        self._font_spin.setRange(10, 20)
        self._font_spin.setValue(self.prefs.get('appearance', 'font_scale', 13))
        self._font_spin.valueChanged.connect(self._on_font_changed)
        font_row.addWidget(self._font_spin)
        font_row.addStretch()
        app_layout.addLayout(font_row)

        layout.addWidget(appearance_group)

        # Plot defaults section
        plot_group = QGroupBox('Plot Defaults')
        plot_layout = QVBoxLayout(plot_group)

        # Default Y-scale
        yscale_row = QHBoxLayout()
        yscale_row.addWidget(QLabel('Default Y-Scale:'))
        self._settings_yscale = QComboBox()
        self._settings_yscale.addItems(['Log', 'Linear'])
        self._settings_yscale.setCurrentText(self.prefs.get('plot', 'yscale_velocity', 'Log'))
        self._settings_yscale.currentTextChanged.connect(
            lambda v: (self.prefs.set('plot', 'yscale_velocity', v), self.prefs.save())
        )
        yscale_row.addWidget(self._settings_yscale)
        yscale_row.addStretch()
        plot_layout.addLayout(yscale_row)

        # Marker size
        ms_row = QHBoxLayout()
        ms_row.addWidget(QLabel('Marker Size:'))
        self._settings_marker = QSpinBox()
        self._settings_marker.setRange(10, 200)
        self._settings_marker.setValue(self.prefs.get('plot', 'marker_size', 50))
        self._settings_marker.valueChanged.connect(
            lambda v: (self.prefs.set('plot', 'marker_size', v), self.prefs.save())
        )
        ms_row.addWidget(self._settings_marker)
        ms_row.addStretch()
        plot_layout.addLayout(ms_row)

        # Alpha
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel('Point Alpha:'))
        self._settings_alpha = QDoubleSpinBox()
        self._settings_alpha.setRange(0.1, 1.0)
        self._settings_alpha.setSingleStep(0.05)
        self._settings_alpha.setValue(self.prefs.get('plot', 'point_alpha', 0.6))
        self._settings_alpha.valueChanged.connect(
            lambda v: (self.prefs.set('plot', 'point_alpha', v), self.prefs.save())
        )
        alpha_row.addWidget(self._settings_alpha)
        alpha_row.addStretch()
        plot_layout.addLayout(alpha_row)

        # Histogram bins
        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel('Histogram Bins:'))
        self._settings_bins = QSpinBox()
        self._settings_bins.setRange(5, 100)
        self._settings_bins.setValue(self.prefs.get('plot', 'hist_bins', 30))
        self._settings_bins.valueChanged.connect(
            lambda v: (self.prefs.set('plot', 'hist_bins', v), self.prefs.save())
        )
        bins_row.addWidget(self._settings_bins)
        bins_row.addStretch()
        plot_layout.addLayout(bins_row)

        # Export DPI
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel('Export DPI:'))
        self._settings_dpi = QSpinBox()
        self._settings_dpi.setRange(72, 600)
        self._settings_dpi.setValue(self.prefs.get('plot', 'export_dpi', 300))
        self._settings_dpi.valueChanged.connect(
            lambda v: (self.prefs.set('plot', 'export_dpi', v), self.prefs.save())
        )
        dpi_row.addWidget(self._settings_dpi)
        dpi_row.addStretch()
        plot_layout.addLayout(dpi_row)

        # Grid
        grid_row = QHBoxLayout()
        self._settings_grid = QCheckBox('Show Grid by default')
        self._settings_grid.setChecked(self.prefs.get('plot', 'show_grid', True))
        self._settings_grid.toggled.connect(
            lambda v: (self.prefs.set('plot', 'show_grid', v), self.prefs.save())
        )
        grid_row.addWidget(self._settings_grid)
        grid_row.addStretch()
        plot_layout.addLayout(grid_row)

        layout.addWidget(plot_group)

        # Data section
        data_group = QGroupBox('Data')
        data_layout = QVBoxLayout(data_group)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel('Default Results Directory:'))
        self._results_dir_edit = QLineEdit(RESULTS_DIR)
        dir_row.addWidget(self._results_dir_edit)
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self._browse_results_dir)
        dir_row.addWidget(browse_btn)
        data_layout.addLayout(dir_row)

        remember_cb = QCheckBox('Remember last loaded file')
        remember_cb.setChecked(self.prefs.get('general', 'remember_last_source', True))
        remember_cb.toggled.connect(
            lambda v: (self.prefs.set('general', 'remember_last_source', v), self.prefs.save())
        )
        data_layout.addWidget(remember_cb)

        clear_btn = QPushButton('Clear Recent Files')
        clear_btn.setFixedWidth(150)
        clear_btn.clicked.connect(self._clear_recents)
        data_layout.addWidget(clear_btn)

        layout.addWidget(data_group)

        # About section
        about_group = QGroupBox('About')
        about_layout = QVBoxLayout(about_group)
        about_text = QLabel(
            f'{APP_TITLE} v{APP_VERSION}\n'
            f'Advanced Visualization Suite for Project Vortex\n\n'
            f'Part of Vortex Desktop Suite by Mishra Rockets\n'
            f'\u00A9 2026 Agastya Mishra'
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        layout.addWidget(about_group)

        layout.addStretch()
        scroll.setWidget(inner)

        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        self.content_stack.addWidget(page)

    def _on_settings_theme_changed(self, text):
        self._theme_mode = text.lower()
        self.prefs.set('appearance', 'theme', text)
        self.prefs.save()
        self._apply_theme()
        self._update_theme_button()

    def _on_accent_changed(self, name):
        color = ACCENT_PRESETS.get(name, ACCENT_BLUE)
        self._accent_color = color
        self.prefs.set('appearance', 'accent_color', color)
        self.prefs.save()
        self._apply_theme()

    def _on_font_changed(self, size):
        self.prefs.set('appearance', 'font_scale', size)
        self.prefs.save()

    def _browse_results_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Select Results Directory', RESULTS_DIR)
        if d:
            self._results_dir_edit.setText(d)

    def _clear_recents(self):
        self.prefs.set('general', 'recent_sources', [])
        self.prefs.save()
        self._refresh_recent_list()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Data CSV',
            RESULTS_DIR,
            'CSV files (*.csv);;All files (*.*)'
        )
        if filepath:
            self._load_csv_file(filepath)

    def _load_csv_file(self, filepath):
        try:
            df = pd.read_csv(filepath)
            cols = set(df.columns)

            if {'Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'}.issubset(cols):
                self.data_store.set('trajectory', df, {'source': filepath})
                self.status_bar.showMessage(f'Loaded trajectory: {len(df)} rows')
            elif any('ignition' in c.lower() for c in cols) and any('success' in c.lower() for c in cols):
                self.data_store.set('optimization', df, {'source': filepath})
                self.status_bar.showMessage(f'Loaded optimization: {len(df)} rows')
            else:
                self.data_store.set('legacy', df, {'source': filepath})
                self.data = df
                self.filtered_data = df.copy()

            self.prefs.add_recent(filepath)
            self._refresh_recent_list()
            self._update_data_summary()
            self.status_bar.showMessage(f'Loaded {len(df)} records from {os.path.basename(filepath)}')

        except Exception as e:
            QMessageBox.critical(self, 'Load Error', f'Failed to load:\n{str(e)}')

    def _load_trajectory(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Trajectory CSV', RESULTS_DIR, 'CSV files (*.csv);;All (*.*)'
        )
        if filepath:
            try:
                self.data_store.load_trajectory_csv(filepath)
                self.prefs.add_recent(filepath)
                self._refresh_recent_list()
                self._update_data_summary()
                self.status_bar.showMessage(f'Loaded trajectory from {os.path.basename(filepath)}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))

    def _load_results_dir(self):
        dirpath = QFileDialog.getExistingDirectory(self, 'Select Results Directory', RESULTS_DIR)
        if dirpath:
            try:
                self.data_store.load_results_directory(dirpath)
                self._update_data_summary()
                self.status_bar.showMessage(f'Loaded results from {os.path.basename(dirpath)}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))

    def _load_sample_data(self):
        """Generate and load sample data for all graphs."""
        try:
            # Let each extension populate the data store
            for ext in self.ext_registry.get_extensions():
                try:
                    if hasattr(ext, 'generate_sample_data'):
                        ext.generate_sample_data(self.data_store)
                    elif hasattr(ext, 'legacy_plugin'):
                        lp = ext.legacy_plugin
                        if hasattr(lp, 'generate_sample_data'):
                            lp.generate_sample_data(self.data_store)
                except Exception:
                    traceback.print_exc()

            # Generate built-in data if not present
            if not self.data_store.has('legacy'):
                sample_dir = os.path.join(RESULTS_DIR, 'sample_data')
                os.makedirs(sample_dir, exist_ok=True)
                sample_file = os.path.join(sample_dir, 'sample_visualization_data.csv')
                self._generate_sample_data(sample_file)
                self.data_store.load_csv(sample_file, 'legacy')

            legacy = self.data_store.get('legacy')
            if legacy is not None:
                self.data = legacy
                self.filtered_data = legacy.copy()

            self._update_data_summary()
            self.status_bar.showMessage('Sample data loaded for all graphs')

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate sample data:\n{str(e)}')
            traceback.print_exc()

    def _generate_sample_data(self, output_file):
        """Generate realistic sample data for built-in visualizations."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.random.seed(42)
        records = []

        for i in range(150):
            dry_mass = np.random.uniform(45, 65)
            propellant_mass = np.random.uniform(8, 12)
            diameter = np.random.uniform(0.3, 0.4)
            thrust_avg = np.random.uniform(900, 1100)
            wind_speed = np.random.uniform(0, 12)
            drag_coef = np.random.uniform(0.45, 0.65)
            air_density = np.random.uniform(1.15, 1.25)
            initial_alt = np.random.uniform(900, 1300)
            initial_vel = np.random.uniform(45, 65)
            fault_intensity = np.random.beta(2, 4) * 0.98

            base_landing_vel = np.random.uniform(0.5, 1.0)
            mass_impact = (dry_mass + propellant_mass - 55) * 0.04
            wind_impact = wind_speed * 0.12

            # Optimization agent
            opt_fault = (fault_intensity ** 3.8) * 45.0
            opt_vel = base_landing_vel + opt_fault + mass_impact + wind_impact
            opt_vel += np.random.normal(0, 0.1 + fault_intensity * 8)
            opt_vel = max(0.1, opt_vel)
            opt_base = 4 + wind_speed * 0.5 + (fault_intensity ** 2.2) * 50
            opt_err = np.random.rayleigh(opt_base)
            opt_x, opt_y = np.random.normal(0, opt_err), np.random.normal(0, opt_err)
            records.append({
                'Type': 'Optimization', 'Landing Velocity': opt_vel,
                'Success': opt_vel < 2.0, 'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel,
                'Landing X': opt_x, 'Landing Y': opt_y,
                'Landing Distance': np.sqrt(opt_x**2 + opt_y**2)
            })

            # ML agent
            ml_base = base_landing_vel + (fault_intensity * 6.5)
            ml_vel = ml_base + (mass_impact * 0.4) + (wind_impact * 0.3)
            ml_vel += np.random.normal(0, 0.1 + fault_intensity * 0.3)
            if np.random.rand() < 0.97:
                ml_vel = min(ml_vel, 2.0)
            else:
                ml_vel = min(ml_vel, 35.0)
            ml_vel = max(0.1, ml_vel)
            ml_hbase = 2 + wind_speed * 0.15 + fault_intensity * 5
            ml_err = np.random.rayleigh(ml_hbase)
            ml_x, ml_y = np.random.normal(0, ml_err), np.random.normal(0, ml_err)
            records.append({
                'Type': 'ML', 'Landing Velocity': ml_vel,
                'Success': ml_vel < 2.0, 'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel,
                'Landing X': ml_x, 'Landing Y': ml_y,
                'Landing Distance': np.sqrt(ml_x**2 + ml_y**2)
            })

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)

    def _update_data_summary(self):
        info = self.data_store.summary()
        self.data_summary_label.setText(info if info else 'No data loaded yet.')
        if self.filtered_data is not None:
            n = len(self.filtered_data)
            total = len(self.data) if self.data is not None else n
            self._filter_info.setText(f'{n} / {total} records')

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_filters(self):
        if self.data is None:
            return
        self.filtered_data = self.data.copy()

        type_val = self._type_filter.currentText()
        if type_val != 'All' and 'Type' in self.filtered_data.columns:
            self.filtered_data = self.filtered_data[self.filtered_data['Type'] == type_val]

        success_val = self._success_filter.currentText()
        if success_val != 'All' and 'Success' in self.filtered_data.columns:
            self.filtered_data = self.filtered_data[self.filtered_data['Success'] == (success_val == 'True')]

        n = len(self.filtered_data)
        total = len(self.data)
        self._filter_info.setText(f'{n} / {total} records')
        self.status_bar.showMessage(f'Filtered: {n} / {total} records')

        # Refresh current graph
        self._refresh_current_graph()

    def _on_yscale_changed(self, val):
        self.prefs.set('plot', 'yscale_velocity', val)
        self.prefs.save()
        self._refresh_current_graph()

    def _on_marker_size_changed(self, val):
        self._marker_label.setText(str(val))
        self.prefs.set('plot', 'marker_size', val)
        self._refresh_current_graph()

    def _on_alpha_changed(self, val):
        self._alpha_label.setText(f'{val / 100:.2f}')
        self.prefs.set('plot', 'point_alpha', val / 100)
        self._refresh_current_graph()

    def _on_grid_changed(self, checked):
        self.prefs.set('plot', 'show_grid', checked)
        self._refresh_current_graph()

    def _refresh_current_graph(self):
        if self.last_view is None:
            return
        dispatch = {
            'velocity': self._plot_velocity_vs_intensity,
            'heatmap': self._plot_success_heatmap,
            'accuracy': self._plot_landing_accuracy,
            'topdown': self._plot_landing_topdown,
            'montecarlo': self._plot_monte_carlo,
        }
        func = dispatch.get(self.last_view)
        if func:
            func()
        elif self.last_view.startswith('plugin:'):
            key = self.last_view[7:]
            self._render_plugin_graph(key)

    # ------------------------------------------------------------------
    # Graph rendering helpers
    # ------------------------------------------------------------------

    def _clear_graph_area(self):
        """Remove all widgets from the graph layout."""
        while self._graph_layout.count():
            item = self._graph_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.current_canvas = None
        self.current_figure = None

    def _embed_figure(self, fig):
        """Embed a matplotlib Figure in the graph area."""
        self._clear_graph_area()
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, None)
        self._graph_layout.addWidget(toolbar)
        self._graph_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_figure = fig
        canvas.draw()

    def _get_plot_params(self):
        return {
            'marker_size': self._marker_slider.value(),
            'alpha': self._alpha_slider.value() / 100,
            'show_grid': self._grid_cb.isChecked(),
            'yscale': self._yscale_combo.currentText(),
        }

    def _check_data(self, cols=None):
        if self.filtered_data is None or len(self.filtered_data) == 0:
            QMessageBox.warning(self, 'No Data', 'Please load data first.')
            return False
        if cols:
            missing = [c for c in cols if c not in self.filtered_data.columns]
            if missing:
                QMessageBox.warning(self, 'Missing Columns', f'Required columns: {", ".join(missing)}')
                return False
        return True

    # ------------------------------------------------------------------
    # Built-in graphs
    # ------------------------------------------------------------------

    def _plot_velocity_vs_intensity(self):
        if not self._check_data(['Landing Velocity', 'Total Fault Intensity']):
            return
        self.last_view = 'velocity'
        params = self._get_plot_params()
        style = get_matplotlib_style(self._theme_mode)

        fig = Figure(figsize=(12, 7), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))
        ax = fig.add_subplot(111)

        with plt.style.context(style):
            ax.set_facecolor(style.get('axes.facecolor', '#22222a'))
            types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
            colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}

            for t in types:
                if t not in colors:
                    continue
                subset = self.filtered_data[self.filtered_data['Type'] == t] if 'Type' in self.filtered_data.columns else self.filtered_data
                ax.scatter(subset['Total Fault Intensity'], subset['Landing Velocity'],
                          c=colors[t], s=params['marker_size'], alpha=params['alpha'],
                          label=t, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Total Fault Intensity', fontsize=12, fontweight='bold', color=style.get('axes.labelcolor', '#c0c0c8'))
            ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, fontweight='bold', color=style.get('axes.labelcolor', '#c0c0c8'))
            ax.set_title('Landing Velocity vs Fault Intensity', fontsize=14, fontweight='bold', color=style.get('text.color', '#e0e0e0'))

            if params['yscale'] == 'Log':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
                ax.yaxis.set_minor_formatter(NullFormatter())
                y_min = max(0.1, self.filtered_data['Landing Velocity'].min() * 0.5)
                y_max = self.filtered_data['Landing Velocity'].max() * 2.0
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_yscale('linear')
                y_max = max(5.0, self.filtered_data['Landing Velocity'].max() * 1.1)
                ax.set_ylim(0, y_max)
                if y_max > 50: ax.yaxis.set_major_locator(MultipleLocator(10))
                elif y_max > 20: ax.yaxis.set_major_locator(MultipleLocator(5))
                else: ax.yaxis.set_major_locator(MultipleLocator(2))

            if params['show_grid']:
                ax.grid(True, alpha=0.3, linestyle='--', which='both', color=style.get('grid.color', '#2e2e36'))

            ax.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Success Threshold (2 m/s)')
            ax.legend(fontsize=11, frameon=True, shadow=True,
                     facecolor=style.get('legend.facecolor', '#22222a'),
                     edgecolor=style.get('legend.edgecolor', '#3a3a42'),
                     labelcolor=style.get('legend.labelcolor', '#e0e0e0'))
            ax.tick_params(colors=style.get('xtick.color', '#808088'))

        fig.tight_layout()
        self._embed_figure(fig)
        self.status_bar.showMessage(f'Showing: Landing Velocity vs Fault Intensity ({params["yscale"]} scale)')

    def _plot_success_heatmap(self):
        if not self._check_data(['Success']):
            return
        self.last_view = 'heatmap'
        style = get_matplotlib_style(self._theme_mode)
        params = self._get_plot_params()

        fig = Figure(figsize=(12, 8), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))

        pairs = [
            ('Wind Speed', 'Total Fault Intensity'),
            ('Initial Altitude', 'Total Fault Intensity'),
            ('Dry Mass', 'Wind Speed'),
            ('Initial Velocity', 'Total Fault Intensity'),
        ]

        for idx, (x_col, y_col) in enumerate(pairs):
            ax = fig.add_subplot(2, 2, idx + 1)
            ax.set_facecolor(style.get('axes.facecolor', '#22222a'))
            self._render_heatmap(ax, x_col, y_col, 'Success', style)

        fig.suptitle('Success Rate Heatmaps', fontsize=14, fontweight='bold',
                     color=style.get('text.color', '#e0e0e0'))
        fig.tight_layout()
        self._embed_figure(fig)
        self.status_bar.showMessage('Showing: Success Rate Heatmaps')

    def _render_heatmap(self, ax, x_col, y_col, z_col, style):
        if x_col not in self.filtered_data.columns or y_col not in self.filtered_data.columns:
            ax.text(0.5, 0.5, f'Missing: {x_col} or {y_col}',
                   ha='center', va='center', transform=ax.transAxes,
                   color=style.get('text.color', '#e0e0e0'))
            return

        n_bins = 10
        x_edges = np.linspace(self.filtered_data[x_col].min(), self.filtered_data[x_col].max(), n_bins + 1)
        y_edges = np.linspace(self.filtered_data[y_col].min(), self.filtered_data[y_col].max(), n_bins + 1)

        heatmap = np.zeros((n_bins, n_bins))
        counts = np.zeros((n_bins, n_bins))

        for _, row in self.filtered_data.iterrows():
            xi = np.clip(np.searchsorted(x_edges, row[x_col]) - 1, 0, n_bins - 1)
            yi = np.clip(np.searchsorted(y_edges, row[y_col]) - 1, 0, n_bins - 1)
            counts[yi, xi] += 1
            if row[z_col]:
                heatmap[yi, xi] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)

        im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', origin='lower',
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, vmax=1)
        ax.set_xlabel(x_col, fontsize=9, color=style.get('axes.labelcolor', '#c0c0c8'))
        ax.set_ylabel(y_col, fontsize=9, color=style.get('axes.labelcolor', '#c0c0c8'))
        ax.set_title(f'{z_col} Rate', fontsize=10, fontweight='bold',
                    color=style.get('text.color', '#e0e0e0'))
        ax.tick_params(colors=style.get('xtick.color', '#808088'))
        fig = ax.get_figure()
        fig.colorbar(im, ax=ax, label='Success Rate')

    def _plot_landing_accuracy(self):
        if not self._check_data(['Landing Distance']):
            return
        self.last_view = 'accuracy'
        style = get_matplotlib_style(self._theme_mode)
        params = self._get_plot_params()

        fig = Figure(figsize=(12, 6), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(style.get('axes.facecolor', '#22222a'))
        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}

        for t in types:
            if t not in colors:
                continue
            subset = self.filtered_data[self.filtered_data['Type'] == t] if 'Type' in self.filtered_data.columns else self.filtered_data
            ax1.hist(subset['Landing Distance'], bins=params.get('hist_bins', 30), alpha=params['alpha'],
                    color=colors[t], label=t, edgecolor='black')
        ax1.set_xlabel('Landing Distance (m)', fontsize=11, fontweight='bold',
                       color=style.get('axes.labelcolor', '#c0c0c8'))
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold',
                       color=style.get('axes.labelcolor', '#c0c0c8'))
        ax1.set_title('Accuracy Distribution', fontsize=12, fontweight='bold',
                      color=style.get('text.color', '#e0e0e0'))
        ax1.legend(facecolor=style.get('legend.facecolor', '#22222a'),
                  edgecolor=style.get('legend.edgecolor', '#3a3a42'),
                  labelcolor=style.get('legend.labelcolor', '#e0e0e0'))
        if params['show_grid']:
            ax1.grid(True, alpha=0.3, linestyle='--', color=style.get('grid.color', '#2e2e36'))
        ax1.tick_params(colors=style.get('xtick.color', '#808088'))

        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(style.get('axes.facecolor', '#22222a'))
        for t in types:
            if t not in colors:
                continue
            subset = self.filtered_data[self.filtered_data['Type'] == t] if 'Type' in self.filtered_data.columns else self.filtered_data
            ax2.scatter(subset['Total Fault Intensity'], subset['Landing Distance'],
                       c=colors[t], s=params['marker_size'], alpha=params['alpha'],
                       label=t, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Total Fault Intensity', fontsize=11, fontweight='bold',
                       color=style.get('axes.labelcolor', '#c0c0c8'))
        ax2.set_ylabel('Landing Distance (m)', fontsize=11, fontweight='bold',
                       color=style.get('axes.labelcolor', '#c0c0c8'))
        ax2.set_title('Accuracy vs Fault Intensity', fontsize=12, fontweight='bold',
                      color=style.get('text.color', '#e0e0e0'))

        if params['yscale'] == 'Log':
            ax2.set_yscale('log')
            ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax2.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax2.set_yscale('linear')

        ax2.legend(facecolor=style.get('legend.facecolor', '#22222a'),
                  edgecolor=style.get('legend.edgecolor', '#3a3a42'),
                  labelcolor=style.get('legend.labelcolor', '#e0e0e0'))
        if params['show_grid']:
            ax2.grid(True, alpha=0.3, linestyle='--', which='both', color=style.get('grid.color', '#2e2e36'))
        ax2.tick_params(colors=style.get('xtick.color', '#808088'))

        fig.tight_layout()
        self._embed_figure(fig)
        self.status_bar.showMessage(f'Showing: Landing Accuracy ({params["yscale"]} scale)')

    def _plot_landing_topdown(self):
        if not self._check_data(['Landing X', 'Landing Y']):
            return
        self.last_view = 'topdown'
        style = get_matplotlib_style(self._theme_mode)
        params = self._get_plot_params()

        fig = Figure(figsize=(10, 10), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.get('axes.facecolor', '#22222a'))

        target = plt.Circle((0, 0), 2, color='red', fill=False, linewidth=3,
                             label='Target (2m radius)', zorder=10)
        ax.add_patch(target)

        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}
        markers = {'ML': 'o', 'Optimization': 's', 'All': 'o'}

        for t in types:
            if t not in colors:
                continue
            subset = self.filtered_data[self.filtered_data['Type'] == t] if 'Type' in self.filtered_data.columns else self.filtered_data
            ax.scatter(subset['Landing X'], subset['Landing Y'],
                      c=colors[t], marker=markers.get(t, 'o'),
                      s=params['marker_size'], alpha=params['alpha'],
                      label=t, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold',
                      color=style.get('axes.labelcolor', '#c0c0c8'))
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold',
                      color=style.get('axes.labelcolor', '#c0c0c8'))
        ax.set_title('Landing Positions \u2014 Top-Down View', fontsize=14, fontweight='bold',
                     color=style.get('text.color', '#e0e0e0'))
        ax.set_aspect('equal')
        if params['show_grid']:
            ax.grid(True, alpha=0.3, linestyle='--', color=style.get('grid.color', '#2e2e36'))
        ax.legend(fontsize=10,
                 facecolor=style.get('legend.facecolor', '#22222a'),
                 edgecolor=style.get('legend.edgecolor', '#3a3a42'),
                 labelcolor=style.get('legend.labelcolor', '#e0e0e0'))
        ax.axhline(y=0, color=style.get('grid.color', '#2e2e36'), linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color=style.get('grid.color', '#2e2e36'), linewidth=0.5, alpha=0.5)
        ax.tick_params(colors=style.get('xtick.color', '#808088'))

        fig.tight_layout()
        self._embed_figure(fig)
        self.status_bar.showMessage('Showing: Landing Positions (Top-Down)')

    def _plot_monte_carlo(self):
        if not self._check_data():
            return
        self.last_view = 'montecarlo'
        style = get_matplotlib_style(self._theme_mode)
        params = self._get_plot_params()

        fig = Figure(figsize=(12, 8), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))

        dist_cols = [
            ('Landing Velocity', 'm/s', 231),
            ('Landing Distance', 'm', 232),
            ('Total Fault Intensity', '', 233),
            ('Dry Mass', 'kg', 235),
            ('Wind Speed', 'm/s', 236),
        ]

        types_list = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}

        for col, unit, pos in dist_cols:
            if col not in self.filtered_data.columns:
                continue
            ax = fig.add_subplot(pos)
            ax.set_facecolor(style.get('axes.facecolor', '#22222a'))
            for t in types_list:
                if t not in colors:
                    continue
                subset = self.filtered_data[self.filtered_data['Type'] == t] if 'Type' in self.filtered_data.columns else self.filtered_data
                ax.hist(subset[col], bins=20, alpha=params['alpha'],
                       color=colors[t], label=t, edgecolor='black')
            xlabel = f'{col} ({unit})' if unit else col
            ax.set_xlabel(xlabel, fontsize=9, color=style.get('axes.labelcolor', '#c0c0c8'))
            ax.set_ylabel('Frequency', fontsize=9, color=style.get('axes.labelcolor', '#c0c0c8'))
            ax.set_title(col, fontsize=10, fontweight='bold', color=style.get('text.color', '#e0e0e0'))
            ax.legend(fontsize=8, facecolor=style.get('legend.facecolor', '#22222a'),
                     edgecolor=style.get('legend.edgecolor', '#3a3a42'),
                     labelcolor=style.get('legend.labelcolor', '#e0e0e0'))
            if params['show_grid']:
                ax.grid(True, alpha=0.3, linestyle='--', color=style.get('grid.color', '#2e2e36'))
            ax.tick_params(colors=style.get('xtick.color', '#808088'))

        # Success rate bar chart
        if 'Success' in self.filtered_data.columns and 'Type' in self.filtered_data.columns:
            ax = fig.add_subplot(234)
            ax.set_facecolor(style.get('axes.facecolor', '#22222a'))
            labels_list, rates, bar_colors = [], [], []
            for t in types_list:
                if t not in colors:
                    continue
                subset = self.filtered_data[self.filtered_data['Type'] == t]
                rate = subset['Success'].sum() / len(subset) * 100
                labels_list.append(t)
                rates.append(rate)
                bar_colors.append(colors[t])
            ax.bar(labels_list, rates, color=bar_colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Success Rate (%)', fontsize=9, color=style.get('axes.labelcolor', '#c0c0c8'))
            ax.set_title('Success Rate by Type', fontsize=10, fontweight='bold',
                        color=style.get('text.color', '#e0e0e0'))
            ax.set_ylim([0, 100])
            if params['show_grid']:
                ax.grid(True, alpha=0.3, linestyle='--', axis='y', color=style.get('grid.color', '#2e2e36'))
            for i, (lbl, rate) in enumerate(zip(labels_list, rates)):
                ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold',
                       color=style.get('text.color', '#e0e0e0'))
            ax.tick_params(colors=style.get('xtick.color', '#808088'))

        fig.suptitle('Monte Carlo Distribution Analysis', fontsize=14, fontweight='bold',
                     color=style.get('text.color', '#e0e0e0'))
        fig.tight_layout()
        self._embed_figure(fig)
        self.status_bar.showMessage('Showing: Monte Carlo Distributions')

    # ------------------------------------------------------------------
    # Plugin graph rendering
    # ------------------------------------------------------------------

    def _render_plugin_graph(self, graph_key):
        if graph_key not in self.plugin_graphs:
            QMessageBox.warning(self, 'Error', f'Unknown graph: {graph_key}')
            return

        self.last_view = f'plugin:{graph_key}'
        style = get_matplotlib_style(self._theme_mode)
        ext_or_plugin, gdef = self.plugin_graphs[graph_key]

        fig = Figure(figsize=(12, 8), dpi=100)
        fig.set_facecolor(style.get('figure.facecolor', '#1a1a20'))

        try:
            if hasattr(ext_or_plugin, 'render_graph'):
                ext_or_plugin.render_graph(
                    graph_key, self.data_store, fig,
                    yscale=self._yscale_combo.currentText()
                )
        except Exception as e:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error rendering graph:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, color='red')
            traceback.print_exc()

        self._embed_figure(fig)
        label = getattr(gdef, 'label', graph_key)
        self.status_bar.showMessage(f'Showing: {label}')

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_plot(self):
        if self.current_figure is None:
            QMessageBox.warning(self, 'No Plot', 'No plot to export.')
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export Plot', '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)'
        )
        if filepath:
            try:
                dpi = self.prefs.get('plot', 'export_dpi', 300)
                self.current_figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
                self.status_bar.showMessage(f'Exported to {filepath}')
            except Exception as e:
                QMessageBox.critical(self, 'Export Error', str(e))

    # ------------------------------------------------------------------
    # Drag and drop support for .vortexext files
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.vortexext'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.endswith('.vortexext'):
                try:
                    manifest = self.ext_manager.install_from_vortexext(filepath)
                    self._refresh_installed_list()
                    QMessageBox.information(
                        self, 'Installed',
                        f'Installed: {manifest.name} v{manifest.version}\n'
                        'Restart for full effect.'
                    )
                except Exception as e:
                    QMessageBox.critical(self, 'Install Failed', str(e))

    def closeEvent(self, event):
        """Save preferences on close."""
        self.prefs.save()
        self.ext_registry.deactivate_all()
        event.accept()


# ===================================================================
# Entry point
# ===================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)

    # Set icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'hexakinetic_icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Windows taskbar icon fix
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f'vortex.plotvisual.v{APP_VERSION}'
        )
    except Exception:
        pass

    window = PlotVisualWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

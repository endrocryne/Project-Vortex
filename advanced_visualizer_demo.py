"""
HexaVisual Demo v1.0 — Locked-down Demo Viewer for Project Vortex
===================================================================
Fork of HexaVisual v2.0 (advanced_visualizer.py) with:
  - 9 pre-selected demo runs in sidebar (no manual file loading)
  - Windows Hello authentication for quit, settings, file operations
  - Curated trajectories matching PlotVisual sample data (seed=42)
  - Simplified menus (no extensions, no HexaKinetic integration)

Run:
    python advanced_visualizer_demo.py
"""

import sys
import os
import json
import time
import ctypes
import math

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import CubicSpline

from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit,
    QSlider, QComboBox, QGroupBox, QMessageBox, QCheckBox,
    QAction, QScrollArea, QFrame, QSizePolicy, QSplitter,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

# Import base visualizer — reuses all rendering, physics, playback code
from advanced_visualizer import RocketVisualizer


# ---------------------------------------------------------------------------
# HexaVisual Demo App
# ---------------------------------------------------------------------------
class RocketVisualizerDemo(RocketVisualizer):
    """
    Locked-down demo version of HexaVisual with pre-loaded trajectories,
    sidebar navigation, and Windows Hello authentication gates.

    Inherits all rendering, physics, flame, particle, camera, HUD, and
    playback code from RocketVisualizer. Overrides only:
      - Preferences I/O (hexavisual_demo_prefs.json)
      - UI layout (sidebar + no file-loading buttons)
      - Menu (no extensions, no HexaKinetic)
      - Extension system (disabled)
    """

    def __init__(self):
        # Pre-init state needed by overridden methods that run during super().__init__
        self._demo_manifest = None
        self._demo_runs = []
        self._selected_run_index = -1
        self._run_cards = []
        self._info_labels = {}
        self._info_desc = None
        self._last_trail_update_frame = -999999
        self._trail_update_stride = 3
        self._pose_smooth_pos = None
        self._display_rot_matrix = None
        self._wind_event_actor_visible = False
        self._debris_active = False
        self._debris_start_time = None
        self._debris_start_pos = None
        self._debris_velocity = None
        self._fault_anim_plan = {"wind_start": None, "mass_loss_start": None, "wind_dir": np.array([1.0, 0.0, 0.0])}
        self._wind_streaks = []  # list of streak particle dicts
        self._wind_streak_actors = []

        # Pre-load manifest so init_ui (called inside super().__init__) can build cards
        self._load_manifest_early()

        # Parent __init__ calls (via dynamic dispatch) our overridden:
        #   load_preferences  → reads hexavisual_demo_prefs.json
        #   init_ui           → builds sidebar + viewport (no Load CSV/STL)
        #   create_menu       → simplified menus
        #   _init_extensions  → no-op
        # And inherits unchanged: create_ground, create_default_rocket,
        # load_explosion_asset, _apply_satellite_texture, _setup_lighting
        super().__init__()

        # Override window title, size, and taskbar identity set by parent
        self.setWindowTitle("HexaVisual Demo \u2014 Project Vortex")
        self.resize(1600, 950)
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "vortex.hexavisual.demo.v1.0"
            )
        except Exception:
            pass

        # Override HUD title (parent sets "HexaVisual v2.0")
        if hasattr(self, "hud_title"):
            self.hud_title.setText("HexaVisual Demo")

        # Auto-load first demo run
        if self._demo_runs:
            self.load_demo_run(0)

    # ------------------------------------------------------------------
    # Extension system — disabled in demo mode
    # ------------------------------------------------------------------
    def _init_extensions(self):
        """Override: no extensions in demo mode."""
        self.ext_manager = None
        self.ext_registry = None

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------
    def _load_manifest_early(self):
        """Load the demo manifest JSON before UI construction."""
        base = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(base, "results", "demo_runs", "demo_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    self._demo_manifest = json.load(f)
                self._demo_runs = self._demo_manifest.get("runs", [])
                print(f"[HexaVisual Demo] Loaded manifest: {len(self._demo_runs)} demo runs")
            except Exception as e:
                print(f"[HexaVisual Demo] Failed to load manifest: {e}")
                self._demo_runs = []
        else:
            print(f"[HexaVisual Demo] Manifest not found: {manifest_path}")
            self._demo_runs = []

    # ------------------------------------------------------------------
    # Preferences I/O — uses hexavisual_demo_prefs.json
    # ------------------------------------------------------------------
    def load_preferences(self):
        """Load preferences from the demo-specific file with demo-tuned defaults."""
        pref_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hexavisual_demo_prefs.json"
        )

        # Demo-tuned defaults (different from parent: higher scaling, slower playback,
        # Chase camera, arctic launch site coordinates)
        self.results_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "demo_runs")
        )
        self.default_rocket_stl = ""
        self.flame_stl_path = ""
        self.explosion_stl_path = os.path.join("assets", "Dramatic_explosion_effect_part.stl")
        self.lat_scale_val = "25.0"
        self.scale_lat_enabled = True
        self.vert_scale_val = "10.0"
        self.scale_vert_enabled = True
        self.default_playback_speed = "0.1x"
        self.launch_lat = 67.4
        self.launch_lon = -98.45
        self.map_api_key = ""
        self.map_zoom = 14
        self.map_enabled = True
        self.default_camera_mode = "Chase"
        self.enable_ssao = True
        self.enable_shadows = False
        self.show_trail = True
        self.mesh_decimate_target = 0.5
        self.mesh_decimate_threshold = 50000

        if os.path.exists(pref_path):
            try:
                with open(pref_path, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                self.results_dir = prefs.get("results_dir", self.results_dir)
                self.default_rocket_stl = prefs.get("default_rocket_stl", self.default_rocket_stl)
                self.flame_stl_path = prefs.get("flame_stl_path", self.flame_stl_path)
                self.explosion_stl_path = prefs.get("explosion_stl_path", self.explosion_stl_path)
                self.lat_scale_val = str(prefs.get("lat_scale", self.lat_scale_val))
                self.scale_lat_enabled = prefs.get("scale_lat_enabled", self.scale_lat_enabled)
                self.vert_scale_val = str(prefs.get("vert_scale", self.vert_scale_val))
                self.scale_vert_enabled = prefs.get("scale_vert_enabled", self.scale_vert_enabled)
                self.default_playback_speed = prefs.get("playback_speed", self.default_playback_speed)
                self.launch_lat = float(prefs.get("launch_lat", self.launch_lat))
                self.launch_lon = float(prefs.get("launch_lon", self.launch_lon))
                self.map_api_key = prefs.get("map_api_key", self.map_api_key)
                self.map_zoom = int(prefs.get("map_zoom", self.map_zoom))
                self.map_enabled = prefs.get("map_enabled", self.map_enabled)
                self.default_camera_mode = prefs.get("camera_mode", self.default_camera_mode)
                self.enable_ssao = prefs.get("enable_ssao", self.enable_ssao)
                self.enable_shadows = prefs.get("enable_shadows", self.enable_shadows)
                self.show_trail = prefs.get("show_trail", self.show_trail)
                self.mesh_decimate_target = float(
                    prefs.get("mesh_decimate_target", self.mesh_decimate_target)
                )
                self.mesh_decimate_threshold = int(
                    prefs.get("mesh_decimate_threshold", self.mesh_decimate_threshold)
                )
            except Exception as e:
                print(f"[DemoPrefs] Error loading preferences: {e}")

    def save_preferences(self):
        """Save preferences to the demo-specific prefs file."""
        self.results_dir = self.inp_res_dir.text()
        self.default_rocket_stl = self.inp_rock_stl.text()
        self.flame_stl_path = self.inp_flame_stl.text()
        self.explosion_stl_path = self.inp_exp_stl.text()
        try:
            self.launch_lat = float(self.inp_launch_lat.text())
        except ValueError:
            pass
        try:
            self.launch_lon = float(self.inp_launch_lon.text())
        except ValueError:
            pass
        self.map_api_key = self.inp_map_key.text()
        try:
            self.map_zoom = int(self.inp_map_zoom.text())
        except ValueError:
            pass
        self.map_enabled = self.chk_map_enabled.isChecked()
        self.enable_ssao = self.chk_ssao.isChecked()
        self.enable_shadows = self.chk_shadows.isChecked()

        pref_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hexavisual_demo_prefs.json"
        )
        prefs = {
            "results_dir": self.results_dir,
            "default_rocket_stl": self.default_rocket_stl,
            "flame_stl_path": self.flame_stl_path,
            "explosion_stl_path": self.explosion_stl_path,
            "lat_scale": self.inp_lat_scale.text(),
            "scale_lat_enabled": self.chk_scale_lat.isChecked(),
            "vert_scale": self.inp_vert_scale.text(),
            "scale_vert_enabled": self.chk_scale_vert.isChecked(),
            "playback_speed": self.combo_speed.currentText(),
            "launch_lat": self.launch_lat,
            "launch_lon": self.launch_lon,
            "map_api_key": self.map_api_key,
            "map_zoom": self.map_zoom,
            "map_enabled": self.map_enabled,
            "camera_mode": self.combo_camera.currentText(),
            "enable_ssao": self.enable_ssao,
            "enable_shadows": self.enable_shadows,
            "show_trail": self.chk_trail.isChecked(),
            "mesh_decimate_target": self.mesh_decimate_target,
            "mesh_decimate_threshold": self.mesh_decimate_threshold,
            "demo_mode": True,
        }
        try:
            with open(pref_path, "w", encoding="utf-8") as f:
                json.dump(prefs, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save preferences: {e}")

        # Reload assets if changed
        if self.default_rocket_stl and os.path.exists(self.default_rocket_stl):
            self.create_default_rocket()
        if self.flame_stl_path and os.path.exists(self.flame_stl_path):
            self.load_custom_flame(self.flame_stl_path)
        if self.explosion_stl_path:
            self.load_explosion_asset_path(self.explosion_stl_path)

        # Re-apply satellite imagery
        self._apply_satellite_texture()
        # Re-apply lighting
        self._setup_lighting()

        self.pref_dialog.accept()

    # ------------------------------------------------------------------
    # UI construction — sidebar + viewport (no file loading buttons)
    # ------------------------------------------------------------------
    def init_ui(self):
        """Override parent init_ui: adds sidebar, removes Loading group."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Menu (our simplified override)
        self.create_menu()

        # Main horizontal layout: [sidebar | viewport + controls]
        main_h_layout = QHBoxLayout(central_widget)
        main_h_layout.setContentsMargins(0, 0, 0, 0)
        main_h_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        # --- Sidebar (left, fixed width) ---
        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        # --- Right panel (viewport + controls) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(2)

        # 3D viewport container (with HUD overlay)
        viewport_container = QWidget()
        viewport_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        viewport_layout.addWidget(self.plotter.interactor)

        # HUD overlay (inherited from parent — transparent Qt labels)
        self._build_hud_overlay(viewport_container)

        right_layout.addWidget(viewport_container, stretch=1)

        # --- Controls (no Loading group — demo runs are loaded via sidebar) ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(2, 2, 2, 2)
        controls_layout.setSpacing(2)

        # Row 1: Scaling + Visualization
        row1 = QHBoxLayout()

        # Scaling group
        scale_group = QGroupBox("Scaling")
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(4)

        self.chk_scale_lat = QCheckBox("Lateral (X/Y)")
        self.chk_scale_lat.setChecked(self.scale_lat_enabled)
        self.chk_scale_lat.stateChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.chk_scale_lat)

        self.inp_lat_scale = QLineEdit(self.lat_scale_val)
        self.inp_lat_scale.setFixedWidth(45)
        self.inp_lat_scale.editingFinished.connect(self._on_scale_changed)
        scale_layout.addWidget(self.inp_lat_scale)

        scale_layout.addWidget(QLabel("\u00d7"))

        self.chk_scale_vert = QCheckBox("Vertical (Z)")
        self.chk_scale_vert.setChecked(self.scale_vert_enabled)
        self.chk_scale_vert.stateChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.chk_scale_vert)

        self.inp_vert_scale = QLineEdit(self.vert_scale_val)
        self.inp_vert_scale.setFixedWidth(45)
        self.inp_vert_scale.editingFinished.connect(self._on_scale_changed)
        scale_layout.addWidget(self.inp_vert_scale)

        scale_group.setLayout(scale_layout)
        row1.addWidget(scale_group)

        # Visualization group
        vis_group = QGroupBox("Visualization")
        vis_layout = QHBoxLayout()
        vis_layout.setSpacing(4)

        self.chk_trail = QCheckBox("Trajectory Trail")
        self.chk_trail.setChecked(self.show_trail)
        self.chk_trail.stateChanged.connect(self._toggle_trail)
        vis_layout.addWidget(self.chk_trail)

        vis_layout.addWidget(QLabel("Camera:"))
        self.combo_camera = QComboBox()
        self.combo_camera.addItems(["Free", "Follow", "Chase", "Orbit"])
        self.combo_camera.setCurrentText(self.default_camera_mode)
        self._camera_mode = self.default_camera_mode
        self.combo_camera.currentTextChanged.connect(self._on_camera_mode_changed)
        vis_layout.addWidget(self.combo_camera)

        vis_group.setLayout(vis_layout)
        row1.addWidget(vis_group)

        controls_layout.addLayout(row1)

        # Row 2: Playback
        timeline_group = QGroupBox("Playback")
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(4)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFixedWidth(60)
        timeline_layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_change)
        timeline_layout.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("T: 0.00s")
        self.lbl_time.setFixedWidth(85)
        timeline_layout.addWidget(self.lbl_time)

        self.combo_speed = QComboBox()
        self.combo_speed.setEditable(True)
        self.combo_speed.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "2.0x", "5.0x", "10.0x"])
        self.combo_speed.setCurrentText(self.default_playback_speed)
        self.update_speed(self.default_playback_speed)
        self.combo_speed.currentTextChanged.connect(self.update_speed)
        self.combo_speed.setFixedWidth(65)
        timeline_layout.addWidget(self.combo_speed)

        timeline_layout.addWidget(QLabel("Skip(s):"))
        self.inp_skip = QLineEdit("5.0")
        self.inp_skip.setFixedWidth(35)
        timeline_layout.addWidget(self.inp_skip)

        self.btn_back = QPushButton("<<")
        self.btn_back.setFixedWidth(30)
        self.btn_back.clicked.connect(lambda: self.skip_time(-1))
        timeline_layout.addWidget(self.btn_back)

        self.btn_fwd = QPushButton(">>")
        self.btn_fwd.setFixedWidth(30)
        self.btn_fwd.clicked.connect(lambda: self.skip_time(1))
        timeline_layout.addWidget(self.btn_fwd)

        timeline_group.setLayout(timeline_layout)
        controls_layout.addWidget(timeline_group)

        right_layout.addWidget(controls_widget, stretch=0)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1260])
        main_h_layout.addWidget(splitter)

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ------------------------------------------------------------------
    # Sidebar construction
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        """Build the demo-run sidebar with clickable cards and info panel."""
        sidebar = QWidget()
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(520)
        sidebar.setStyleSheet("QWidget#demo_sidebar { background-color: #1e1e22; }")
        sidebar.setObjectName("demo_sidebar")

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 8)
        sidebar_layout.setSpacing(4)

        # Title
        title = QLabel("Demo Runs")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #00BFFF; background: transparent; padding: 0 0 2px 0;")
        title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title)

        subtitle = QLabel("Synthetic runs anchored to optimization_20260222_120325")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #888888; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(subtitle)

        # Scrollable run list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: #2a2a2e; border: none; }"
            "QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self._cards_layout = QVBoxLayout(scroll_content)
        self._cards_layout.setSpacing(4)
        self._cards_layout.setContentsMargins(0, 4, 0, 4)

        # Create run cards from manifest
        self._run_cards = []
        if self._demo_runs:
            for i, run in enumerate(self._demo_runs):
                card = self._create_run_card(i, run)
                self._cards_layout.addWidget(card)
                self._run_cards.append(card)
        else:
            no_data = QLabel(
                "No demo runs found.\n\n"
                "Run scripts/generate_demo_trajectories.py\n"
                "to generate demo data."
            )
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setStyleSheet("color: #ff5555; background: transparent; padding: 20px;")
            no_data.setWordWrap(True)
            self._cards_layout.addWidget(no_data)

        self._cards_layout.addStretch()
        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll, stretch=1)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444444; background: #444444; max-height: 1px;")
        sidebar_layout.addWidget(sep)

        # Info panel header
        info_header = QLabel("Run Details")
        info_header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        info_header.setStyleSheet("color: #CCCCCC; background: transparent; padding: 4px 0;")
        sidebar_layout.addWidget(info_header)

        # Description
        self._info_desc = QLabel("Select a run to view details")
        self._info_desc.setFont(QFont("Segoe UI", 8))
        self._info_desc.setStyleSheet("color: #999999; background: transparent;")
        self._info_desc.setWordWrap(True)
        self._info_desc.setMaximumHeight(52)
        sidebar_layout.addWidget(self._info_desc)

        # Detail fields
        self._info_labels = {}
        info_fields = [
            ("mode", "Mode"),
            ("fi", "Fault Intensity"),
            ("faults", "Fault Types"),
            ("velocity", "Landing Vel"),
            ("distance", "Landing Dist"),
            ("altitude", "Apogee"),
            ("init_vel", "Peak Vz"),
            ("mass", "Dry Mass"),
            ("thrust", "Avg Thrust"),
        ]

        for key, label_text in info_fields:
            row = QHBoxLayout()
            row.setSpacing(4)

            lbl = QLabel(f"{label_text}:")
            lbl.setFont(QFont("Segoe UI", 8))
            lbl.setStyleSheet("color: #777777; background: transparent;")
            lbl.setFixedWidth(90)
            row.addWidget(lbl)

            val = QLabel("\u2014")
            val.setFont(QFont("Consolas", 9))
            val.setStyleSheet("color: #DDDDDD; background: transparent;")
            val.setWordWrap(True)
            row.addWidget(val, stretch=1)

            self._info_labels[key] = val
            sidebar_layout.addLayout(row)

        return sidebar

    def _create_run_card(self, index, run):
        """Create a single clickable run card for the sidebar."""
        color = run.get("color", "#888888")
        status = run.get("status_label", "UNKNOWN")
        name = f"Demo {index + 1}"
        fi = run.get("fault_intensity", 0.0)
        mode = run.get("mode", "")

        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setCursor(Qt.PointingHandCursor)
        card.setFixedHeight(64)
        card.setProperty("run_index", index)
        card.setObjectName(f"card_{index}")
        card.setStyleSheet(
            f"QFrame#card_{index} {{"
            f"  background-color: #2a2a2e;"
            f"  border: 2px solid transparent;"
            f"  border-left: 4px solid {color};"
            f"  border-radius: 6px;"
            f"  padding: 4px;"
            f"}}"
            f"QFrame#card_{index}:hover {{"
            f"  background-color: #35353a;"
            f"  border: 2px solid #555555;"
            f"  border-left: 4px solid {color};"
            f"}}"
        )

        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Run number
        num_lbl = QLabel(f"#{index + 1}")
        num_lbl.setFont(QFont("Consolas", 14, QFont.Bold))
        num_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
        num_lbl.setFixedWidth(34)
        num_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(num_lbl)

        # Info column
        info_col = QVBoxLayout()
        info_col.setSpacing(1)

        name_lbl = QLabel(name)
        name_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        name_lbl.setStyleSheet("color: #DDDDDD; background: transparent; border: none;")
        name_lbl.setWordWrap(False)
        info_col.addWidget(name_lbl)

        detail_text = f"{mode} | fi={fi:.2f}"
        detail_lbl = QLabel(detail_text)
        detail_lbl.setFont(QFont("Consolas", 7))
        detail_lbl.setStyleSheet("color: #999999; background: transparent; border: none;")
        info_col.addWidget(detail_lbl)

        layout.addLayout(info_col, stretch=1)

        # Status badge
        badge = QLabel(status)
        badge.setFont(QFont("Segoe UI", 7, QFont.Bold))
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(58)
        badge.setFixedHeight(20)

        if status == "SUCCESS":
            badge.setStyleSheet(
                "color: white; background-color: #4CAF50; border-radius: 4px; "
                "padding: 2px; border: none;"
            )
        elif status == "FAIL":
            badge.setStyleSheet(
                "color: white; background-color: #FF9800; border-radius: 4px; "
                "padding: 2px; border: none;"
            )
        elif status == "CRASH":
            badge.setStyleSheet(
                "color: white; background-color: #F44336; border-radius: 4px; "
                "padding: 2px; border: none;"
            )
        else:
            badge.setStyleSheet(
                "color: white; background-color: #666666; border-radius: 4px; "
                "padding: 2px; border: none;"
            )

        layout.addWidget(badge)

        # Click handler — loads the demo run
        card.mousePressEvent = lambda event, idx=index: self.load_demo_run(idx)

        return card

    def _update_run_selection(self):
        """Update visual selection state of all run cards (highlight current)."""
        for i, card in enumerate(self._run_cards):
            if i >= len(self._demo_runs):
                break
            run = self._demo_runs[i]
            color = run.get("color", "#888888")

            if i == self._selected_run_index:
                card.setStyleSheet(
                    f"QFrame#card_{i} {{"
                    f"  background-color: #2a3a4e;"
                    f"  border: 2px solid #00BFFF;"
                    f"  border-left: 4px solid {color};"
                    f"  border-radius: 6px;"
                    f"  padding: 4px;"
                    f"}}"
                    f"QFrame#card_{i}:hover {{"
                    f"  background-color: #2a3a4e;"
                    f"  border: 2px solid #00BFFF;"
                    f"  border-left: 4px solid {color};"
                    f"}}"
                )
            else:
                card.setStyleSheet(
                    f"QFrame#card_{i} {{"
                    f"  background-color: #2a2a2e;"
                    f"  border: 2px solid transparent;"
                    f"  border-left: 4px solid {color};"
                    f"  border-radius: 6px;"
                    f"  padding: 4px;"
                    f"}}"
                    f"QFrame#card_{i}:hover {{"
                    f"  background-color: #35353a;"
                    f"  border: 2px solid #555555;"
                    f"  border-left: 4px solid {color};"
                    f"}}"
                )

    # ------------------------------------------------------------------
    # Demo run loading
    # ------------------------------------------------------------------
    def load_demo_run(self, index):
        """Load a demo run by its index in the manifest."""
        if index < 0 or index >= len(self._demo_runs):
            return

        run = self._demo_runs[index]
        self._selected_run_index = index

        # Stop any current playback
        self.is_playing = False
        self.btn_play.setText("Play")
        self.timer.stop()

        # Update card highlight
        self._update_run_selection()

        # Resolve CSV path relative to demo_runs directory
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "demo_runs"
        )
        csv_path = os.path.join(base_dir, run.get("trajectory_path", ""))

        if os.path.exists(csv_path):
            self.load_csv_file(csv_path)
        else:
            QMessageBox.warning(
                self, "Missing Data",
                f"Trajectory file not found:\n{csv_path}\n\n"
                f"Run scripts/generate_demo_trajectories.py to generate demo data.",
            )
            return

        self._prepare_fault_animation_plan(run)
        if "demo_wind_event" in self.plotter.renderer.actors:
            self.plotter.renderer.actors["demo_wind_event"].SetVisibility(False)
        if "demo_mass_debris" in self.plotter.renderer.actors:
            self.plotter.renderer.actors["demo_mass_debris"].SetVisibility(False)

        # Update sidebar info panel
        self._update_info_panel(run)

        # Update HUD title
        if hasattr(self, "hud_title"):
            self.hud_title.setText(f"Demo {index + 1}")

        # Update window title
        self.setWindowTitle(
            f"HexaVisual Demo \u2014 Demo {index + 1} \u2014 Project Vortex"
        )

        print(f"[HexaVisual Demo] Loaded demo {index + 1}")

    # ------------------------------------------------------------------
    # High-resolution CSV loading with cubic-spline upsampling
    # ------------------------------------------------------------------
    def load_csv_file(self, path):
        """Load trajectory CSV then upsample to 5× resolution via cubic splines."""
        super().load_csv_file(path)
        if self.df is not None and len(self.df) >= 4:
            self._upsample_trajectory(factor=5)

    def _upsample_trajectory(self, factor=5):
        """Resample trajectory data to `factor` × higher resolution using cubic splines.

        Smooth columns (X, Y, Z, VX, VY, VZ) use CubicSpline for C² continuity.
        Monotone-safe columns (QW, QX, QY, QZ, Mass) use linear interpolation to
        avoid Runge-style overshoot on near-constant regions.
        """
        if self.df is None or len(self.df) < 4:
            return

        t_orig = self.df["Time"].values.astype(float)
        n_new = (len(t_orig) - 1) * factor + 1
        t_new = np.linspace(t_orig[0], t_orig[-1], n_new)

        spline_cols = [c for c in ["X", "Y", "Z", "VX", "VY", "VZ"] if c in self.df.columns]
        linear_cols = [c for c in ["QW", "QX", "QY", "QZ", "Mass"] if c in self.df.columns]

        new_data = {"Time": t_new}

        for col in spline_cols:
            try:
                cs = CubicSpline(t_orig, self.df[col].values.astype(float))
                new_data[col] = cs(t_new)
            except Exception:
                new_data[col] = np.interp(t_new, t_orig, self.df[col].values.astype(float))

        for col in linear_cols:
            new_data[col] = np.interp(t_new, t_orig, self.df[col].values.astype(float))
            # Re-normalise quaternion vector after interpolation
        if all(c in new_data for c in ["QW", "QX", "QY", "QZ"]):
            q = np.column_stack([new_data["QW"], new_data["QX"],
                                  new_data["QY"], new_data["QZ"]])
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            q = q / norms
            new_data["QW"] = q[:, 0]
            new_data["QX"] = q[:, 1]
            new_data["QY"] = q[:, 2]
            new_data["QZ"] = q[:, 3]

        self.df = pd.DataFrame(new_data)

        # Update time step and slider range to match new resolution
        self.time_step = float(t_new[1] - t_new[0])
        self.slider.setRange(0, len(self.df) - 1)
        self.slider.setValue(0)
        self.current_frame = 0

        # Invalidate cached apogee frame index — will be recomputed on next render
        self._apogee_frame = None

        # Scale trail-update stride proportionally so trail redraws happen at the
        # same real-time rate as with the original 50 Hz data (stride=3 at 1×).
        self._trail_update_stride = max(1, factor * 3)

        # Rebuild trail polyline at the new higher resolution
        try:
            self._precompute_trail()
        except Exception:
            pass

        print(f"[HexaVisual Demo] Upsampled trajectory {factor}× → {len(self.df)} frames "
              f"(dt={self.time_step*1000:.2f} ms)")

    def _prepare_fault_animation_plan(self, run):
        """Prepare timing/direction for wind and mass-loss visual effects."""
        self._fault_anim_plan = {
            "wind_start": None,
            "mass_loss_start": None,
            "wind_dir": np.array([1.0, 0.0, 0.0], dtype=float),
        }
        self._debris_active = False
        self._debris_start_time = None
        self._debris_start_pos = None
        self._debris_velocity = None

        faults = run.get("fault_types", [])
        lx = float(run.get("landing_x", 0.0))
        ly = float(run.get("landing_y", 0.0))
        norm = max((lx * lx + ly * ly) ** 0.5, 1e-6)
        self._fault_anim_plan["wind_dir"] = np.array([lx / norm, ly / norm, 0.0], dtype=float)

        if self.df is not None and len(self.df) > 5 and "Z" in self.df.columns and "Time" in self.df.columns:
            i_apogee = int(self.df["Z"].idxmax())
            t_apogee = float(self.df.iloc[i_apogee]["Time"])
        else:
            t_apogee = 7.5

        if "WIND_GUST" in faults:
            self._fault_anim_plan["wind_start"] = t_apogee + 0.7
        if "MASS_LOSS" in faults:
            self._fault_anim_plan["mass_loss_start"] = t_apogee + 1.1

    def _update_info_panel(self, run):
        """Update the sidebar info panel with details from the selected run."""
        # Description
        desc = run.get("description", "")
        if self._info_desc is not None:
            self._info_desc.setText(desc)

        # Fields
        self._info_labels["mode"].setText(run.get("mode", "\u2014"))

        fi = run.get("fault_intensity", 0)
        self._info_labels["fi"].setText(f"{fi:.2f}")

        faults = run.get("fault_types", [])
        self._info_labels["faults"].setText(", ".join(faults) if faults else "None")

        vel = run.get("landing_velocity", 0)
        color = run.get("color", "#DDDDDD")
        self._info_labels["velocity"].setText(f"{vel:.3f} m/s")
        self._info_labels["velocity"].setStyleSheet(
            f"color: {color}; background: transparent;"
        )

        dist = run.get("landing_distance", 0)
        self._info_labels["distance"].setText(f"{dist:.1f} m")

        alt = run.get("apogee", run.get("initial_altitude", 0))
        self._info_labels["altitude"].setText(f"{alt:.0f} m")

        iv = run.get("peak_vertical_velocity", run.get("initial_velocity", 0))
        self._info_labels["init_vel"].setText(f"{iv:.1f} m/s")

        mass = run.get("dry_mass", 0)
        self._info_labels["mass"].setText(f"{mass:.1f} kg")

        thrust = run.get("thrust_average", 0)
        self._info_labels["thrust"].setText(f"{thrust:.0f} N")

    # ------------------------------------------------------------------
    # Menu — simplified for demo
    # ------------------------------------------------------------------
    def create_menu(self):
        """Build simplified menu without extensions or HexaKinetic integration."""
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.open_about)
        file_menu.addAction(about_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit
        edit_menu = menubar.addMenu("Edit")

        pref_action = QAction("Preferences", self)
        pref_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(pref_action)

        # View
        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(lambda: self.plotter.camera.zoom(1.2))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(lambda: self.plotter.camera.zoom(0.8))
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()

        reset_cam_action = QAction("Reset Camera", self)
        reset_cam_action.triggered.connect(lambda: self.plotter.reset_camera())
        view_menu.addAction(reset_cam_action)

        view_menu.addSeparator()

        toggle_trail_action = QAction("Toggle Trajectory Trail", self)
        toggle_trail_action.triggered.connect(
            lambda: self.chk_trail.setChecked(not self.chk_trail.isChecked())
        )
        view_menu.addAction(toggle_trail_action)

    # ------------------------------------------------------------------
    # About dialog — demo version
    # ------------------------------------------------------------------
    def open_about(self):
        """Show the demo-specific About dialog."""
        QMessageBox.about(
            self,
            "About",
            "HexaVisual Demo v1.0\n\n"
            "Pre-configured demonstration of Project Vortex\n"
            "3D flight trajectory visualization\n\n"
            "9 curated demo runs showing Optimization vs. ML\n"
            "flight computer performance across fault intensities\n\n"
            "Synthetic data anchored to optimization_20260222_120325\n\n"
            "Based on HexaVisual v2.0\n"
            "\u00a9 2026 Agastya Mishra",
        )

    def load_csv_file(self, path):
        """Load CSV then smooth positions for stable visuals while preserving original velocities."""
        super().load_csv_file(path)
        if self.df is None or len(self.df) < 8:
            return

        n = len(self.df)
        # Save original velocities — these are the ground truth from the trajectory generator
        orig_vx = self.df["VX"].to_numpy(dtype=float).copy() if "VX" in self.df.columns else None
        orig_vy = self.df["VY"].to_numpy(dtype=float).copy() if "VY" in self.df.columns else None
        orig_vz = self.df["VZ"].to_numpy(dtype=float).copy() if "VZ" in self.df.columns else None

        # Smooth positions with tapered window: strong in the middle, no smoothing at endpoints
        # This prevents the rolling mean from blurring the start/end positions
        for col in ["X", "Y", "Z"]:
            if col in self.df.columns:
                raw = self.df[col].to_numpy(dtype=float).copy()
                smoothed = (
                    self.df[col]
                    .rolling(window=5, center=True, min_periods=1)
                    .mean()
                    .to_numpy(dtype=float)
                )
                # Taper: blend raw at endpoints, smoothed in middle
                # First and last 15 frames use original data, gradual blend
                blend_margin = min(15, n // 6)
                blend = np.ones(n, dtype=float)
                for i in range(blend_margin):
                    f = i / max(blend_margin, 1)
                    blend[i] = f
                    blend[n - 1 - i] = f
                self.df[col] = raw * (1.0 - blend) + smoothed * blend

        if "Z" in self.df.columns:
            self.df["Z"] = self.df["Z"].clip(lower=0.0)

        # Recompute velocities from smoothed positions for the interior only
        if "Time" in self.df.columns:
            t = self.df["Time"].to_numpy(dtype=float)
            for p_col, v_col, orig_v in [("X", "VX", orig_vx), ("Y", "VY", orig_vy), ("Z", "VZ", orig_vz)]:
                if p_col in self.df.columns and orig_v is not None:
                    p = self.df[p_col].to_numpy(dtype=float)
                    recomp = np.gradient(p, t, edge_order=1)
                    # Blend: use original velocity at endpoints, recomputed in middle
                    blend_v = np.ones(n, dtype=float)
                    v_margin = min(20, n // 5)
                    for i in range(v_margin):
                        f = i / max(v_margin, 1)
                        blend_v[i] = f
                        blend_v[n - 1 - i] = f
                    self.df[v_col] = orig_v * (1.0 - blend_v) + recomp * blend_v

        # Light smoothing on velocities (only interior, not endpoints)
        for v_col, orig_v in [("VX", orig_vx), ("VY", orig_vy), ("VZ", orig_vz)]:
            if v_col in self.df.columns and orig_v is not None:
                smoothed_v = self.df[v_col].rolling(window=3, center=True, min_periods=1).mean().to_numpy(dtype=float)
                blend_vs = np.ones(n, dtype=float)
                vs_margin = min(10, n // 8)
                for i in range(vs_margin):
                    f = i / max(vs_margin, 1)
                    blend_vs[i] = f
                    blend_vs[n - 1 - i] = f
                self.df[v_col] = orig_v * (1.0 - blend_vs) + smoothed_v * blend_vs

        self._precompute_trail()
        self._last_trail_update_frame = -999999
        self._pose_smooth_pos = None
        self._display_rot_matrix = None
        self._apogee_frame = None
        self._ignition_frame = None
        self._tumble_flip_axis = None
        self._landing_burn_start_frame = None
        self.update_scene()

    def _compute_display_rotation(self):
        """Physics-based rocket orientation with nose-down freefall and RCS flip.

        Timeline (all times relative to loaded trajectory):
          - Ascent burn: follows thrust vector
          - Pre-apogee coast: gentle upright lean
          - After apogee (TUMBLE_DELAY): aerodynamic tumble 0° → 180° over ~1.65 s
            Physics: rocket falling tail-first is aerodynamically unstable;
            CP is ahead of CG → torque flips it nose-down
          - Stable nose-down: inverted (~180°) with small aerodynamic oscillation
          - RCS flip (starts ~1.95 s before ignition): cold-gas thrusters rotate
            180° → 0° before main motor ignites
          - Landing burn: nose-up, small TVC lean
        """
        if self.df is None or len(self.df) < 2:
            return np.eye(3)

        row = self.df.iloc[self.current_frame]
        lat_scale = self._get_lat_scale()
        vert_scale = self._get_vert_scale()

        # --- Cache key frame indices ---
        if not hasattr(self, '_apogee_frame') or self._apogee_frame is None:
            self._apogee_frame = int(self.df['Z'].idxmax()) if 'Z' in self.df.columns else 0

        if not hasattr(self, '_ignition_frame') or self._ignition_frame is None:
            self._ignition_frame = len(self.df) - 1
            if 'Mass' in self.df.columns:
                apg = self._apogee_frame
                for i in range(apg, len(self.df) - 1):
                    if float(self.df.iloc[i + 1]['Mass']) < float(self.df.iloc[i]['Mass']) - 1e-9:
                        self._ignition_frame = i
                        break

        # --- Cache tumble flip axis from lateral velocity near apogee ---
        if not hasattr(self, '_tumble_flip_axis') or self._tumble_flip_axis is None:
            apg = self._apogee_frame
            sample_f = min(apg + max(1, int(round(0.4 / max(self.time_step, 1e-9)))), len(self.df) - 1)
            vx_s = float(self.df.iloc[sample_f].get('VX', 0.0))
            vy_s = float(self.df.iloc[sample_f].get('VY', 0.0))
            lat_mag = math.sqrt(vx_s * vx_s + vy_s * vy_s)
            if lat_mag > 0.05:
                self._tumble_flip_axis = np.array([-vy_s / lat_mag, vx_s / lat_mag, 0.0])
            else:
                self._tumble_flip_axis = np.array([0.0, 1.0, 0.0])

        flip_axis = self._tumble_flip_axis
        fx, fy = flip_axis[0], flip_axis[1]

        # --- Timing ---
        apogee_t = float(self.df.iloc[self._apogee_frame]['Time'])
        ignition_t = float(self.df.iloc[self._ignition_frame]['Time'])
        freefall_dur = max(ignition_t - apogee_t, 1.0)
        curr_t = float(row['Time'])

        TUMBLE_DELAY = 0.25
        TUMBLE_DUR = min(1.65, freefall_dur * 0.38)
        RCS_DUR = min(1.80, freefall_dur * 0.42)
        RCS_MARGIN = 0.15

        tumble_start_t = apogee_t + TUMBLE_DELAY
        tumble_end_t = tumble_start_t + TUMBLE_DUR
        rcs_end_t = ignition_t - RCS_MARGIN
        rcs_start_t = rcs_end_t - RCS_DUR
        if rcs_start_t < tumble_end_t:
            rcs_start_t = tumble_end_t + 0.05

        # --- Flight state flags ---
        is_ascending = self.current_frame <= self._apogee_frame
        is_burning = False
        if self.current_frame > 0 and 'Mass' in self.df.columns:
            m_prev = float(self.df.iloc[self.current_frame - 1]['Mass'])
            m_curr = float(row['Mass'])
            is_burning = m_curr < m_prev - 1e-9
        is_landing_burn = is_burning and not is_ascending

        # --- Trajectory tangent for lateral lean ---
        span = max(2, int(round(0.10 / max(self.time_step, 1e-9))))
        i0 = max(0, self.current_frame - span)
        i1 = min(len(self.df) - 1, self.current_frame + span)
        if i0 == i1:
            i1 = min(len(self.df) - 1, i0 + 1)
        p0_t = np.array([
            float(self.df.iloc[i0]['X']) * lat_scale,
            float(self.df.iloc[i0]['Y']) * lat_scale,
            float(self.df.iloc[i0]['Z']) * vert_scale,
        ])
        p1_t = np.array([
            float(self.df.iloc[i1]['X']) * lat_scale,
            float(self.df.iloc[i1]['Y']) * lat_scale,
            float(self.df.iloc[i1]['Z']) * vert_scale,
        ])
        tangent = p1_t - p0_t
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            return self._display_rot_matrix if self._display_rot_matrix is not None else np.eye(3)
        tangent_dir = tangent / tangent_norm
        lateral_xy = np.array([tangent_dir[0], tangent_dir[1], 0.0])
        lat_norm = np.linalg.norm(lateral_xy)
        lateral_unit = lateral_xy / lat_norm if lat_norm > 1e-6 else np.array([0.0, 0.0, 0.0])

        # --- Smoothstep helper ---
        def ss(u: float) -> float:
            u = max(0.0, min(1.0, u))
            return u * u * (3.0 - 2.0 * u)

        # --- Rodrigues rotation of nose [0,0,1] around flip_axis by angle theta ---
        def nose_from_angle(theta: float) -> np.ndarray:
            s_t = math.sin(theta)
            c_t = math.cos(theta)
            return np.array([fy * s_t, -fx * s_t, c_t])

        # --- Determine z_axis (nose direction) and alpha (smoothing speed) ---
        during_flip = (tumble_start_t <= curr_t < ignition_t) and not is_landing_burn and not is_ascending

        if is_ascending or curr_t < apogee_t:
            if is_burning:
                z_axis = tangent_dir.copy()
                if z_axis[2] < 0.1:
                    z_axis[2] = 0.1
                alpha = 0.45
            else:
                z_axis = np.array([0.0, 0.0, 1.0]) + lateral_unit * min(0.10, lat_norm * 0.18)
                alpha = 0.18
            self._landing_burn_start_frame = None

        elif curr_t < tumble_start_t:
            z_axis = np.array([0.0, 0.0, 1.0]) + lateral_unit * min(0.06, lat_norm * 0.12)
            alpha = 0.12
            self._landing_burn_start_frame = None

        elif curr_t < tumble_end_t:
            u = (curr_t - tumble_start_t) / max(TUMBLE_DUR, 1e-6)
            theta = math.pi * ss(u)
            z_axis = nose_from_angle(theta)
            lean_fade = 1.0 - abs(math.cos(theta)) * 0.7
            z_axis += lateral_unit * min(0.07, lat_norm * 0.11) * lean_fade
            alpha = 0.65
            self._landing_burn_start_frame = None

        elif curr_t < rcs_start_t:
            t_nd = curr_t - tumble_end_t
            wobble = 0.035 * math.sin(1.8 * t_nd) * math.exp(-0.4 * t_nd)
            theta = math.pi + wobble
            z_axis = nose_from_angle(theta)
            z_axis += lateral_unit * min(0.07, lat_norm * 0.11)
            alpha = 0.14
            self._landing_burn_start_frame = None

        elif curr_t < ignition_t:
            u = (curr_t - rcs_start_t) / max(rcs_end_t - rcs_start_t, 1e-6)
            theta = math.pi * (1.0 - ss(u))
            z_axis = nose_from_angle(theta)
            lean_fade = theta / math.pi
            z_axis += lateral_unit * min(0.06, lat_norm * 0.10) * lean_fade
            alpha = 0.70
            self._landing_burn_start_frame = None

        elif is_landing_burn:
            tilt_amount = min(0.12, lat_norm * 0.25)
            z_axis = np.array([0.0, 0.0, 1.0]) + lateral_unit * tilt_amount
            alpha = 0.55
            self._landing_burn_start_frame = self._landing_burn_start_frame or self.current_frame

        else:
            tilt_amount = min(0.10, lat_norm * 0.18)
            z_axis = np.array([0.0, 0.0, 1.0]) + lateral_unit * tilt_amount
            alpha = 0.25
            self._landing_burn_start_frame = None

        # --- Normalize nose direction ---
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0])
        else:
            z_axis = z_axis / z_norm

        # --- Build rotation matrix ---
        if during_flip:
            x_axis = flip_axis.copy()
            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-6:
                y_axis = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            else:
                y_axis = y_axis / y_norm
            x_axis = np.cross(y_axis, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                x_axis = flip_axis.copy()
            else:
                x_axis = x_axis / x_norm
        else:
            if abs(z_axis[2]) > 0.999:
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(ref, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                x_axis = np.array([1.0, 0.0, 0.0])
            else:
                x_axis = x_axis / x_norm
            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-6:
                y_axis = np.array([0.0, 1.0, 0.0])
            else:
                y_axis = y_axis / y_norm

        target_rot = np.column_stack([x_axis, y_axis, z_axis])
        if np.linalg.det(target_rot) < 0:
            target_rot[:, 2] = -target_rot[:, 2]

        if self._display_rot_matrix is None:
            self._display_rot_matrix = target_rot
            return target_rot

        blended = (1.0 - alpha) * self._display_rot_matrix + alpha * target_rot
        u_svd, _, vt = np.linalg.svd(blended)
        smoothed = u_svd @ vt
        if np.linalg.det(smoothed) < 0:
            u_svd[:, 2] = -u_svd[:, 2]
            smoothed = u_svd @ vt

        self._display_rot_matrix = smoothed
        return smoothed

    def _update_hud(self, row, is_burning, is_crashed):
        """Update HUD with flight-phase-aware status labels."""
        super()._update_hud(row, is_burning, is_crashed)

        if is_crashed or is_burning:
            return

        curr_t = float(row.get('Time', 0.0))

        apogee_t = float(self.df.iloc[self._apogee_frame]['Time']) if (self.df is not None and self._apogee_frame is not None) else 0.0
        ignition_t = float(self.df.iloc[self._ignition_frame]['Time']) if (self.df is not None and self._ignition_frame is not None) else 1e9
        freefall_dur = max(ignition_t - apogee_t, 1.0)

        TUMBLE_DELAY = 0.25
        TUMBLE_DUR = min(1.65, freefall_dur * 0.38)
        RCS_DUR = min(1.80, freefall_dur * 0.42)
        RCS_MARGIN = 0.15
        tumble_start_t = apogee_t + TUMBLE_DELAY
        tumble_end_t = tumble_start_t + TUMBLE_DUR
        rcs_end_t = ignition_t - RCS_MARGIN
        rcs_start_t = rcs_end_t - RCS_DUR
        if rcs_start_t < tumble_end_t:
            rcs_start_t = tumble_end_t + 0.05

        is_ascending = self.current_frame <= self._apogee_frame if self._apogee_frame is not None else True

        if is_ascending or curr_t <= apogee_t:
            return

        if curr_t < tumble_start_t:
            label, color = "STATUS: COAST", "#00BFFF"
        elif curr_t < tumble_end_t:
            label, color = "STATUS: TUMBLING", "#FF6600"
        elif curr_t < rcs_start_t:
            label, color = "STATUS: NOSE-DOWN", "#FF4400"
        elif curr_t < ignition_t:
            label, color = "STATUS: RCS FLIP", "#FFDD00"
        else:
            return

        if hasattr(self, 'hud_status'):
            self.hud_status.setText(label)
            self.hud_status.setStyleSheet(
                f"color: {color}; background-color: rgba(0,0,0,160); "
                f"padding: 2px 6px; border-radius: 3px;"
            )

    def _update_fault_effects_animation(self, row, pos, rot_matrix):
        """Animate wind streaks and mass-loss events for active demo faults."""
        t = float(row.get("Time", 0.0))

        wind_start = self._fault_anim_plan.get("wind_start")
        wind_dir = self._fault_anim_plan.get("wind_dir", np.array([1.0, 0.0, 0.0]))
        wind_active = wind_start is not None and (wind_start <= t <= wind_start + 2.5)

        if wind_active:
            # Subtle wind streaks — small translucent lines drifting past the rocket
            age = t - wind_start
            fade_in = min(1.0, age / 0.5)
            fade_out = max(0.0, 1.0 - max(0.0, age - 1.8) / 0.7)
            opacity = 0.45 * fade_in * fade_out

            num_streaks = 8
            streak_pts = []
            for si in range(num_streaks):
                # Each streak is a short line segment near the rocket, offset in different positions
                phase = si * 0.7 + age * 3.5
                # Distribute streaks around the rocket at varying heights and lateral offsets
                perp = np.array([-wind_dir[1], wind_dir[0], 0.0])  # perpendicular to wind
                offset_perp = perp * (2.5 * math.sin(phase * 1.3 + si * 1.1))
                offset_z = np.array([0.0, 0.0, 3.0 * math.sin(phase * 0.9 + si * 0.8) + 2.0])

                # Streak moves with wind direction over time
                travel = wind_dir * (age * 6.0 + si * 1.5)
                # Wrap position to stay near rocket
                wrap_offset = wind_dir * (((si * 2.3 + age * 4.0) % 8.0) - 4.0)

                streak_start = pos + offset_perp + offset_z + wrap_offset - wind_dir * 1.5
                streak_end = streak_start + wind_dir * (1.8 + 0.8 * math.sin(phase))
                streak_pts.append(streak_start)
                streak_pts.append(streak_end)

            if len(streak_pts) >= 4:
                pts_array = np.array(streak_pts)
                n_lines = len(streak_pts) // 2
                # Build line segments
                lines = []
                for li in range(n_lines):
                    lines.extend([2, li * 2, li * 2 + 1])
                poly = pv.PolyData(pts_array)
                poly.lines = np.array(lines)
                self.plotter.add_mesh(
                    poly,
                    name="demo_wind_event",
                    color="#88ccee",
                    opacity=opacity,
                    line_width=2,
                    lighting=False,
                    emissive=True,
                )
                self._wind_event_actor_visible = True
        elif self._wind_event_actor_visible and "demo_wind_event" in self.plotter.renderer.actors:
            self.plotter.renderer.actors["demo_wind_event"].SetVisibility(False)
            self._wind_event_actor_visible = False

        loss_start = self._fault_anim_plan.get("mass_loss_start")
        if loss_start is not None and t >= loss_start and not self._debris_active:
            self._debris_active = True
            self._debris_start_time = t
            self._debris_start_pos = (pos + rot_matrix @ np.array([0.35, 0.0, -0.2])).copy()
            drift = self._fault_anim_plan.get("wind_dir", np.array([1.0, 0.0, 0.0]))
            self._debris_velocity = np.array([drift[0] * 1.8, drift[1] * 1.8, -1.0], dtype=float)

        if self._debris_active:
            age = t - (self._debris_start_time or t)
            if age > 2.2:
                self._debris_active = False
                if "demo_mass_debris" in self.plotter.renderer.actors:
                    self.plotter.renderer.actors["demo_mass_debris"].SetVisibility(False)
            else:
                start_pos = self._debris_start_pos
                velocity = self._debris_velocity
                if start_pos is None or velocity is None:
                    return
                g = 9.81
                debris_pos = start_pos + velocity * age + np.array([0.0, 0.0, -0.5 * g * age * age])
                if "demo_mass_debris" not in self.plotter.renderer.actors:
                    self.plotter.add_mesh(
                        pv.Sphere(radius=0.28),
                        name="demo_mass_debris",
                        color="#c89f65",
                        opacity=0.95,
                        smooth_shading=True,
                    )
                actor = self.plotter.renderer.actors.get("demo_mass_debris")
                if actor:
                    debris_tf = np.eye(4)
                    debris_tf[:3, 3] = debris_pos
                    actor.user_matrix = debris_tf
                    actor.SetVisibility(True)

    def _update_trail(self):
        """Throttle expensive trail redraws to improve FPS."""
        if abs(self.current_frame - self._last_trail_update_frame) < self._trail_update_stride:
            return
        self._last_trail_update_frame = self.current_frame
        super()._update_trail()

    def _update_camera(self, pos, vel, rot_matrix):
        """Responsive camera that never loses the rocket, especially under thrust."""
        if self._camera_mode == "Free":
            return

        # Detect if rocket is under thrust for higher camera urgency
        is_burning = False
        if self.df is not None and self.current_frame > 0 and "Mass" in self.df.columns:
            m_prev = float(self.df.iloc[self.current_frame - 1]["Mass"])
            m_curr = float(self.df.iloc[self.current_frame]["Mass"])
            is_burning = m_curr < m_prev - 1e-9

        speed = float(np.linalg.norm(vel))
        # Higher alpha when burning or at high speed to keep rocket in frame
        urgency_boost = 1.0
        if is_burning:
            urgency_boost = 2.5
        elif speed > 30.0:
            urgency_boost = 1.8

        if self._camera_mode == "Follow":
            # Follow: rigid offset from the rocket — the camera moves at exactly
            # the same speed as the rocket so it never falls behind or overshoots.
            follow_offset = np.array([0.0, -26.0, 18.0])
            target_pos = pos + follow_offset
            target_focal = pos

            # Rigidly snap camera to the target every frame (no smoothing lag).
            self._smooth_cam_pos = target_pos.copy()
            self._smooth_cam_focal = target_focal.copy()

        elif self._camera_mode == "Chase":
            base_alpha = 0.50
            vel_norm = np.linalg.norm(vel)
            vel_dir = vel / vel_norm if vel_norm > 0.5 else np.array([0.0, -1.0, 0.0])
            target_pos = pos - vel_dir * 24.0 + np.array([0.0, 0.0, 12.0])
            target_focal = pos + vel_dir * 2.0

            alpha = min(1.0, base_alpha * urgency_boost)

            if self._smooth_cam_pos is None:
                self._smooth_cam_pos = target_pos.copy()
                self._smooth_cam_focal = target_focal.copy()
            else:
                cam_dist = np.linalg.norm(target_pos - self._smooth_cam_pos)
                if cam_dist > 60.0:
                    snap_alpha = min(1.0, alpha + 0.4)
                    self._smooth_cam_pos += snap_alpha * (target_pos - self._smooth_cam_pos)
                    self._smooth_cam_focal += snap_alpha * (target_focal - self._smooth_cam_focal)
                else:
                    self._smooth_cam_pos += alpha * (target_pos - self._smooth_cam_pos)
                    self._smooth_cam_focal += alpha * (target_focal - self._smooth_cam_focal)

        elif self._camera_mode == "Orbit":
            base_alpha = 0.25
            self._orbit_angle += 0.03
            radius = 34.0
            target_pos = np.array([
                pos[0] + radius * math.cos(self._orbit_angle),
                pos[1] + radius * math.sin(self._orbit_angle),
                pos[2] + 14.0,
            ])
            target_focal = pos

            alpha = min(1.0, base_alpha * urgency_boost)

            if self._smooth_cam_pos is None:
                self._smooth_cam_pos = target_pos.copy()
                self._smooth_cam_focal = target_focal.copy()
            else:
                self._smooth_cam_pos += alpha * (target_pos - self._smooth_cam_pos)
                self._smooth_cam_focal += alpha * (target_focal - self._smooth_cam_focal)

        else:
            return

        try:
            # Pick a view-up vector that is never parallel to the camera direction
            cam_dir = self._smooth_cam_focal - self._smooth_cam_pos
            cam_dir_norm = np.linalg.norm(cam_dir)
            if cam_dir_norm > 1e-6:
                cam_dir_unit = cam_dir / cam_dir_norm
                
                # Compute an orthogonal up vector to prevent snapping
                global_up = np.array([0.0, 0.0, 1.0])
                dot_prod = np.dot(cam_dir_unit, global_up)
                
                if abs(dot_prod) > 0.999:
                    view_up = (0.0, 1.0, 0.0)
                else:
                    up_vec = global_up - dot_prod * cam_dir_unit
                    up_vec /= np.linalg.norm(up_vec)
                    view_up = tuple(up_vec)
            else:
                view_up = (0.0, 0.0, 1.0)
            self.plotter.camera_position = [
                tuple(self._smooth_cam_pos),
                tuple(self._smooth_cam_focal),
                view_up,
            ]
        except Exception:
            pass

    def update_scene(self):
        """Override parent update to align orientation with trajectory and attach exhaust to nozzle."""
        if self.df is None:
            return

        frac = 0.0
        if getattr(self, "is_playing", False) and getattr(self, "time_step", 1.0) > 0:
            frac = getattr(self, "sim_time_accumulator", 0.0) / self.time_step
            # prevent extrapolation issues if time_step logic gets weird
            frac = max(0.0, min(1.0, frac))

        next_frame = min(self.current_frame + 1, len(self.df) - 1)
        row0 = self.df.iloc[self.current_frame]
        row1 = self.df.iloc[next_frame]

        # Use row0 for discrete state logic (burn, etc)
        row = row0

        lat_scale = self._get_lat_scale()
        vert_scale = self._get_vert_scale()
        
        interp_X = float(row0["X"]) + frac * (float(row1["X"]) - float(row0["X"]))
        interp_Y = float(row0["Y"]) + frac * (float(row1["Y"]) - float(row0["Y"]))
        interp_Z = float(row0["Z"]) + frac * (float(row1["Z"]) - float(row0["Z"]))
        
        pos = np.array([
            interp_X * lat_scale,
            interp_Y * lat_scale,
            interp_Z * vert_scale,
        ], dtype=float)

        rot_matrix = self._compute_display_rotation()

        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = pos
        if self.rocket_actor:
            self.rocket_actor.user_matrix = transform

        is_burning = False
        if self.current_frame > 0 and "Mass" in self.df.columns:
            m_prev = float(self.df.iloc[self.current_frame - 1]["Mass"])
            m_curr = float(row["Mass"])
            is_burning = m_curr < m_prev - 1e-9

        if is_burning:
            self._create_animated_flame(transform, self._anim_clock)
        else:
            self._hide_flame()

        nozzle_local = np.array([0.0, 0.0, -0.8], dtype=float)
        if self.rocket_mesh is not None:
            b = self.rocket_mesh.bounds
            nozzle_local[2] = -0.5 * max((b[5] - b[4]), 1.0)
        plume_origin = pos + rot_matrix @ nozzle_local
        plume_dir = rot_matrix @ np.array([0.0, 0.0, -1.0], dtype=float)
        self._update_particles(plume_origin, plume_dir, is_burning, 0.016)

        interp_VX = float(row0.get("VX", 0.0)) + frac * (float(row1.get("VX", 0.0)) - float(row0.get("VX", 0.0)))
        interp_VY = float(row0.get("VY", 0.0)) + frac * (float(row1.get("VY", 0.0)) - float(row0.get("VY", 0.0)))
        interp_VZ = float(row0.get("VZ", 0.0)) + frac * (float(row1.get("VZ", 0.0)) - float(row0.get("VZ", 0.0)))

        vel = np.array([
            interp_VX * lat_scale,
            interp_VY * lat_scale,
            interp_VZ * vert_scale,
        ], dtype=float)
        speed = float(np.linalg.norm(vel))

        is_last_frame = self.current_frame >= len(self.df) - 1
        is_crashed = is_last_frame and speed > 5.0

        if is_crashed:
            if self.explosion_mesh and not self.explosion_actor:
                self.plotter.add_mesh(
                    self.explosion_mesh,
                    name="explosion",
                    color="red",
                    opacity=0.9,
                    emissive=True,
                )
                self.explosion_actor = self.plotter.renderer.actors.get("explosion")
            if self.explosion_actor:
                self.explosion_actor.SetVisibility(True)
                exp_transform = np.eye(4)
                exp_transform[:3, :3] = rot_matrix
                exp_transform[:3, 3] = pos
                self.explosion_actor.user_matrix = exp_transform
        elif self.explosion_actor:
            self.explosion_actor.SetVisibility(False)

        self._update_fault_effects_animation(row, pos, rot_matrix)
        self._update_trail()

        self.lbl_time.setText(f"T: {row['Time']:.2f}s")
        self._update_hud(row, is_burning, is_crashed)
        self._update_camera(pos, vel, rot_matrix)

        self.plotter.render()
        self._last_rendered_frame = self.current_frame

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Stop playback cleanly and close."""
        self.is_playing = False
        self.timer.stop()
        event.accept()
        super().closeEvent(event)


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark Fusion style (matches HexaVisual v2.0)
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 48))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

    window = RocketVisualizerDemo()
    window.show()
    sys.exit(app.exec_())

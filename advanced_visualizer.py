"""
HexaVisual v2.0 - Advanced 3D Flight Visualization for Project Vortex
=====================================================================
Major improvements over v1:
  - 2D Qt HUD overlays instead of slow 3D text recreation (lag reduction)
  - Batched geometry updates with dirty-flag render skipping (lag reduction)
  - Mesh LOD decimation for heavy STLs (lag reduction)
  - Google Maps / OpenStreetMap satellite imagery on ground plane
  - Procedural animated flame with multi-layer flickering + particle plume
  - Vertical scaling (independent of lateral scaling)
  - Trajectory trail visualization (past = solid, future = dashed)
  - Camera follow modes (Free / Follow / Chase / Orbit)
  - Enhanced telemetry HUD with color-coded margins
  - Improved lighting (sun light, SSAO, optional shadows)
"""

import sys
import os
import json
import hashlib
import math
import pandas as pd
import numpy as np
import time
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QWidget,
    QSlider, QComboBox, QGroupBox, QMessageBox, QCheckBox,
    QAction, QDialog, QScrollArea, QFrame, QGridLayout,
    QSplitter, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QPalette
from scipy.spatial.transform import Rotation as R
import ctypes

# Optional: PIL for satellite texture loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Optional: urllib for tile fetching
try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# ---------------------------------------------------------------------------
# Particle system for exhaust plume
# ---------------------------------------------------------------------------
class ExhaustParticleSystem:
    """Lightweight particle system for rocket exhaust plume visualization."""

    def __init__(self, max_particles=400):
        self.max_particles = max_particles
        self.positions = np.zeros((max_particles, 3), dtype=np.float64)
        self.velocities = np.zeros((max_particles, 3), dtype=np.float64)
        self.ages = np.zeros(max_particles, dtype=np.float64)
        self.lifetimes = np.zeros(max_particles, dtype=np.float64)
        self.alive = np.zeros(max_particles, dtype=bool)
        self.next_idx = 0
        self._rng = np.random.default_rng(42)

    def reset(self):
        self.alive[:] = False
        self.next_idx = 0

    def emit(self, origin, direction, count=8, spread=0.6, speed_base=15.0):
        """Spawn *count* particles at *origin* heading in *direction*."""
        for _ in range(count):
            idx = self.next_idx % self.max_particles
            self.next_idx += 1

            # Random spread around the main direction
            rand_dir = direction + self._rng.normal(0, spread, size=3)
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-6:
                rand_dir /= norm

            speed = speed_base * (0.7 + 0.6 * self._rng.random())
            self.positions[idx] = origin
            self.velocities[idx] = rand_dir * speed
            self.ages[idx] = 0.0
            self.lifetimes[idx] = 0.3 + 0.5 * self._rng.random()
            self.alive[idx] = True

    def step(self, dt):
        """Advance all living particles by *dt* seconds."""
        mask = self.alive
        self.positions[mask] += self.velocities[mask] * dt
        # Gravity + drag
        self.velocities[mask, 2] -= 3.0 * dt  # slight gravity pull
        self.velocities[mask] *= (1.0 - 1.5 * dt)  # drag
        # Turbulence
        self.positions[mask] += self._rng.normal(0, 0.15 * dt, size=(mask.sum(), 3))
        self.ages[mask] += dt
        # Kill old particles
        expired = self.ages > self.lifetimes
        self.alive[expired] = False

    def get_render_data(self):
        """Return (positions, scalars) for living particles. Scalars = normalized age."""
        mask = self.alive
        if not mask.any():
            return None, None
        pos = self.positions[mask].copy()
        ages = self.ages[mask]
        lt = self.lifetimes[mask]
        scalars = np.clip(ages / lt, 0, 1)  # 0 = young (hot), 1 = old (cool)
        return pos, scalars


# ---------------------------------------------------------------------------
# Satellite tile fetcher
# ---------------------------------------------------------------------------
class TileFetcher:
    """Download and cache satellite imagery tiles."""

    GOOGLE_URL = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        "center={lat},{lon}&zoom={zoom}&size=1024x1024"
        "&maptype=satellite&key={key}"
    )
    OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    ESRI_URL = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "assets", "map_tiles")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _meters_to_latlon(x_m, y_m, ref_lat, ref_lon):
        """Convert local ENU metres offset to lat/lon (simple approximation)."""
        lat = ref_lat + (y_m / 111320.0)
        lon = ref_lon + (x_m / (111320.0 * math.cos(math.radians(ref_lat))))
        return lat, lon

    @staticmethod
    def _latlon_to_tile(lat, lon, zoom):
        """Convert lat/lon + zoom to OSM tile x/y."""
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _cache_path(self, key_str):
        h = hashlib.md5(key_str.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.png")

    def fetch_google(self, lat, lon, zoom=18, api_key=""):
        if not api_key or not HAS_URLLIB or not HAS_PIL:
            return None
        url = self.GOOGLE_URL.format(lat=lat, lon=lon, zoom=zoom, key=api_key)
        cache = self._cache_path(f"google_{lat}_{lon}_{zoom}")
        if os.path.exists(cache):
            return cache
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HexaVisual/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(cache, "wb") as f:
                f.write(data)
            return cache
        except Exception as exc:
            print(f"[TileFetcher] Google fetch failed: {exc}")
            return None

    def fetch_esri(self, lat, lon, zoom=17):
        """Fetch ESRI World Imagery tile (free, no API key)."""
        if not HAS_URLLIB or not HAS_PIL:
            return None
        tx, ty = self._latlon_to_tile(lat, lon, zoom)
        url = self.ESRI_URL.format(z=zoom, x=tx, y=ty)
        cache = self._cache_path(f"esri_{zoom}_{tx}_{ty}")
        if os.path.exists(cache):
            return cache
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HexaVisual/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(cache, "wb") as f:
                f.write(data)
            return cache
        except Exception as exc:
            print(f"[TileFetcher] ESRI fetch failed: {exc}")
            return None

    def fetch_osm(self, lat, lon, zoom=17):
        """Fetch OpenStreetMap tile (fallback, free, no key)."""
        if not HAS_URLLIB or not HAS_PIL:
            return None
        tx, ty = self._latlon_to_tile(lat, lon, zoom)
        url = self.OSM_URL.format(z=zoom, x=tx, y=ty)
        cache = self._cache_path(f"osm_{zoom}_{tx}_{ty}")
        if os.path.exists(cache):
            return cache
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HexaVisual/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(cache, "wb") as f:
                f.write(data)
            return cache
        except Exception as exc:
            print(f"[TileFetcher] OSM fetch failed: {exc}")
            return None

    def get_texture(self, lat, lon, zoom=17, api_key="", size_m=2000):
        """Attempt to fetch imagery and return a PyVista texture, or None."""
        path = None
        # Priority: Google (if key) > ESRI (free satellite) > OSM
        if api_key:
            path = self.fetch_google(lat, lon, zoom, api_key)
        if path is None:
            path = self.fetch_esri(lat, lon, zoom)
        if path is None:
            path = self.fetch_osm(lat, lon, zoom)
        if path and HAS_PIL:
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((1024, 1024), Image.LANCZOS)
                arr = np.array(img)
                texture = pv.numpy_to_texture(arr)
                return texture
            except Exception as exc:
                print(f"[TileFetcher] Texture load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main Visualizer
# ---------------------------------------------------------------------------
class RocketVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexaVisual v2.0 - Vortex Desktop")
        self.resize(1400, 900)

        # Window icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "hexavisual_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            myappid = "vortex.hexavisual.v2.0"
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass

        # --- Data ---
        self.df = None
        self.rocket_mesh = None
        self.custom_flame_mesh = None
        self.rocket_actor = None
        self.flame_actors = []  # list of multi-layer flame actors
        self.ground_actor = None
        self.explosion_mesh = None
        self.explosion_actor = None
        self.trail_actor = None
        self.particle_actor = None

        # --- State ---
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.time_step = 0.01
        self.last_real_time = 0
        self.sim_time_accumulator = 0
        self._scene_dirty = True
        self._last_rendered_frame = -1
        self._anim_clock = 0.0  # continuous clock for flame animation

        # Particle system
        self.particles = ExhaustParticleSystem(max_particles=400)

        # Tile fetcher
        self.tile_fetcher = TileFetcher()

        # Camera mode state
        self._camera_mode = "Free"
        self._orbit_angle = 0.0
        self._smooth_cam_pos = None
        self._smooth_cam_focal = None

        # Load prefs first (populates defaults)
        self.load_preferences()

        # Build UI
        self.init_ui()

        # Default scene objects
        self.create_ground()
        self.create_default_rocket()
        self.load_explosion_asset()

        # Apply satellite imagery if configured
        self._apply_satellite_texture()

        # Setup improved lighting
        self._setup_lighting()

    # ------------------------------------------------------------------
    # Preferences I/O
    # ------------------------------------------------------------------
    def load_preferences(self):
        pref_path = os.path.join(os.path.dirname(__file__), "hexavisual_prefs.json")
        # Defaults
        self.results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
        self.default_rocket_stl = ""
        self.flame_stl_path = ""
        self.explosion_stl_path = os.path.join("assets", "Dramatic_explosion_effect_part.stl")
        self.lat_scale_val = "10.0"
        self.scale_lat_enabled = False
        self.vert_scale_val = "1.0"
        self.scale_vert_enabled = False
        self.default_playback_speed = "1.0x"
        # Map / satellite prefs
        self.launch_lat = 32.990
        self.launch_lon = -106.975
        self.map_api_key = ""
        self.map_zoom = 17
        self.map_enabled = True
        # Camera
        self.default_camera_mode = "Free"
        # Lighting
        self.enable_ssao = True
        self.enable_shadows = False
        # Trail
        self.show_trail = True
        # Mesh LOD
        self.mesh_decimate_target = 0.5  # keep 50% of triangles for heavy meshes
        self.mesh_decimate_threshold = 50000  # only decimate if n_faces > this

        if os.path.exists(pref_path):
            try:
                with open(pref_path, "r") as f:
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
                self.mesh_decimate_target = float(prefs.get("mesh_decimate_target", self.mesh_decimate_target))
                self.mesh_decimate_threshold = int(prefs.get("mesh_decimate_threshold", self.mesh_decimate_threshold))
            except Exception as e:
                print(f"[Prefs] Error loading preferences: {e}")

    def save_preferences(self):
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

        pref_path = os.path.join(os.path.dirname(__file__), "hexavisual_prefs.json")
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
        }
        try:
            with open(pref_path, "w") as f:
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

        # Re-apply satellite
        self._apply_satellite_texture()
        # Re-apply lighting
        self._setup_lighting()

        self.pref_dialog.accept()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Menu
        self.create_menu()

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        # --- 3D viewport container (with HUD overlay) ---
        viewport_container = QWidget()
        viewport_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        viewport_layout.addWidget(self.plotter.interactor)

        # --- HUD overlay (Qt labels on top of 3D view) ---
        self._build_hud_overlay(viewport_container)

        main_layout.addWidget(viewport_container, stretch=1)

        # --- Controls ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(2, 2, 2, 2)
        controls_layout.setSpacing(2)

        # Row 1: Loading + Scaling
        row1 = QHBoxLayout()

        # Loading group
        files_group = QGroupBox("Loading")
        files_layout = QHBoxLayout()
        files_layout.setSpacing(4)

        self.btn_load_csv = QPushButton("Load CSV Run")
        self.btn_load_csv.clicked.connect(self.load_csv)
        files_layout.addWidget(self.btn_load_csv)

        self.btn_load_stl = QPushButton("Load Rocket STL")
        self.btn_load_stl.clicked.connect(self.load_stl)
        files_layout.addWidget(self.btn_load_stl)

        files_group.setLayout(files_layout)
        row1.addWidget(files_group)

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

        scale_layout.addWidget(QLabel("×"))

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

        main_layout.addWidget(controls_widget, stretch=0)

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _build_hud_overlay(self, parent):
        """Create transparent Qt label overlays for telemetry HUD."""
        # Container that floats over the 3D viewport
        self.hud_widget = QWidget(parent)
        self.hud_widget.setStyleSheet("background: transparent;")
        self.hud_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud_widget.setGeometry(10, 10, 320, 280)

        hud_layout = QVBoxLayout(self.hud_widget)
        hud_layout.setContentsMargins(8, 8, 8, 8)
        hud_layout.setSpacing(2)
        hud_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        hud_font = QFont("Consolas", 11)
        hud_font.setBold(True)

        def make_label(text="", color="#00FF00"):
            lbl = QLabel(text)
            lbl.setFont(hud_font)
            lbl.setStyleSheet(
                f"color: {color}; background-color: rgba(0,0,0,160); "
                f"padding: 2px 6px; border-radius: 3px;"
            )
            lbl.setAlignment(Qt.AlignLeft)
            hud_layout.addWidget(lbl)
            return lbl

        self.hud_title = make_label("HexaVisual v2.0", "#00BFFF")
        self.hud_time = make_label("T: 0.000 s")
        self.hud_alt = make_label("ALT: 0.0 m")
        self.hud_vvel = make_label("V.VEL: 0.0 m/s")
        self.hud_hvel = make_label("H.VEL: 0.0 m/s")
        self.hud_speed = make_label("SPEED: 0.0 m/s")
        self.hud_mass = make_label("MASS: 0.0 kg")
        self.hud_status = make_label("STATUS: IDLE", "#FFFF00")
        self.hud_tti = make_label("")  # Time to impact

        hud_layout.addStretch()
        self.hud_widget.raise_()

    def _update_hud(self, row, is_burning, is_crashed):
        """Update HUD labels from current data row. Fast — just setText calls."""
        t = row["Time"]
        z = row["Z"]
        vx = row.get("VX", 0.0)
        vy = row.get("VY", 0.0)
        vz = row.get("VZ", 0.0)
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        h_vel = math.sqrt(vx**2 + vy**2)
        mass = row.get("Mass", 0.0)

        self.hud_time.setText(f"T: {t:.3f} s")

        # Altitude — color by proximity to ground
        if z < 10:
            alt_color = "#FF4444"
        elif z < 50:
            alt_color = "#FFAA00"
        else:
            alt_color = "#00FF00"
        self.hud_alt.setText(f"ALT: {z:.1f} m")
        self.hud_alt.setStyleSheet(
            f"color: {alt_color}; background-color: rgba(0,0,0,160); "
            f"padding: 2px 6px; border-radius: 3px;"
        )

        # Vertical velocity — color by magnitude
        if abs(vz) > 50:
            vv_color = "#FF4444"
        elif abs(vz) > 20:
            vv_color = "#FFAA00"
        else:
            vv_color = "#00FF00"
        self.hud_vvel.setText(f"V.VEL: {vz:.1f} m/s")
        self.hud_vvel.setStyleSheet(
            f"color: {vv_color}; background-color: rgba(0,0,0,160); "
            f"padding: 2px 6px; border-radius: 3px;"
        )

        self.hud_hvel.setText(f"H.VEL: {h_vel:.1f} m/s")
        self.hud_speed.setText(f"SPEED: {speed:.1f} m/s")
        self.hud_mass.setText(f"MASS: {mass:.2f} kg")

        # Status
        if is_crashed:
            self.hud_status.setText("STATUS: CRASHED")
            self.hud_status.setStyleSheet(
                "color: #FF0000; background-color: rgba(80,0,0,200); "
                "padding: 2px 6px; border-radius: 3px;"
            )
        elif is_burning:
            self.hud_status.setText("STATUS: BURN")
            self.hud_status.setStyleSheet(
                "color: #FF8800; background-color: rgba(0,0,0,160); "
                "padding: 2px 6px; border-radius: 3px;"
            )
        elif self.is_playing:
            self.hud_status.setText("STATUS: COAST")
            self.hud_status.setStyleSheet(
                "color: #00BFFF; background-color: rgba(0,0,0,160); "
                "padding: 2px 6px; border-radius: 3px;"
            )
        else:
            self.hud_status.setText("STATUS: PAUSED")
            self.hud_status.setStyleSheet(
                "color: #AAAAAA; background-color: rgba(0,0,0,160); "
                "padding: 2px 6px; border-radius: 3px;"
            )

        # Time-to-impact estimate
        if z > 0.5 and vz < -0.5:
            tti = -z / vz
            self.hud_tti.setText(f"TTI: {tti:.1f} s")
            tti_color = "#FF4444" if tti < 2.0 else "#FFAA00" if tti < 5.0 else "#00FF00"
            self.hud_tti.setStyleSheet(
                f"color: {tti_color}; background-color: rgba(0,0,0,160); "
                f"padding: 2px 6px; border-radius: 3px;"
            )
            self.hud_tti.setVisible(True)
        else:
            self.hud_tti.setVisible(False)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def create_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("File")

        import_action = QAction("Import from HexaKinetic", self)
        import_action.triggered.connect(self.open_import_dialog)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        about_action = QAction("About", self)
        about_action.triggered.connect(self.open_about)
        file_menu.addAction(about_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit
        edit_menu = menubar.addMenu("Edit")

        pref_action = QAction("Preferences", self)
        pref_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(pref_action)

        edit_menu.addSeparator()

        launch_gui_action = QAction("Launch HexaKinetic Simulation", self)
        launch_gui_action.triggered.connect(self.launch_main_gui)
        edit_menu.addAction(launch_gui_action)

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
        toggle_trail_action.triggered.connect(lambda: self.chk_trail.setChecked(not self.chk_trail.isChecked()))
        view_menu.addAction(toggle_trail_action)

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------
    def open_about(self):
        QMessageBox.about(
            self,
            "About",
            "HexaVisual v2.0\n\n"
            "Advanced 3D flight visualization for Project Vortex\n\n"
            "Features: Animated flame, satellite imagery, trajectory trails,\n"
            "camera modes, particle plume, telemetry HUD\n\n"
            "© 2026 Agastya Mishra",
        )

    def launch_main_gui(self):
        import subprocess
        try:
            subprocess.Popen([sys.executable, "gui.py"])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch GUI: {e}")

    def open_preferences(self):
        self.pref_dialog = QDialog(self)
        self.pref_dialog.setWindowTitle("Preferences")
        self.pref_dialog.resize(550, 650)
        layout = QVBoxLayout()

        # --- Files ---
        files_group = QGroupBox("File Paths")
        fl = QVBoxLayout()

        fl.addWidget(QLabel("Results Directory:"))
        res_row = QHBoxLayout()
        self.inp_res_dir = QLineEdit(self.results_dir)
        res_row.addWidget(self.inp_res_dir)
        btn_br_res = QPushButton("Browse")
        btn_br_res.clicked.connect(
            lambda: self.inp_res_dir.setText(
                QFileDialog.getExistingDirectory(self, "Select Results Dir") or self.inp_res_dir.text()
            )
        )
        res_row.addWidget(btn_br_res)
        fl.addLayout(res_row)

        fl.addWidget(QLabel("Default Rocket STL:"))
        rock_row = QHBoxLayout()
        self.inp_rock_stl = QLineEdit(self.default_rocket_stl)
        rock_row.addWidget(self.inp_rock_stl)
        btn_br_rock = QPushButton("Browse")
        btn_br_rock.clicked.connect(
            lambda: self.inp_rock_stl.setText(
                QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0] or self.inp_rock_stl.text()
            )
        )
        rock_row.addWidget(btn_br_rock)
        fl.addLayout(rock_row)

        fl.addWidget(QLabel("Flame STL (optional):"))
        flame_row = QHBoxLayout()
        self.inp_flame_stl = QLineEdit(self.flame_stl_path)
        flame_row.addWidget(self.inp_flame_stl)
        btn_br_flame = QPushButton("Browse")
        btn_br_flame.clicked.connect(
            lambda: self.inp_flame_stl.setText(
                QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0] or self.inp_flame_stl.text()
            )
        )
        flame_row.addWidget(btn_br_flame)
        fl.addLayout(flame_row)

        fl.addWidget(QLabel("Explosion STL:"))
        exp_row = QHBoxLayout()
        self.inp_exp_stl = QLineEdit(self.explosion_stl_path)
        exp_row.addWidget(self.inp_exp_stl)
        btn_br_exp = QPushButton("Browse")
        btn_br_exp.clicked.connect(
            lambda: self.inp_exp_stl.setText(
                QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0] or self.inp_exp_stl.text()
            )
        )
        exp_row.addWidget(btn_br_exp)
        fl.addLayout(exp_row)

        files_group.setLayout(fl)
        layout.addWidget(files_group)

        # --- Map / Satellite ---
        map_group = QGroupBox("Satellite Imagery")
        ml = QGridLayout()

        self.chk_map_enabled = QCheckBox("Enable satellite ground texture")
        self.chk_map_enabled.setChecked(self.map_enabled)
        ml.addWidget(self.chk_map_enabled, 0, 0, 1, 2)

        ml.addWidget(QLabel("Launch Latitude:"), 1, 0)
        self.inp_launch_lat = QLineEdit(str(self.launch_lat))
        ml.addWidget(self.inp_launch_lat, 1, 1)

        ml.addWidget(QLabel("Launch Longitude:"), 2, 0)
        self.inp_launch_lon = QLineEdit(str(self.launch_lon))
        ml.addWidget(self.inp_launch_lon, 2, 1)

        ml.addWidget(QLabel("Map Zoom (14-19):"), 3, 0)
        self.inp_map_zoom = QLineEdit(str(self.map_zoom))
        ml.addWidget(self.inp_map_zoom, 3, 1)

        ml.addWidget(QLabel("Google API Key (optional):"), 4, 0)
        self.inp_map_key = QLineEdit(self.map_api_key)
        self.inp_map_key.setEchoMode(QLineEdit.Password)
        ml.addWidget(self.inp_map_key, 4, 1)

        ml.addWidget(QLabel("(Leave blank for free ESRI/OSM tiles)"), 5, 0, 1, 2)

        map_group.setLayout(ml)
        layout.addWidget(map_group)

        # --- Rendering ---
        render_group = QGroupBox("Rendering")
        rl = QVBoxLayout()

        self.chk_ssao = QCheckBox("Screen-Space Ambient Occlusion (SSAO)")
        self.chk_ssao.setChecked(self.enable_ssao)
        rl.addWidget(self.chk_ssao)

        self.chk_shadows = QCheckBox("Shadows (may reduce performance)")
        self.chk_shadows.setChecked(self.enable_shadows)
        rl.addWidget(self.chk_shadows)

        render_group.setLayout(rl)
        layout.addWidget(render_group)

        # Save button
        btn_save = QPushButton("Save Preferences")
        btn_save.setStyleSheet(
            "QPushButton { background-color: #007bff; color: white; padding: 8px; "
            "border-radius: 4px; font-weight: bold; } "
            "QPushButton:hover { background-color: #0056b3; }"
        )
        btn_save.clicked.connect(self.save_preferences)
        layout.addWidget(btn_save)

        self.pref_dialog.setLayout(layout)
        self.pref_dialog.exec_()

    def open_import_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Import from HexaKinetic")
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)

        header = QHBoxLayout()
        header.addWidget(QLabel("Available Runs"))
        chk_opt = QCheckBox("Show Optimization Runs")
        header.addWidget(chk_opt)
        header.addStretch()
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.cards_layout = QGridLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        path = self.results_dir
        if os.path.exists(path):
            row_idx, col_idx = 0, 0
            try:
                dirs = [
                    d
                    for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d)) and d.startswith("single_run_")
                ]
                dirs.sort(reverse=True)

                for d in dirs:
                    folder_path = os.path.join(path, d)
                    csv_path = os.path.join(folder_path, "single_run.csv")
                    png_path = os.path.join(folder_path, "trajectory_3d.png")

                    if os.path.exists(csv_path):
                        pts = d.split("_")
                        if len(pts) >= 4:
                            date_str = pts[2]
                            time_str = pts[3]
                            try:
                                formatted_dt = (
                                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} "
                                    f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                                )
                            except Exception:
                                formatted_dt = d
                        else:
                            formatted_dt = d

                        card = QFrame()
                        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
                        card.setFixedSize(260, 260)
                        card.setStyleSheet(
                            "QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; }"
                        )
                        cl = QVBoxLayout(card)

                        lbl_name = QLabel(formatted_dt)
                        lbl_name.setAlignment(Qt.AlignCenter)
                        lbl_name.setStyleSheet("font-weight: bold; color: #333;")
                        cl.addWidget(lbl_name)

                        thumb = QLabel()
                        thumb.setAlignment(Qt.AlignCenter)
                        thumb.setMinimumHeight(150)
                        if os.path.exists(png_path):
                            pix = QPixmap(png_path)
                            thumb.setPixmap(
                                pix.scaled(240, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            )
                        else:
                            thumb.setText("No Preview Available")
                            thumb.setStyleSheet(
                                "background-color: #e9ecef; color: #6c757d; border-radius: 4px;"
                            )
                        cl.addWidget(thumb, 1)

                        btn_load = QPushButton("Load Simulation")
                        btn_load.setStyleSheet(
                            "QPushButton { background-color: #007bff; color: white; "
                            "border-radius: 4px; padding: 5px; } "
                            "QPushButton:hover { background-color: #0056b3; }"
                        )
                        btn_load.clicked.connect(
                            lambda checked, p=csv_path: [self.load_csv_file(p), dlg.accept()]
                        )
                        cl.addWidget(btn_load)

                        self.cards_layout.addWidget(card, row_idx, col_idx)
                        col_idx += 1
                        if col_idx > 2:
                            col_idx = 0
                            row_idx += 1

                if not dirs:
                    layout.addWidget(QLabel("No 'single_run_' folders found in the results directory."))

            except Exception as e:
                layout.addWidget(QLabel(f"Error scanning results: {e}"))
        else:
            layout.addWidget(QLabel(f"Results directory not found: {path}\nPlease check Preferences."))

        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(dlg.reject)
        layout.addWidget(btn_cancel)

        dlg.exec_()

    # ------------------------------------------------------------------
    # Scene creation
    # ------------------------------------------------------------------
    def create_ground(self):
        """Create textured ground plane with UV coordinates for satellite imagery."""
        self.ground_mesh = pv.Plane(
            center=(0, 0, 0), direction=(0, 0, 1), i_size=2000, j_size=2000,
            i_resolution=1, j_resolution=1,
        )
        # The plane already has texture coordinates from PyVista — use them
        self.plotter.add_mesh(
            self.ground_mesh,
            name="ground",
            color="dimgray",
            opacity=0.7,
            show_edges=False,
            lighting=True,
        )
        self.ground_actor = self.plotter.renderer.actors.get("ground")

    def _apply_satellite_texture(self):
        """Fetch and apply satellite imagery to the ground plane."""
        if not self.map_enabled:
            return
        texture = self.tile_fetcher.get_texture(
            self.launch_lat, self.launch_lon,
            zoom=self.map_zoom, api_key=self.map_api_key,
        )
        if texture is not None:
            try:
                # Re-add ground with texture
                self.plotter.add_mesh(
                    self.ground_mesh,
                    name="ground",
                    texture=texture,
                    opacity=1.0,
                    show_edges=False,
                    lighting=True,
                )
                self.ground_actor = self.plotter.renderer.actors.get("ground")
                print("[HexaVisual] Satellite imagery applied to ground plane.")
            except Exception as exc:
                print(f"[HexaVisual] Failed to apply satellite texture: {exc}")

    def _setup_lighting(self):
        """Configure scene lighting for realism."""
        try:
            # Clear default lights and add a sun-like directional light
            self.plotter.renderer.RemoveAllLights()
            sun = pv.Light(
                position=(500, 500, 2000),
                focal_point=(0, 0, 0),
                color="white",
                intensity=1.0,
                light_type="scene",
            )
            self.plotter.add_light(sun)

            # Ambient fill light
            fill = pv.Light(
                position=(-500, -200, 500),
                focal_point=(0, 0, 0),
                color=(0.6, 0.65, 0.8),
                intensity=0.3,
                light_type="scene",
            )
            self.plotter.add_light(fill)

            # SSAO for depth
            if self.enable_ssao:
                try:
                    self.plotter.enable_ssao(radius=15, bias=0.5, kernel_size=128)
                except Exception:
                    pass

            # Shadows (optional, can be slow)
            if self.enable_shadows:
                try:
                    self.plotter.enable_shadows()
                except Exception:
                    pass

        except Exception as exc:
            print(f"[Lighting] Setup error (non-fatal): {exc}")

    def update_ground_level(self):
        if self.ground_actor:
            self.ground_actor.position = (0, 0, 0)

    def create_default_rocket(self):
        if hasattr(self, "default_rocket_stl") and self.default_rocket_stl and os.path.exists(self.default_rocket_stl):
            try:
                mesh = pv.read(self.default_rocket_stl)
                self.apply_rocket_mesh(mesh)
                return
            except Exception:
                pass
        # Fallback: procedural rocket
        body = pv.Cylinder(radius=0.15, height=5.0, direction=(0, 0, 1), resolution=32)
        nose = pv.Cone(center=(0, 0, 5.5), direction=(0, 0, 1), height=1.5, radius=0.15, resolution=32)
        mesh = body.merge(nose)
        self.apply_rocket_mesh(mesh)

    def apply_rocket_mesh(self, mesh):
        """Standardize, optionally decimate, and add rocket mesh."""
        # LOD decimation for heavy meshes
        if mesh.n_cells > self.mesh_decimate_threshold:
            try:
                target_reduction = 1.0 - self.mesh_decimate_target
                mesh = mesh.decimate(target_reduction)
                print(f"[LOD] Decimated mesh: {mesh.n_cells} faces remaining")
            except Exception as exc:
                print(f"[LOD] Decimation failed (non-fatal): {exc}")

        b = mesh.bounds
        x_center = (b[0] + b[1]) / 2
        y_center = (b[2] + b[3]) / 2
        z_min = b[4]
        mesh.translate([-x_center, -y_center, -z_min], inplace=True)

        self.rocket_mesh = mesh
        self.plotter.add_mesh(
            self.rocket_mesh, name="rocket", color="silver",
            smooth_shading=True, pbr=True, metallic=0.7, roughness=0.3,
        )
        self.rocket_actor = self.plotter.renderer.actors.get("rocket")
        self.update_ground_level()
        self._scene_dirty = True

    def load_explosion_asset(self):
        path = self.explosion_stl_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        self.load_explosion_asset_path(path)

    def load_explosion_asset_path(self, path):
        if os.path.exists(path):
            try:
                mesh = pv.read(path)
                b = mesh.bounds
                center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
                mesh.translate([-c for c in center], inplace=True)
                mesh.scale(2.0, inplace=True)
                self.explosion_mesh = mesh
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CSV / STL loading
    # ------------------------------------------------------------------
    def load_csv_file(self, path):
        try:
            self.df = pd.read_csv(path)
            required = ["Time", "X", "Y", "Z"]
            if not all(col in self.df.columns for col in required):
                raise ValueError("CSV missing required columns: Time, X, Y, Z")

            self.slider.setRange(0, len(self.df) - 1)
            self.slider.setValue(0)
            self.current_frame = 0
            self.is_playing = False
            self.btn_play.setText("Play")
            self.timer.stop()

            if len(self.df) > 1:
                self.time_step = self.df["Time"].iloc[1] - self.df["Time"].iloc[0]

            # Pre-compute trajectory for trail
            self._precompute_trail()

            # Reset particles
            self.particles.reset()

            self._scene_dirty = True
            self._last_rendered_frame = -1
            self.update_scene()
            self.plotter.reset_camera()
            print(f"[HexaVisual] Loaded {len(self.df)} frames from {path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV Simulation", "", "CSV Files (*.csv)")
        if path:
            self.load_csv_file(path)

    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Rocket STL", "", "STL Files (*.stl)")
        if not path:
            return
        try:
            mesh = pv.read(path)
            self.apply_rocket_mesh(mesh)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load STL: {e}")

    def load_custom_flame(self, path):
        try:
            mesh = pv.read(path)
            b = mesh.bounds
            center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
            mesh.translate([-c for c in center], inplace=True)
            self.custom_flame_mesh = mesh
            # Reset flame actors
            self._remove_flame_actors()
            self._scene_dirty = True
        except Exception:
            pass

    def load_flame_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Flame STL", "", "STL Files (*.stl)")
        if not path:
            return
        try:
            mesh = pv.read(path)
            b = mesh.bounds
            center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
            mesh.translate([-c for c in center], inplace=True)
            self.custom_flame_mesh = mesh
            self._remove_flame_actors()
            self._scene_dirty = True
            self.update_scene()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Flame STL: {e}")

    # ------------------------------------------------------------------
    # Trail / trajectory visualization
    # ------------------------------------------------------------------
    def _precompute_trail(self):
        """Pre-compute full trajectory polyline for efficient rendering."""
        if self.df is None:
            return
        lat_scale = self._get_lat_scale()
        vert_scale = self._get_vert_scale()

        xs = self.df["X"].values * lat_scale
        ys = self.df["Y"].values * lat_scale
        zs = self.df["Z"].values * vert_scale
        self._trail_points = np.column_stack([xs, ys, zs])

    def _update_trail(self):
        """Update trajectory trail visualization, coloring past vs future."""
        if not self.show_trail or self.df is None or not hasattr(self, "_trail_points"):
            if self.trail_actor and "trail" in self.plotter.renderer.actors:
                self.plotter.renderer.actors["trail"].SetVisibility(False)
            return

        n = len(self._trail_points)
        if n < 2:
            return

        # Recompute with current scaling
        lat_scale = self._get_lat_scale()
        vert_scale = self._get_vert_scale()
        pts = np.column_stack([
            self.df["X"].values * lat_scale,
            self.df["Y"].values * lat_scale,
            self.df["Z"].values * vert_scale,
        ])

        # Create scalars: 0 = start, current_frame value = transition point, n-1 = end
        scalars = np.zeros(n, dtype=np.float64)
        scalars[:self.current_frame + 1] = 1.0  # Past = 1.0
        scalars[self.current_frame + 1:] = 0.3   # Future = 0.3

        try:
            poly = pv.lines_from_points(pts)
            # Point scalars — need n values, polyline has n points
            self.plotter.add_mesh(
                poly, name="trail",
                scalars=scalars, cmap="cool",
                line_width=2, opacity=0.8,
                show_scalar_bar=False,
            )
            self.trail_actor = self.plotter.renderer.actors.get("trail")
        except Exception:
            pass

    def _toggle_trail(self):
        self.show_trail = self.chk_trail.isChecked()
        self._scene_dirty = True
        if self.df is not None:
            self.update_scene()

    # ------------------------------------------------------------------
    # Flame animation
    # ------------------------------------------------------------------
    def _remove_flame_actors(self):
        """Remove all flame layer actors from scene."""
        for name in ["flame_core", "flame_mid", "flame_outer", "flame"]:
            if name in self.plotter.renderer.actors:
                try:
                    self.plotter.remove_actor(name)
                except Exception:
                    pass
        self.flame_actors = []

    def _create_animated_flame(self, transform, anim_time):
        """
        Create/update multi-layer animated flame with flickering.
        Returns True if flame was created/updated.
        """
        if self.rocket_mesh is None:
            return False

        b_rock = self.rocket_mesh.bounds
        rocket_width = max(b_rock[1] - b_rock[0], b_rock[3] - b_rock[2])
        base_radius = max(0.4, rocket_width * 0.8)

        # Flickering parameters driven by anim_time
        flicker1 = math.sin(anim_time * 25.0) * 0.15 + 1.0
        flicker2 = math.sin(anim_time * 18.0 + 1.2) * 0.2 + 1.0
        flicker3 = math.sin(anim_time * 30.0 + 2.5) * 0.25 + 1.0
        height_pulse = math.sin(anim_time * 15.0) * 0.2 + 1.0

        layers = [
            # (name, radius_mult, height, color, opacity, z_offset_of_center)
            ("flame_core",  0.35 * flicker1, 3.5 * height_pulse, (1.0, 0.95, 0.7), 1.0,  -1.75 * height_pulse),
            ("flame_mid",   0.65 * flicker2, 5.0 * height_pulse, (1.0, 0.55, 0.1), 0.85, -2.5 * height_pulse),
            ("flame_outer", 1.0 * flicker3,  6.5 * height_pulse, (1.0, 0.25, 0.0), 0.55, -3.25 * height_pulse),
        ]

        for name, r_mult, h, color, opacity, z_center in layers:
            radius = base_radius * r_mult
            cone = pv.Cone(
                center=(0, 0, z_center),
                direction=(0, 0, 1),
                height=h,
                radius=radius,
                resolution=24,
            )
            self.plotter.add_mesh(
                cone, name=name,
                color=color, opacity=opacity,
                emissive=True, lighting=False,
                smooth_shading=True,
            )
            actor = self.plotter.renderer.actors.get(name)
            if actor:
                actor.user_matrix = transform
                actor.SetVisibility(True)

        return True

    def _hide_flame(self):
        """Hide all flame layers."""
        for name in ["flame_core", "flame_mid", "flame_outer", "flame"]:
            if name in self.plotter.renderer.actors:
                self.plotter.renderer.actors[name].SetVisibility(False)

    # ------------------------------------------------------------------
    # Particle plume
    # ------------------------------------------------------------------
    def _update_particles(self, origin, direction, is_burning, dt):
        """Emit and step particles, then render."""
        if is_burning:
            self.particles.emit(origin, direction, count=10, spread=0.5, speed_base=18.0)

        self.particles.step(dt)

        pos, scalars = self.particles.get_render_data()
        if pos is not None and len(pos) > 1:
            try:
                cloud = pv.PolyData(pos)
                cloud["age"] = scalars
                self.plotter.add_mesh(
                    cloud, name="particles",
                    scalars="age", cmap="hot_r",
                    point_size=6, render_points_as_spheres=True,
                    opacity=0.7, show_scalar_bar=False,
                    lighting=False,
                )
                self.particle_actor = self.plotter.renderer.actors.get("particles")
            except Exception:
                pass
        else:
            if "particles" in self.plotter.renderer.actors:
                self.plotter.renderer.actors["particles"].SetVisibility(False)

    # ------------------------------------------------------------------
    # Camera modes
    # ------------------------------------------------------------------
    def _on_camera_mode_changed(self, mode):
        self._camera_mode = mode
        self._smooth_cam_pos = None
        self._smooth_cam_focal = None
        self._orbit_angle = 0.0

    def _update_camera(self, pos, vel, rot_matrix):
        """Update camera position based on selected mode."""
        if self._camera_mode == "Free":
            return  # user controls camera

        # Smoothing factor
        alpha = 0.08

        if self._camera_mode == "Follow":
            # Follow from behind and above
            offset = np.array([0.0, -30.0, 20.0])
            target_pos = pos + offset
            target_focal = pos

        elif self._camera_mode == "Chase":
            # Chase camera: behind the rocket along velocity vector
            vel_norm = np.linalg.norm(vel)
            if vel_norm > 0.5:
                vel_dir = vel / vel_norm
            else:
                vel_dir = np.array([0.0, -1.0, 0.0])
            target_pos = pos - vel_dir * 30.0 + np.array([0, 0, 10.0])
            target_focal = pos

        elif self._camera_mode == "Orbit":
            # Orbit around rocket
            self._orbit_angle += 0.02
            radius = 40.0
            cx = pos[0] + radius * math.cos(self._orbit_angle)
            cy = pos[1] + radius * math.sin(self._orbit_angle)
            cz = pos[2] + 15.0
            target_pos = np.array([cx, cy, cz])
            target_focal = pos
        else:
            return

        # Smooth interpolation
        if self._smooth_cam_pos is None:
            self._smooth_cam_pos = target_pos.copy()
            self._smooth_cam_focal = target_focal.copy()
        else:
            self._smooth_cam_pos += alpha * (target_pos - self._smooth_cam_pos)
            self._smooth_cam_focal += alpha * (target_focal - self._smooth_cam_focal)

        try:
            self.plotter.camera_position = [
                tuple(self._smooth_cam_pos),
                tuple(self._smooth_cam_focal),
                (0, 0, 1),
            ]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------
    def toggle_play(self):
        if self.df is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            self.last_real_time = time.time()
            self._anim_clock = 0.0
            self.timer.start(16)  # 60 FPS target
        else:
            self.btn_play.setText("Play")
            self.timer.stop()
            self._scene_dirty = True
            self.update_scene()  # one final render to update HUD status

    def update_speed(self, text):
        try:
            speed_str = text.replace("x", "")
            self.playback_speed = float(speed_str)
        except (ValueError, ZeroDivisionError):
            pass

    def skip_time(self, direction):
        if self.df is None:
            return
        try:
            skip_sec = float(self.inp_skip.text())
            frames_to_skip = int(skip_sec / self.time_step)
            new_frame = self.current_frame + (direction * frames_to_skip)
            new_frame = max(0, min(new_frame, len(self.df) - 1))
            self.slider.setValue(new_frame)
        except ValueError:
            pass

    def on_slider_change(self, val):
        self.current_frame = val
        self._scene_dirty = True
        self.update_scene()

    def _on_scale_changed(self, *args):
        """Called when any scale checkbox/value changes."""
        self._scene_dirty = True
        if self.df is not None:
            self._precompute_trail()
            self.update_scene()

    def _get_lat_scale(self):
        try:
            return float(self.inp_lat_scale.text()) if self.chk_scale_lat.isChecked() else 1.0
        except ValueError:
            return 1.0

    def _get_vert_scale(self):
        try:
            return float(self.inp_vert_scale.text()) if self.chk_scale_vert.isChecked() else 1.0
        except ValueError:
            return 1.0

    def update_frame(self):
        """Timer callback — advance playback based on real elapsed time."""
        if self.df is None:
            return

        now = time.time()
        dt_real = now - self.last_real_time
        self.last_real_time = now

        # Advance animation clock (for flame flickering even at slow playback)
        self._anim_clock += dt_real

        sim_dt = dt_real * self.playback_speed
        frames_to_advance = int(sim_dt / self.time_step)

        if frames_to_advance < 1 and sim_dt > 0:
            self.sim_time_accumulator += sim_dt
            if self.sim_time_accumulator >= self.time_step:
                frames_to_advance = int(self.sim_time_accumulator / self.time_step)
                self.sim_time_accumulator -= frames_to_advance * self.time_step
        else:
            self.sim_time_accumulator = 0

        new_frame = self.current_frame + frames_to_advance

        if new_frame < len(self.df) - 1:
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                self._scene_dirty = True
                self.slider.blockSignals(True)
                self.slider.setValue(self.current_frame)
                self.slider.blockSignals(False)
                self.update_scene()
            else:
                # Even if frame hasn't changed, update flame animation
                self.update_scene()
        else:
            self.current_frame = len(self.df) - 1
            self.slider.setValue(self.current_frame)
            self._scene_dirty = True
            self.update_scene()
            self.is_playing = False
            self.btn_play.setText("Play")
            self.timer.stop()

    # ------------------------------------------------------------------
    # Main scene update — the render loop
    # ------------------------------------------------------------------
    def update_scene(self):
        """Update all scene elements. Batches geometry, then renders once."""
        if self.df is None:
            return

        row = self.df.iloc[self.current_frame]

        # --- Compute position with scaling ---
        lat_scale = self._get_lat_scale()
        vert_scale = self._get_vert_scale()
        pos = np.array([
            row["X"] * lat_scale,
            row["Y"] * lat_scale,
            row["Z"] * vert_scale,
        ])

        # --- Orientation ---
        rot_matrix = np.eye(3)
        if "QW" in row and "QX" in row:
            q = [row["QX"], row["QY"], row["QZ"], row["QW"]]
            try:
                r = R.from_quat(q)
                rot_matrix = r.as_matrix()
            except Exception:
                pass

        # --- Rocket transform ---
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = pos

        if self.rocket_actor:
            self.rocket_actor.user_matrix = transform

        # --- Motor burn detection ---
        is_burning = False
        if self.current_frame > 0 and "Mass" in self.df.columns:
            m_prev = self.df.iloc[self.current_frame - 1]["Mass"]
            m_curr = row["Mass"]
            if m_curr < m_prev - 1e-9:
                is_burning = True

        # --- Animated flame ---
        if is_burning:
            self._create_animated_flame(transform, self._anim_clock)
        else:
            self._hide_flame()

        # --- Particle plume ---
        plume_dir = rot_matrix @ np.array([0, 0, -1])  # nozzle points down
        dt_particle = 0.016  # ~60 fps timestep
        self._update_particles(pos, plume_dir, is_burning, dt_particle)

        # --- Velocity ---
        vel = np.array([
            row.get("VX", 0.0),
            row.get("VY", 0.0),
            row.get("VZ", 0.0),
        ])
        speed = np.linalg.norm(vel)

        # --- Crash detection ---
        is_last_frame = self.current_frame >= len(self.df) - 1
        is_crashed = is_last_frame and speed > 5.0

        # --- Explosion ---
        if is_crashed:
            if self.explosion_mesh and not self.explosion_actor:
                self.plotter.add_mesh(
                    self.explosion_mesh, name="explosion",
                    color="red", opacity=0.9, emissive=True,
                )
                self.explosion_actor = self.plotter.renderer.actors.get("explosion")
            if self.explosion_actor:
                self.explosion_actor.SetVisibility(True)
                exp_transform = np.eye(4)
                exp_transform[:3, :3] = rot_matrix
                exp_transform[:3, 3] = pos
                self.explosion_actor.user_matrix = exp_transform
        else:
            if self.explosion_actor:
                self.explosion_actor.SetVisibility(False)

        # --- Trajectory trail ---
        self._update_trail()

        # --- HUD (2D Qt labels — no 3D text actor recreation) ---
        self.lbl_time.setText(f"T: {row['Time']:.2f}s")
        self._update_hud(row, is_burning, is_crashed)

        # --- Camera ---
        self._update_camera(pos, vel, rot_matrix)

        # --- Single render call ---
        self.plotter.render()
        self._last_rendered_frame = self.current_frame

    # ------------------------------------------------------------------
    # Resize event — reposition HUD overlay
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "hud_widget"):
            self.hud_widget.setGeometry(10, 40, 320, 280)
            self.hud_widget.raise_()


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark fusion style
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

    window = RocketVisualizer()

    # Command-line CSV loading
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path) and csv_path.lower().endswith(".csv"):
            window.load_csv_file(csv_path)
            window.toggle_play()

    window.show()
    sys.exit(app.exec_())

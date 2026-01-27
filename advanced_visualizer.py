import sys
import os
import json
import pandas as pd
import numpy as np
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter, QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QLineEdit, QWidget,
                             QSlider, QComboBox, QGroupBox, QMessageBox, QCheckBox,
                             QAction, QDialog, QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QIcon
from scipy.spatial.transform import Rotation as R
import ctypes

class RocketVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexaVisual - Vortex Desktop")
        self.resize(1280, 800)
        
        # Set Icon
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'hexavisual_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            # Fix taskbar icon on Windows
            myappid = 'vortex.hexavisual.v0.1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        # Data
        self.df = None
        self.rocket_mesh = None
        self.custom_flame_mesh = None
        self.rocket_actor = None
        self.flame_actor = None
        self.ground_actor = None
        self.explosion_mesh = None
        self.explosion_actor = None
        
        # State
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.time_step = 0.01 # estimated from CSV
        self.last_real_time = 0
        self.sim_time_accumulator = 0
        
        # Load Preferences
        self.load_preferences()

        # UI Setup
        self.init_ui()
        
        # Default objects
        self.create_ground()
        self.create_default_rocket()
        self.load_explosion_asset()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Menu Bar
        self.create_menu()
        
        # Main Layout: 3D View on top, Controls on bottom
        layout = QVBoxLayout(central_widget)
        
        # 3D Plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.add_axes()
        layout.addWidget(self.plotter.interactor)
        
        # Controls Area
        controls_layout = QVBoxLayout()
        layout.addLayout(controls_layout)
        
        # Files & Environment
        files_group = QGroupBox("Loading")
        files_layout = QHBoxLayout()
        
        self.btn_load_csv = QPushButton("Load CSV Run")
        self.btn_load_csv.clicked.connect(self.load_csv)
        files_layout.addWidget(self.btn_load_csv)
        
        self.btn_load_stl = QPushButton("Load Rocket STL")
        self.btn_load_stl.clicked.connect(self.load_stl)
        files_layout.addWidget(self.btn_load_stl)
        
        files_layout.addStretch()
        self.chk_scale_lat = QCheckBox("Scale Lateral (X/Y)")
        self.chk_scale_lat.setChecked(self.scale_lat_enabled)
        self.chk_scale_lat.setStyleSheet("color: black;")
        files_layout.addWidget(self.chk_scale_lat)
        
        self.inp_lat_scale = QLineEdit(self.lat_scale_val)
        self.inp_lat_scale.setFixedWidth(40)
        files_layout.addWidget(self.inp_lat_scale)
        
        files_group.setLayout(files_layout)
        controls_layout.addWidget(files_group)
        
        # Timeline Controls
        timeline_group = QGroupBox("Playback")
        timeline_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        timeline_layout.addWidget(self.btn_play)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_change)
        timeline_layout.addWidget(self.slider)
        
        self.lbl_time = QLabel("T: 0.00s")
        self.lbl_time.setFixedWidth(80)
        timeline_layout.addWidget(self.lbl_time)
        
        self.combo_speed = QComboBox()
        self.combo_speed.setEditable(True)
        self.combo_speed.addItems(["0.1x", "0.5x", "1.0x", "2.0x", "5.0x", "10.0x"])
        self.combo_speed.setCurrentText(self.default_playback_speed)
        self.update_speed(self.default_playback_speed) # Set initial playback speed
        self.combo_speed.currentTextChanged.connect(self.update_speed)
        timeline_layout.addWidget(self.combo_speed)
        
        # Jumps / Skips
        timeline_layout.addWidget(QLabel(" Skip (s):"))
        self.inp_skip = QLineEdit("5.0")
        self.inp_skip.setFixedWidth(40)
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
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_preferences(self):
        pref_path = os.path.join(os.path.dirname(__file__), 'hexavisual_prefs.json')
        # Defaults
        self.results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
        self.default_rocket_stl = ""
        self.flame_stl_path = ""
        self.explosion_stl_path = os.path.join("assets", "Dramatic_explosion_effect_part.stl")
        self.lat_scale_val = "10.0"
        self.scale_lat_enabled = False
        self.default_playback_speed = "1.0x"

        if os.path.exists(pref_path):
            try:
                with open(pref_path, 'r') as f:
                    prefs = json.load(f)
                    self.results_dir = prefs.get('results_dir', self.results_dir)
                    self.default_rocket_stl = prefs.get('default_rocket_stl', self.default_rocket_stl)
                    self.flame_stl_path = prefs.get('flame_stl_path', self.flame_stl_path)
                    self.explosion_stl_path = prefs.get('explosion_stl_path', self.explosion_stl_path)
                    self.lat_scale_val = prefs.get('lat_scale', self.lat_scale_val)
                    self.scale_lat_enabled = prefs.get('scale_lat_enabled', self.scale_lat_enabled)
                    self.default_playback_speed = prefs.get('playback_speed', self.default_playback_speed)
            except Exception as e:
                print(f"Error loading preferences: {e}")

    def create_menu(self):
        # PyQt5 Menu Bar
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('File')
        
        import_action = QAction('Import from HexaKinetic', self)
        import_action.triggered.connect(self.open_import_dialog)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.open_about)
        file_menu.addAction(about_action)
        
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Edit Menu
        edit_menu = menubar.addMenu('Edit')
        
        pref_action = QAction('Preferences', self)
        pref_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(pref_action)
        
        edit_menu.addSeparator()
        
        launch_gui_action = QAction('Launch HexaKinetic Simulation', self)
        launch_gui_action.triggered.connect(self.launch_main_gui)
        edit_menu.addAction(launch_gui_action)
        
        # View Menu
        view_menu = menubar.addMenu('View')
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.triggered.connect(lambda: self.plotter.camera.zoom(1.2))
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.triggered.connect(lambda: self.plotter.camera.zoom(0.8))
        view_menu.addAction(zoom_out_action)

    def open_about(self):
        QMessageBox.about(self, "About", "HexaVisual v0.1\n\nAdvanced Visualion software for HexaKinetic\n\n©️ 2026 Agastya Mishra")

    def launch_main_gui(self):
        import subprocess
        try:
            subprocess.Popen([sys.executable, "gui.py"])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch GUI: {e}")

    def open_preferences(self):
        self.pref_dialog = QDialog(self)
        self.pref_dialog.setWindowTitle("Preferences")
        layout = QVBoxLayout()
        
        # Results Dir
        layout.addWidget(QLabel("Results Directory:"))
        res_layout = QHBoxLayout()
        self.inp_res_dir = QLineEdit(self.results_dir)
        res_layout.addWidget(self.inp_res_dir)
        btn_browse_res = QPushButton("Browse")
        btn_browse_res.clicked.connect(lambda: self.inp_res_dir.setText(QFileDialog.getExistingDirectory(self, "Select Results Dir")))
        res_layout.addWidget(btn_browse_res)
        layout.addLayout(res_layout)
        
        # Default Rocket STL
        layout.addWidget(QLabel("Default Rocket STL:"))
        rock_layout = QHBoxLayout()
        self.inp_rock_stl = QLineEdit(self.default_rocket_stl)
        rock_layout.addWidget(self.inp_rock_stl)
        btn_browse_rock = QPushButton("Browse")
        btn_browse_rock.clicked.connect(lambda: self.inp_rock_stl.setText(QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0]))
        rock_layout.addWidget(btn_browse_rock)
        layout.addLayout(rock_layout)
        
        # Flame STL
        layout.addWidget(QLabel("Flame STL:"))
        flame_layout = QHBoxLayout()
        self.inp_flame_stl = QLineEdit(self.flame_stl_path)
        flame_layout.addWidget(self.inp_flame_stl)
        btn_browse_flame = QPushButton("Browse")
        btn_browse_flame.clicked.connect(lambda: self.inp_flame_stl.setText(QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0]))
        flame_layout.addWidget(btn_browse_flame)
        layout.addLayout(flame_layout)
        
        # Explosion STL
        layout.addWidget(QLabel("Explosion STL:"))
        exp_layout = QHBoxLayout()
        self.inp_exp_stl = QLineEdit(self.explosion_stl_path)
        exp_layout.addWidget(self.inp_exp_stl)
        btn_browse_exp = QPushButton("Browse")
        btn_browse_exp.clicked.connect(lambda: self.inp_exp_stl.setText(QFileDialog.getOpenFileName(self, "Select STL", "", "STL (*.stl)")[0]))
        exp_layout.addWidget(btn_browse_exp)
        layout.addLayout(exp_layout)
        
        btn_save = QPushButton("Save Preferences")
        btn_save.clicked.connect(self.save_preferences)
        layout.addWidget(btn_save)
        
        self.pref_dialog.setLayout(layout)
        self.pref_dialog.exec_()

    def save_preferences(self):
        self.results_dir = self.inp_res_dir.text()
        self.default_rocket_stl = self.inp_rock_stl.text()
        self.flame_stl_path = self.inp_flame_stl.text()
        self.explosion_stl_path = self.inp_exp_stl.text()
        
        # Save to JSON
        pref_path = os.path.join(os.path.dirname(__file__), 'hexavisual_prefs.json')
        prefs = {
            'results_dir': self.results_dir,
            'default_rocket_stl': self.default_rocket_stl,
            'flame_stl_path': self.flame_stl_path,
            'explosion_stl_path': self.explosion_stl_path,
            'lat_scale': self.inp_lat_scale.text(),
            'scale_lat_enabled': self.chk_scale_lat.isChecked(),
            'playback_speed': self.combo_speed.currentText()
        }
        try:
            with open(pref_path, 'w') as f:
                json.dump(prefs, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save preferences to file: {e}")

        # Auto-reload assets if changed
        if self.default_rocket_stl and os.path.exists(self.default_rocket_stl):
            self.create_default_rocket()
        if self.flame_stl_path and os.path.exists(self.flame_stl_path):
             self.load_custom_flame(self.flame_stl_path)
        if self.explosion_stl_path:
             self.load_explosion_asset_path(self.explosion_stl_path)
             
        self.pref_dialog.accept()

    def open_import_dialog(self):
        # Card style dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Import from HexaKinetic")
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)
        
        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Available Runs"))
        chk_opt = QCheckBox("Show Optimization Runs")
        header.addWidget(chk_opt)
        header.addStretch()
        layout.addLayout(header)
        
        # Scroll Area for Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.cards_layout = QGridLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Populate
        path = self.results_dir
        if os.path.exists(path):
            row, col = 0, 0
            # Look for subdirectories starting with 'single_run_'
            try:
                dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('single_run_')]
                # Sort by folder name descending (latest first)
                dirs.sort(reverse=True)
                
                for d in dirs:
                    folder_path = os.path.join(path, d)
                    csv_path = os.path.join(folder_path, 'single_run.csv')
                    png_path = os.path.join(folder_path, 'trajectory_3d.png')
                    
                    if os.path.exists(csv_path):
                        # Parse timestamp: single_run_YYYYMMDD_HHMMSS
                        pts = d.split('_')
                        if len(pts) >= 4:
                            date_str = pts[2]
                            time_str = pts[3]
                            try:
                                formatted_dt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                            except:
                                formatted_dt = d
                        else:
                            formatted_dt = d
                        
                        # Card Frame
                        card = QFrame()
                        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
                        card.setFixedSize(260, 260)
                        card.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; }")
                        cl = QVBoxLayout(card)
                        
                        lbl_name = QLabel(formatted_dt)
                        lbl_name.setAlignment(Qt.AlignCenter)
                        lbl_name.setStyleSheet("font-weight: bold; color: #333;")
                        cl.addWidget(lbl_name)
                        
                        # Thumbnail
                        thumb = QLabel()
                        thumb.setAlignment(Qt.AlignCenter)
                        thumb.setMinimumHeight(150)
                        if os.path.exists(png_path):
                            pix = QPixmap(png_path)
                            thumb.setPixmap(pix.scaled(240, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        else:
                            thumb.setText("No Preview Available")
                            thumb.setStyleSheet("background-color: #e9ecef; color: #6c757d; border-radius: 4px;")
                        cl.addWidget(thumb, 1)
                        
                        btn_load = QPushButton("Load Simulation")
                        btn_load.setStyleSheet("QPushButton { background-color: #007bff; color: white; border-radius: 4px; padding: 5px; } QPushButton:hover { background-color: #0056b3; }")
                        # Use closure default arg to capture csv_path
                        btn_load.clicked.connect(lambda checked, p=csv_path: [self.load_csv_file(p), dlg.accept()])
                        cl.addWidget(btn_load)
                        
                        self.cards_layout.addWidget(card, row, col)
                        col += 1
                        if col > 2:
                            col = 0
                            row += 1
                
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
    
    def load_csv_file(self, path):
         # Direct loader bypasses dialog
         try:
            self.df = pd.read_csv(path)
            
            # Validation
            required = ['Time', 'X', 'Y', 'Z']
            if not all(col in self.df.columns for col in required):
                raise ValueError("CSV missing required columns: Time, X, Y, Z")
                
            self.slider.setRange(0, len(self.df) - 1)
            self.slider.setValue(0)
            self.current_frame = 0
            self.is_playing = False
            self.btn_play.setText("Play")
            self.timer.stop()
            
            # Estimate time step
            if len(self.df) > 1:
                self.time_step = self.df['Time'].iloc[1] - self.df['Time'].iloc[0]
            
            self.update_scene()
            self.plotter.reset_camera()
            print(f"Loaded {len(self.df)} frames from {path}")
            
         except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def load_custom_flame(self, path):
        try:
             mesh = pv.read(path)
             b = mesh.bounds
             center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
             mesh.translate([-c for c in center], inplace=True)
             self.custom_flame_mesh = mesh
             if self.flame_actor:
                 self.plotter.remove_actor('flame')
                 self.flame_actor = None
             self.update_scene()
        except: pass

    def load_explosion_asset_path(self, path):
        if os.path.exists(path):
            try:
                mesh = pv.read(path)
                b = mesh.bounds
                center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
                mesh.translate([-c for c in center], inplace=True)
                mesh.scale(2.0, inplace=True)
                self.explosion_mesh = mesh
            except: pass

    def is_dark_mode(self):
        # Simple heuristic or preference
        return True

    def create_default_rocket(self):
        # Check if we have a preference set and the file exists
        if hasattr(self, 'default_rocket_stl') and self.default_rocket_stl and os.path.exists(self.default_rocket_stl):
            try:
                mesh = pv.read(self.default_rocket_stl)
                self.apply_rocket_mesh(mesh)
                return
            except:
                pass
        
        # Create a simple cylinder/cone rocket if no STL is loaded
        mesh = pv.Cylinder(radius=0.15, height=5.0, direction=(0,0,1))
        self.apply_rocket_mesh(mesh)

    def apply_rocket_mesh(self, mesh):
        """Standardize loading and aligning rocket meshes (Nozzle at origin)"""
        b = mesh.bounds
        # Center in X and Y, but put the bottom (Z min) at local 0,0,0
        x_center = (b[0] + b[1]) / 2
        y_center = (b[2] + b[3]) / 2
        z_min = b[4]
        mesh.translate([-x_center, -y_center, -z_min], inplace=True)
        
        self.rocket_mesh = mesh
        self.plotter.add_mesh(self.rocket_mesh, name='rocket', color='silver', smooth_shading=True)
        self.rocket_actor = self.plotter.renderer.actors['rocket']
        self.update_ground_level()
        self.update_scene()

    def create_ground(self):
        # Create a large plane for the ground
        self.ground_mesh = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=2000, j_size=2000)
        # Check for satellite texture or use default color
        self.plotter.add_mesh(self.ground_mesh, name='ground', color='lightgrey', opacity=0.5, show_edges=True)
        self.ground_actor = self.plotter.renderer.actors['ground']

    def update_ground_level(self):
        # Ground is always at Z=0 because CSV Z now reports nozzle altitude
        if self.ground_actor:
            self.ground_actor.position = (0, 0, 0)

    def load_explosion_asset(self):
        path = os.path.join("assets", "Dramatic_explosion_effect_part.stl")
        if os.path.exists(path):
            try:
                mesh = pv.read(path)
                # Center it
                b = mesh.bounds
                center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
                mesh.translate([-c for c in center], inplace=True)
                # Scale it up slightly?
                mesh.scale(2.0, inplace=True)
                self.explosion_mesh = mesh
            except:
                pass

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV Simulation", "", "CSV Files (*.csv)")
        if not path:
            return
            
        try:
            self.df = pd.read_csv(path)
            
            # Validation
            required = ['Time', 'X', 'Y', 'Z']
            if not all(col in self.df.columns for col in required):
                raise ValueError("CSV missing required columns: Time, X, Y, Z")
                
            self.slider.setRange(0, len(self.df) - 1)
            self.slider.setValue(0)
            self.current_frame = 0
            self.is_playing = False
            self.btn_play.setText("Play")
            self.timer.stop()
            
            # Estimate time step
            if len(self.df) > 1:
                self.time_step = self.df['Time'].iloc[1] - self.df['Time'].iloc[0]
            
            self.update_scene()
            self.plotter.reset_camera()
            
            # Print file info
            print(f"Loaded {len(self.df)} frames from {path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Rocket STL", "", "STL Files (*.stl)")
        if not path:
            return
            
        try:
            mesh = pv.read(path)
            self.apply_rocket_mesh(mesh)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load STL: {e}")

    def load_flame_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Flame STL", "", "STL Files (*.stl)")
        if not path:
            return
            
        try:
            mesh = pv.read(path)
            # Center the mesh at origin
            b = mesh.bounds
            center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
            mesh.translate([-c for c in center], inplace=True)
            self.custom_flame_mesh = mesh
            
            # Reset flame actor to force recreation in update_scene
            if self.flame_actor:
                self.plotter.remove_actor('flame')
                self.flame_actor = None
            
            self.update_scene()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Flame STL: {e}")

    def update_ground_texture(self):
        # Advanced: Fetch map tile
        # For now, just change color to imply update or load a local placeholder
        # TODO: Implement ArcGIS fetching
        pass

    def toggle_play(self):
        if self.df is None:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            self.last_real_time = time.time()
            # Set timer to a fixed high-ish frequency (e.g., 60 FPS = 16ms)
            self.timer.start(16) 
        else:
            self.btn_play.setText("Play")
            self.timer.stop()

    def update_speed(self, text):
        try:
            speed_str = text.replace('x', '')
            self.playback_speed = float(speed_str)
            if self.is_playing:
                # Restart timer with new interval
                self.timer.stop()
                interval = int(self.time_step * 1000 / self.playback_speed)
                self.timer.start(max(1, interval))
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
            self.slider.setValue(new_frame) # This triggers on_slider_change
        except ValueError:
            pass

    def on_slider_change(self, val):
        self.current_frame = val
        self.update_scene()

    def update_frame(self):
        if self.df is None:
            return
            
        # Calculate how much simulation time should have passed
        now = time.time()
        dt_real = now - self.last_real_time
        self.last_real_time = now
        
        # Sim time to advance: real_time_passed * speed
        sim_dt = dt_real * self.playback_speed
        
        # Find the next frame based on Time column
        current_sim_time = self.df.iloc[self.current_frame]['Time']
        target_sim_time = current_sim_time + sim_dt
        
        # Find index of closest time in DF (simple search or assume uniform)
        # Assuming roughly uniform time steps for performance:
        frames_to_advance = int(sim_dt / self.time_step)
        if frames_to_advance < 1 and sim_dt > 0:
            # Accumulate fractional frames if speed is very low
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
                self.slider.blockSignals(True)
                self.slider.setValue(self.current_frame)
                self.slider.blockSignals(False)
                self.update_scene()
        else:
            self.current_frame = len(self.df) - 1
            self.slider.setValue(self.current_frame)
            self.update_scene()
            self.is_playing = False
            self.btn_play.setText("Play")
            self.timer.stop()

    def update_scene(self):
        if self.df is None:
            return
            
        row = self.df.iloc[self.current_frame]
        
        # Position
        try:
            lat_scale = float(self.inp_lat_scale.text()) if self.chk_scale_lat.isChecked() else 1.0
        except ValueError:
            lat_scale = 1.0
            
        pos = np.array([row['X'] * lat_scale, row['Y'] * lat_scale, row['Z']])
        
        # Orientation
        rot_matrix = np.eye(3)
        if 'QW' in row and 'QX' in row:
            # Full 6DOF
            q = [row['QX'], row['QY'], row['QZ'], row['QW']] # Scipy uses x,y,z,w
            r = R.from_quat(q)
            rot_matrix = r.as_matrix()
        else:
            # Fallback: Align z-axis to velocity if moving
            vx, vy, vz = row['VX'], row['VY'], row['VZ']
            vel = np.array([vx, vy, vz])
            norm = np.linalg.norm(vel)
            if norm > 1e-3:
                # Simple alignment logic or just identity
                pass
        
        # Apply Transform to Rocket
        # PyVista/VTK transformation
        # We need to construct a 4x4 matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = pos
        
        self.rocket_actor.user_matrix = transform
        
        # Motor Plume
        is_burning = False
        if self.current_frame > 0 and 'Mass' in self.df.columns:
            m_prev = self.df.iloc[self.current_frame-1]['Mass']
            m_curr = row['Mass']
            # If mass decreases even slightly, the motor is active.
            if m_curr < m_prev - 1e-9:
                is_burning = True
                
        if is_burning:
            # Re-create or fetch flame actor
            if 'flame' not in self.plotter.renderer.actors:
                # Determine a much larger radius to ensure it's visible outside any body/fins
                b_rock = self.rocket_mesh.bounds
                rocket_width = b_rock[1] - b_rock[0]
                # Flame should be slightly wider than the rocket at its base for a "chunky" plume
                flame_radius = max(0.6, rocket_width * 1.2)
                
                if self.custom_flame_mesh:
                    self.flame_mesh = self.custom_flame_mesh.copy()
                    b_f = self.flame_mesh.bounds
                    h_f = b_f[5] - b_f[4]
                    self.flame_mesh.translate([0, 0, -(h_f/2)], inplace=True)
                else:
                    # Pointed end (Apex) at Nozzle (0), Base at -5.0. 
                    # direction=(0,0,1) with center at -2.5 and height 5.0 puts apex at 0.
                    self.flame_mesh = pv.Cone(center=(0,0, -2.5), 
                                             direction=(0,0,1), 
                                             height=5.0, 
                                             radius=flame_radius)
                
                # Use solid bright orange-red for maximum visibility
                self.plotter.add_mesh(self.flame_mesh, name='flame', color='orangered', 
                                      opacity=1.0, emissive=True, lighting=False)
            
            self.flame_actor = self.plotter.renderer.actors['flame']
            self.flame_actor.SetVisibility(True)
            self.flame_actor.user_matrix = transform
        else:
            if 'flame' in self.plotter.renderer.actors:
                self.plotter.renderer.actors['flame'].SetVisibility(False)

        # HUD / Overlay
        # Update text (this is slow in PyVista if recreating actors, use 2D labels if possible)
        # Using status bar or labels in UI for efficiency
        self.lbl_time.setText(f"T: {row['Time']:.2f}s")
        
        # Update overlay text in 3D view
        speed = np.sqrt(row['VX']**2+row['VY']**2+row['VZ']**2)
        hud_text = (f"Alt: {row['Z']:.1f} m\n"
                    f"Vel: {row['VZ']:.1f} m/s\n"
                    f"Speed: {speed:.1f} m/s")
        
        # Crash logic at the end
        is_last_frame = (self.current_frame >= len(self.df) - 1)
        if is_last_frame and speed > 5.0:
            hud_text += "\nCRASHED!"
            self.plotter.add_text(hud_text, name='hud', position='upper_left', font_size=12, color='red')
            
            # Show explosion
            if self.explosion_mesh and not self.explosion_actor:
                self.plotter.add_mesh(self.explosion_mesh, name='explosion', color='red', opacity=0.9, emissive=True)
                self.explosion_actor = self.plotter.renderer.actors['explosion']
            
            if self.explosion_actor:
                self.explosion_actor.SetVisibility(True)
                # Position explosion at the nozzle (body local 0,0,0)
                z_bottom_local = 0.0
                
                # Global offset = rotation * local_offset
                offset_vec = rot_matrix @ np.array([0, 0, z_bottom_local])
                
                exp_transform = np.eye(4)
                exp_transform[:3, :3] = rot_matrix # keep orientation or just identity? user said "bottom of rocket"
                exp_transform[:3, 3] = pos + offset_vec
                
                self.explosion_actor.user_matrix = exp_transform
        else:
            self.plotter.add_text(hud_text, name='hud', position='upper_left', font_size=12, color='black')
            if self.explosion_actor:
                self.explosion_actor.SetVisibility(False)

        self.plotter.render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RocketVisualizer()
    
    # Check for command line arguments (CSV path)
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path) and csv_path.lower().endswith('.csv'):
            window.load_csv_file(csv_path)
            # Try to start playback automatically
            window.toggle_play()
    
    window.show()
    sys.exit(app.exec_())

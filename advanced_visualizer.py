import sys
import os
import pandas as pd
import numpy as np
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter, QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QLineEdit, QWidget,
                             QSlider, QComboBox, QGroupBox, QMessageBox, QCheckBox)
from PyQt5.QtCore import QTimer, Qt
from scipy.spatial.transform import Rotation as R

class RocketVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project Vortex - Advanced Visualizer")
        self.resize(1280, 800)

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
        
        # UI Setup
        self.init_ui()
        
        # Default objects
        self.create_default_rocket()
        self.create_ground()
        self.load_explosion_asset()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
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
        
        self.btn_load_flame = QPushButton("Load Flame STL")
        self.btn_load_flame.clicked.connect(self.load_flame_stl)
        files_layout.addWidget(self.btn_load_flame)
        
        files_layout.addStretch()
        self.chk_scale_lat = QCheckBox("Scale Lateral (X/Y)")
        self.chk_scale_lat.setStyleSheet("color: black;")
        files_layout.addWidget(self.chk_scale_lat)
        
        self.inp_lat_scale = QLineEdit("10.0")
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
        self.combo_speed.setCurrentIndex(2) # 1.0x
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

    def is_dark_mode(self):
        # Simple heuristic or preference
        return True

    def create_default_rocket(self):
        # Create a simple cylinder/cone rocket if no STL is loaded
        self.rocket_mesh = pv.Cylinder(radius=0.15, height=5.0, direction=(0,0,1))
        self.plotter.add_mesh(self.rocket_mesh, name='rocket', color='silver', smooth_shading=True)
        self.rocket_actor = self.plotter.renderer.actors['rocket']
        self.update_ground_level()

    def create_ground(self):
        # Create a large plane for the ground
        self.ground_mesh = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=2000, j_size=2000)
        # Check for satellite texture or use default color
        self.plotter.add_mesh(self.ground_mesh, name='ground', color='lightgrey', opacity=0.5, show_edges=True)
        self.ground_actor = self.plotter.renderer.actors['ground']

    def update_ground_level(self):
        if self.rocket_mesh and self.ground_actor:
            b = self.rocket_mesh.bounds
            height = b[5] - b[4]
            z_bottom = -(height / 2)
            # Plane origin is 0,0,0. We move the actor.
            self.ground_actor.position = (0, 0, z_bottom)

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
            # Center the mesh at origin for proper rotations using bounds
            b = mesh.bounds
            center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
            mesh.translate([-c for c in center], inplace=True)
            self.rocket_mesh = mesh
            
            self.plotter.remove_actor('rocket')
            self.plotter.add_mesh(self.rocket_mesh, name='rocket', color='white', smooth_shading=True)
            self.rocket_actor = self.plotter.renderer.actors['rocket']
            
            self.update_ground_level()
            self.update_scene()
            
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
        # Logic: If mass is decreasing, motor is ON
        is_burning = False
        if self.current_frame > 0:
            prev_mass = self.df.iloc[self.current_frame-1]['Mass']
            curr_mass = row['Mass']
            if curr_mass < prev_mass - 1e-6: # burning
                is_burning = True
                
        if is_burning:
            # Draw flame transformsed relative to rocket
            if not self.flame_actor:
                # Calculate offset based on rocket's height
                b_rocket = self.rocket_mesh.bounds
                height_rocket = b_rocket[5] - b_rocket[4]
                z_bottom = -(height_rocket / 2)
                
                if self.custom_flame_mesh:
                    # Use user mesh
                    self.flame_mesh = self.custom_flame_mesh.copy()
                    # Position its top at rocket bottom
                    b_flame = self.flame_mesh.bounds
                    height_flame = b_flame[5] - b_flame[4]
                    # Flame top is at +height_flame/2 (since centered). We want top at z_bottom.
                    # shift = z_bottom - (height_flame/2) 
                    # Wait, if centered, top is z_max = h/2. We want z_max to be at z_bottom.
                    # So shift = z_bottom - (h/2).
                    self.flame_mesh.translate([0, 0, z_bottom - (height_flame/2)], inplace=True)
                else:
                    # Create default cone
                    # Cone is created centered? No, pv.Cone center defaults to (0,0,0).
                    # height 3.0. center=(0,0, z_bottom - 1.5) puts top at z_bottom.
                    self.flame_mesh = pv.Cone(center=(0,0, z_bottom - 1.5), 
                                             direction=(0,0,-1), 
                                             height=3.0, 
                                             radius=0.5)
                
                self.plotter.add_mesh(self.flame_mesh, name='flame', color='orange', opacity=1.0, emissive=True)
                self.flame_actor = self.plotter.renderer.actors['flame']
            
            self.flame_actor.SetVisibility(True)
            self.flame_actor.user_matrix = transform
        else:
            if self.flame_actor:
                self.flame_actor.SetVisibility(False)
        
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
                # Position explosion at the bottom of the rocket
                b_rocket = self.rocket_mesh.bounds
                height_rocket = b_rocket[5] - b_rocket[4]
                z_bottom_local = -(height_rocket / 2)
                
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
    window.show()
    sys.exit(app.exec_())

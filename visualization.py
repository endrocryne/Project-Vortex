import sys
import pandas as pd
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QLineEdit, QWidget)
from PyQt5.QtCore import QTimer

class RocketVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rocket Retro-Landing Visualizer")
        self.resize(1200, 800)

        # Data storage
        self.rocket_mesh = None
        self.trajectory_data = None
        self.current_frame = 0
        self.is_playing = False
        
        # UI Setup
        self.init_ui()
        
        # 3D Plotter Setup
        self.plotter = BackgroundPlotter(show=False)
        self.ui_layout.addWidget(self.plotter.interactor)
        
        # Animation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.ui_layout = QVBoxLayout(central_widget)
        
        # Controls Header
        controls = QHBoxLayout()
        
        self.btn_stl = QPushButton("Upload STL")
        self.btn_stl.clicked.connect(self.load_stl)
        
        self.btn_csv = QPushButton("Upload CSV")
        self.btn_csv.clicked.connect(self.load_csv)
        
        self.label_z = QLabel("Ignition Z-Alt:")
        self.input_z = QLineEdit("500") # Default value
        
        self.btn_play = QPushButton("Play Animation")
        self.btn_play.clicked.connect(self.toggle_animation)
        
        controls.addWidget(self.btn_stl)
        controls.addWidget(self.btn_csv)
        controls.addWidget(self.label_z)
        controls.addWidget(self.input_z)
        controls.addWidget(self.btn_play)
        self.ui_layout.addLayout(controls)

    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl)")
        if path:
            # Load and center the mesh
            self.rocket_mesh = pv.read(path)
            self.rocket_mesh.center = (0, 0, 0) 
            self.plotter.clear()
            self.plotter.add_mesh(self.rocket_mesh, color="silver", name="rocket")
            self.plotter.add_axes()
            self.plotter.reset_camera()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV Data", "", "CSV Files (*.csv)")
        if path:
            self.trajectory_data = pd.read_csv(path)
            print("CSV Loaded Successfully")

    def toggle_animation(self):
        if self.trajectory_data is None or self.rocket_mesh is None:
            print("Error: Load STL and CSV first.")
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            self.timer.start(33) # ~30 FPS
        else:
            self.btn_play.setText("Play Animation")
            self.timer.stop()

    def update_animation(self):
        if self.current_frame >= len(self.trajectory_data):
            self.timer.stop()
            self.current_frame = 0
            self.btn_play.setText("Restart")
            return

        row = self.trajectory_data.iloc[self.current_frame]
        pos = (row['X'], row['Y'], row['Z'])
        vel = (row['VX'], row['VY'], row['VZ'])
        
        # 1. Update Rocket Position and Orientation
        # We transform a copy of the original mesh to avoid cumulative floating point errors
        transformed_rocket = self.rocket_mesh.copy()
        
        # Rotate rocket to face velocity vector
        direction = np.array(vel)
        if np.linalg.norm(direction) > 0:
            transformed_rocket = transformed_rocket.rotate_vector(
                vector=direction, 
                angle=0, # This helper is simplified; for precision, use transformation matrices
                point=(0,0,0)
            )
        
        transformed_rocket.translate(pos, inplace=True)
        self.plotter.add_mesh(transformed_rocket, color="silver", name="rocket", reset_camera=False)

        # 2. Handle Retroboosters (Flame)
        try:
            ignition_alt = float(self.input_z.text())
        except:
            ignition_alt = 0

        if row['Z'] <= ignition_alt:
            # Create a simple cone for the flame at the base
            flame = pv.Cone(center=(row['X'], row['Y'], row['Z'] - 2), 
                            direction=(0, 0, 1), 
                            height=4, radius=1)
            self.plotter.add_mesh(flame, color="orange", name="flame")
        else:
            self.plotter.remove_actor("flame", render=False)

        self.current_frame += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RocketVisualizer()
    window.show()
    sys.exit(app.exec_())
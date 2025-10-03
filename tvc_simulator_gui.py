import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar,
    QTextEdit, QLabel, QComboBox, QHBoxLayout, QGroupBox, QScrollArea
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtGui import QQuaternion

from simulation_core import run_simulation, DEFAULT_PARAMS, THRUST_POINTS_BASE, rotate_by_quaternion

class MonteCarloWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)

    def __init__(self, n_runs, sim_params, pid_gains):
        super().__init__()
        self.n_runs = n_runs
        self.sim_params = sim_params
        self.pid_gains = pid_gains

    def run(self):
        all_results = []
        for i in range(self.n_runs):
            thrust_multiplier = np.random.normal(1.0, self.sim_params['thrust_std'])
            randomized_thrust_curve = THRUST_POINTS_BASE * thrust_multiplier
            params = {
                'thrust_curve': randomized_thrust_curve,
                'propellant_mass': np.random.normal(self.sim_params['propellant_mass_mean'], self.sim_params['propellant_mass_std']),
                'drag_coefficient': np.random.normal(self.sim_params['drag_coefficient_mean'], self.sim_params['drag_coefficient_std']),
                'dry_mass': np.random.normal(self.sim_params['dry_mass_mean'], self.sim_params['dry_mass_std']),
                'air_density': np.random.normal(self.sim_params['air_density_mean'], self.sim_params['air_density_std']),
                'C_N_alpha': np.random.normal(self.sim_params['c_n_alpha_mean'], self.sim_params['c_n_alpha_std']),
                'cp_z_offset': np.random.normal(self.sim_params['cp_z_offset_mean'], abs(self.sim_params['cp_z_offset_mean']) * self.sim_params['cp_z_offset_std']),
                'yaw_init': np.random.normal(self.sim_params['yaw_init_mean'], abs(self.sim_params['yaw_init_mean']) * self.sim_params['yaw_init_std']),
                'pitch_init': np.random.normal(self.sim_params['pitch_init_mean'], abs(self.sim_params['pitch_init_mean']) * self.sim_params['pitch_init_std']),
                'gyro_noise_std': self.sim_params['gyro_noise_std'],
                'actuator_tau': self.sim_params['actuator_tau'],
            }
            result = run_simulation(params, **self.pid_gains)
            all_results.append(result)
            self.progress.emit(int(((i + 1) / self.n_runs) * 100))
        if all_results:
            self.finished.emit(pd.DataFrame(all_results))

class SingleFlightWorker(QObject):
    state_update = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, sim_params, pid_gains):
        super().__init__()
        self.sim_params = sim_params
        self.pid_gains = pid_gains

    def run(self):
        run_simulation(self.sim_params, **self.pid_gains, full_logs=True, progress_callback=self.emit_state)
        self.finished.emit()

    def emit_state(self, time, state):
        self.state_update.emit({'time': time, 'position': state[0:3], 'velocity': state[3:6], 'quaternion': state[6:10]})

class TunerWorker(QObject):
    progress = pyqtSignal(str)
    best_gains = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, tuning_params, sim_params):
        super().__init__()
        self.tuning_params = tuning_params
        self.sim_params = sim_params

    def run(self):
        best_score = float('inf')
        best_gains_found = {}
        objective = self.tuning_params['objective']
        
        # Use mean parameters for tuning runs
        params = {key.replace('_mean',''): val for key, val in self.sim_params.items() if '_mean' in key}
        params['thrust_curve'] = THRUST_POINTS_BASE * self.sim_params['thrust_mean']
        params.update({k: v for k, v in self.sim_params.items() if '_std' not in k and '_mean' not in k})


        for i in range(self.tuning_params['iterations']):
            kp = np.random.uniform(self.tuning_params['kp_min'], self.tuning_params['kp_max'])
            ki = np.random.uniform(self.tuning_params['ki_min'], self.tuning_params['ki_max'])
            kd = np.random.uniform(self.tuning_params['kd_min'], self.tuning_params['kd_max'])
            
            result = run_simulation(params.copy(), Kp=kp, Ki=ki, Kd=kd)
            
            score = result.get(objective, float('inf'))
            
            log_msg = f"Iteration {i+1}/{self.tuning_params['iterations']}: Score={score:.4f}, Gains: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}"
            self.progress.emit(log_msg)
            
            if score < best_score:
                best_score = score
                best_gains_found = {'Kp': kp, 'Ki': ki, 'Kd': kd}
                self.best_gains.emit(best_gains_found)
        
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TVC Rocket Simulator GUI")
        self.setGeometry(100, 100, 1400, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.mc_results_data = None
        self.mean_sim_params = None
        self.create_monte_carlo_tab()
        self.create_visualization_tab()
        self.create_pid_tuner_tab()

    def create_monte_carlo_tab(self):
        # ... (Identical to previous version)
        self.monte_carlo_tab = QWidget()
        self.tabs.addTab(self.monte_carlo_tab, "Monte Carlo Simulation")
        layout = QHBoxLayout(self.monte_carlo_tab)
        config_group = QGroupBox("Configuration")
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        form_layout = QFormLayout()
        self.mc_runs_input = QSpinBox()
        self.mc_runs_input.setRange(1, 10000)
        self.mc_runs_input.setValue(200)
        form_layout.addRow("Number of Runs:", self.mc_runs_input)
        self.param_inputs = {}
        for key, value in DEFAULT_PARAMS.items():
            label = key.replace('_', ' ').title()
            self.param_inputs[key] = QDoubleSpinBox()
            self.param_inputs[key].setDecimals(4)
            self.param_inputs[key].setRange(-1000, 1000)
            self.param_inputs[key].setValue(value)
            form_layout.addRow(label, self.param_inputs[key])
        self.mc_pid_gains = {'Kp': 0.05, 'Ki': 0.1, 'Kd': 0.01}
        self.mc_kp_input = QDoubleSpinBox(); self.mc_kp_input.setValue(self.mc_pid_gains['Kp'])
        self.mc_ki_input = QDoubleSpinBox(); self.mc_ki_input.setValue(self.mc_pid_gains['Ki'])
        self.mc_kd_input = QDoubleSpinBox(); self.mc_kd_input.setValue(self.mc_pid_gains['Kd'])
        form_layout.addRow(QLabel("--- PID Gains ---"))
        form_layout.addRow("Kp:", self.mc_kp_input)
        form_layout.addRow("Ki:", self.mc_ki_input)
        form_layout.addRow("Kd:", self.mc_kd_input)
        config_layout.addLayout(form_layout)
        config_scroll.setWidget(config_widget)
        control_layout = QVBoxLayout()
        self.run_mc_button = QPushButton("Run Monte Carlo Simulation")
        self.run_mc_button.clicked.connect(self.run_monte_carlo)
        self.mc_progress_bar = QProgressBar()
        control_layout.addWidget(config_scroll)
        control_layout.addWidget(self.run_mc_button)
        control_layout.addWidget(self.mc_progress_bar)
        config_group.setLayout(control_layout)
        layout.addWidget(config_group, 1)
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        plot_buttons_layout = QHBoxLayout()
        self.show_apogee_button = QPushButton("Show Apogee Distribution")
        self.show_apogee_button.clicked.connect(self.plot_apogee_distribution)
        self.show_dispersion_button = QPushButton("Show Landing Dispersion")
        self.show_dispersion_button.clicked.connect(self.plot_landing_dispersion)
        self.show_apogee_button.setEnabled(False)
        self.show_dispersion_button.setEnabled(False)
        plot_buttons_layout.addWidget(self.show_apogee_button)
        plot_buttons_layout.addWidget(self.show_dispersion_button)
        self.mc_plot_widget = pg.PlotWidget()
        self.mc_results_text = QTextEdit()
        self.mc_results_text.setReadOnly(True)
        results_layout.addLayout(plot_buttons_layout)
        results_layout.addWidget(self.mc_plot_widget)
        results_layout.addWidget(self.mc_results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, 2)

    def run_monte_carlo(self):
        # ... (Identical to previous version)
        self.run_mc_button.setEnabled(False)
        self.mc_progress_bar.setValue(0)
        self.mc_results_text.clear()
        self.show_apogee_button.setEnabled(False)
        self.show_dispersion_button.setEnabled(False)
        self.mc_plot_widget.clear()
        sim_params = {key: widget.value() for key, widget in self.param_inputs.items()}
        # Ensure any standard-deviation-like parameters are non-negative
        for k in list(sim_params.keys()):
            if k.endswith('_std'):
                try:
                    sim_params[k] = max(0.0, float(sim_params[k]))
                except Exception:
                    sim_params[k] = 0.0
        pid_gains = {'Kp': self.mc_kp_input.value(), 'Ki': self.mc_ki_input.value(), 'Kd': self.mc_kd_input.value()}
        self.thread = QThread()
        self.worker = MonteCarloWorker(self.mc_runs_input.value(), sim_params, pid_gains)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.mc_progress_bar.setValue)
        self.worker.finished.connect(self.on_mc_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_mc_finished(self, results_df):
        # ... (Identical to previous version)
        self.mc_results_data = results_df
        self.run_mc_button.setEnabled(True)
        self.mc_progress_bar.setValue(100)
        self.show_apogee_button.setEnabled(True)
        self.show_dispersion_button.setEnabled(True)
        self.mean_sim_params = {key.replace('_mean',''): val for key, val in self.param_inputs.items() if '_mean' in key}
        self.mean_sim_params['thrust_curve'] = THRUST_POINTS_BASE * self.param_inputs['thrust_mean'].value()
        self.mean_sim_params['gyro_noise_std'] = self.param_inputs['gyro_noise_std'].value()
        self.mean_sim_params['actuator_tau'] = self.param_inputs['actuator_tau'].value()
        summary = "--- Monte Carlo Simulation Results ---\n"
        success_rate = (results_df['stable'].sum() / len(results_df)) * 100
        summary += f"Success Rate: {success_rate:.2f}% ({results_df['stable'].sum()}/{len(results_df)} stable flights)\n\n"
        summary += "--- Apogee (m) ---\n"
        summary += results_df['apogee'].describe().to_string() + "\n\n"
        summary += "--- Landing Distance (m) ---\n"
        summary += results_df['landing_distance'].describe().to_string()
        self.mc_results_text.setText(summary)
        self.plot_apogee_distribution()
        self.launch_sim_button.setEnabled(True)

    def plot_apogee_distribution(self):
        # ... (Identical to previous version)
        if self.mc_results_data is not None:
            self.mc_plot_widget.clear()
            apogees = self.mc_results_data['apogee'].dropna()
            y, x = np.histogram(apogees, bins=30)
            self.mc_plot_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
            self.mc_plot_widget.setLabel('left', 'Frequency')
            self.mc_plot_widget.setLabel('bottom', 'Apogee (m)')
            self.mc_plot_widget.setTitle('Apogee Distribution')

    def plot_landing_dispersion(self):
        # ... (Identical to previous version)
        if self.mc_results_data is not None:
            self.mc_plot_widget.clear()
            distances = self.mc_results_data['landing_distance'].dropna()
            y, x = np.histogram(distances, bins=30)
            self.mc_plot_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0,255,0,150))
            self.mc_plot_widget.setLabel('left', 'Frequency')
            self.mc_plot_widget.setLabel('bottom', 'Landing Distance (m)')
            self.mc_plot_widget.setTitle('Landing Distance Distribution')

    def create_visualization_tab(self):
        # ... (Identical to previous version)
        self.vis_tab = QWidget()
        self.tabs.addTab(self.vis_tab, "Real-Time 3D Visualization")
        layout = QHBoxLayout(self.vis_tab)

        # Create a PyVista plotter and embed it in a QtInteractor widget
        # PyVista expects window_size as a (width, height) tuple, not a Qt QSize
        sz = self.size()
        # Ensure we don't pass a zero-sized window (which can happen before the
        # window is shown). Use a small minimum size fallback.
        width = max(200, int(sz.width()))
        height = max(200, int(sz.height()))
        window_size = (width, height)

        # Create the QtInteractor first and use its embedded plotter. Passing an
        # externally created Plotter into QtInteractor can trigger MRO/init
        # issues with some pyvista/pyvistaqt/PyQt6 combinations.
        self.plotter_widget = QtInteractor(self.vis_tab)
        self.plotter = getattr(self.plotter_widget, 'plotter', None)
        if self.plotter is not None:
            try:
                self.plotter.window_size = window_size
            except Exception:
                # If setting window_size fails, continue — it's non-fatal for startup
                pass
        layout.addWidget(self.plotter_widget, 4)

        self.setup_3d_scene()
        sidebar_group = QGroupBox("Controls & Telemetry")
        sidebar_layout = QVBoxLayout()
        self.launch_sim_button = QPushButton("Launch Single Simulation")
        self.launch_sim_button.setEnabled(False)
        self.launch_sim_button.clicked.connect(self.run_single_flight)
        sidebar_layout.addWidget(self.launch_sim_button)
        telemetry_layout = QFormLayout()
        self.altitude_label = QLabel("N/A")
        self.velocity_label = QLabel("N/A")
        self.downrange_label = QLabel("N/A")
        self.tilt_angle_label = QLabel("N/A")
        telemetry_layout.addRow("Altitude (m):", self.altitude_label)
        telemetry_layout.addRow("Velocity (m/s):", self.velocity_label)
        telemetry_layout.addRow("Downrange (m):", self.downrange_label)
        telemetry_layout.addRow("Tilt Angle (deg):", self.tilt_angle_label)
        sidebar_layout.addLayout(telemetry_layout)
        sidebar_layout.addStretch()
        sidebar_group.setLayout(sidebar_layout)
        layout.addWidget(sidebar_group, 1)

    def setup_3d_scene(self):
        # ... (Identical to previous version)
        p = self.get_plotter()
        # Wrap plotting calls so the UI can start even if the interactor lacks
        # some pyvista convenience methods in this environment.
        if hasattr(p, 'add_axes'):
            try:
                p.add_axes()
            except Exception:
                pass
        if hasattr(p, 'add_grid'):
            try:
                p.add_grid()
            except Exception:
                pass

        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=2000, j_size=2000)
        # Add ground plane (best-effort)
        try:
            if hasattr(p, 'add_mesh'):
                p.add_mesh(ground, color='green')
        except Exception:
            pass

        body = pv.Cylinder(center=(0, 0, 0.25), direction=(0, 0, 1), radius=0.05, height=0.5)
        nose = pv.Cone(center=(0, 0, 0.5), direction=(0, 0, 1), radius=0.05, height=0.2)
        # Try to add the rocket mesh; if not possible, create a dummy actor so
        # update_3d_scene can still run without attribute errors.
        try:
            if hasattr(p, 'add_mesh'):
                self.rocket_actor = p.add_mesh(body + nose, color="silver")
            else:
                raise AttributeError
        except Exception:
            class DummyActor:
                def __init__(self):
                    self.position = np.zeros(3)
                    self.user_matrix = np.identity(4)
            self.rocket_actor = DummyActor()

        self.trajectory_points = []
        self.trajectory_spline = None
        # Camera settings are best-effort; ignore if unavailable
        try:
            if hasattr(p, 'camera'):
                p.camera_position = 'xy'
                p.camera.azimuth = 45
                p.camera.elevation = 30
                p.camera.zoom(1.5)
        except Exception:
            pass

    def run_single_flight(self):
        # ... (Identical to previous version)
        if not self.mean_sim_params:
            self.mc_results_text.append("\nRun a Monte Carlo simulation first to establish mean parameters.")
            return
        self.launch_sim_button.setEnabled(False)
        p = self.get_plotter()
        try:
            p.clear_actors()
        except Exception:
            pass
        self.setup_3d_scene()
        pid_gains = {'Kp': self.mc_kp_input.value(), 'Ki': self.mc_ki_input.value(), 'Kd': self.mc_kd_input.value()}
        self.vis_thread = QThread()
        self.vis_worker = SingleFlightWorker(self.mean_sim_params, pid_gains)
        self.vis_worker.moveToThread(self.vis_thread)
        self.vis_worker.state_update.connect(self.update_3d_scene)
        self.vis_thread.started.connect(self.vis_worker.run)
        self.vis_worker.finished.connect(self.on_vis_finished)
        self.vis_worker.finished.connect(self.vis_thread.quit)
        self.vis_worker.finished.connect(self.vis_worker.deleteLater)
        self.vis_thread.finished.connect(self.vis_thread.deleteLater)
        self.vis_thread.start()

    @pyqtSlot(object)
    def update_3d_scene(self, state):
        # ... (Identical to previous version)
        pos = state['position']
        quat = state['quaternion']
        self.rocket_actor.position = pos
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = QQuaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()
        self.rocket_actor.user_matrix = transform_matrix
        self.trajectory_points.append(pos)
        if len(self.trajectory_points) > 2:
            p = self.get_plotter()
            try:
                if self.trajectory_spline:
                    p.remove_actor(self.trajectory_spline)
            except Exception:
                pass
            try:
                self.trajectory_spline = p.add_lines(np.array(self.trajectory_points), color='cyan', width=3)
            except Exception:
                pass
        self.altitude_label.setText(f"{pos[2]:.2f}")
        self.velocity_label.setText(f"{np.linalg.norm(state['velocity']):.2f}")
        self.downrange_label.setText(f"{np.linalg.norm(pos[:2]):.2f}")
        z_axis_body = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        tilt_angle = np.rad2deg(np.arccos(np.clip(np.dot(z_axis_body, np.array([0, 0, 1])), -1.0, 1.0)))
        self.tilt_angle_label.setText(f"{tilt_angle:.2f}")

    def on_vis_finished(self):
        # ... (Identical to previous version)
        self.launch_sim_button.setEnabled(True)
        self.trajectory_points = []

    def resizeEvent(self, event):
        """Keep the PyVista plotter window_size in sync with the main window.

        This avoids passing Qt types to PyVista and keeps the embedded view sized
        correctly when the user resizes the application.
        """
        super().resizeEvent(event)
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                sz = self.size()
                self.plotter.window_size = (int(sz.width()), int(sz.height()))
        except Exception:
            # Don't crash the UI for non-fatal sizing issues
            pass

    def get_plotter(self):
        """Return the active pyvista Plotter instance.

        QtInteractor often exposes a `plotter` attribute; if not, fall back to a
        stored reference. Raises AttributeError if none available.
        """
        # Prefer an explicitly stored plotter
        if hasattr(self, 'plotter') and self.plotter is not None:
            return self.plotter

        # Some QtInteractor implementations expose plotting methods on the
        # widget itself (e.g., add_mesh, add_axes). If so, return the widget.
        if hasattr(self, 'plotter_widget'):
            w = self.plotter_widget
            if any(hasattr(w, m) for m in ('add_mesh', 'add_axes', 'add_lines')):
                return w
            if hasattr(w, 'plotter') and w.plotter is not None:
                self.plotter = w.plotter
                return self.plotter

        # As a last resort create a standalone Plotter and keep it.
        try:
            self.plotter = pv.Plotter(off_screen=False)
            return self.plotter
        except Exception:
            raise AttributeError('No pyvista Plotter available')

    def create_pid_tuner_tab(self):
        self.pid_tab = QWidget()
        self.tabs.addTab(self.pid_tab, "PID Gain Auto-Tuner")
        layout = QVBoxLayout(self.pid_tab)
        config_group = QGroupBox("Tuning Configuration")
        config_layout = QFormLayout()
        self.kp_min_input = QDoubleSpinBox(); self.kp_max_input = QDoubleSpinBox()
        self.ki_min_input = QDoubleSpinBox(); self.ki_max_input = QDoubleSpinBox()
        self.kd_min_input = QDoubleSpinBox(); self.kd_max_input = QDoubleSpinBox()
        self.tuning_iterations_input = QSpinBox(); self.tuning_iterations_input.setRange(1, 10000); self.tuning_iterations_input.setValue(100)
        self.tuning_objective_input = QComboBox()
        self.tuning_objective_input.addItems(["max_tilt_angle", "landing_distance"])
        config_layout.addRow("Kp Range:", self.create_range_widget(self.kp_min_input, self.kp_max_input, (0.0, 1.0)))
        config_layout.addRow("Ki Range:", self.create_range_widget(self.ki_min_input, self.ki_max_input, (0.0, 1.0)))
        config_layout.addRow("Kd Range:", self.create_range_widget(self.kd_min_input, self.kd_max_input, (0.0, 0.5)))
        config_layout.addRow("Number of Iterations:", self.tuning_iterations_input)
        config_layout.addRow("Optimization Objective:", self.tuning_objective_input)
        self.start_tuning_button = QPushButton("Start Tuning")
        self.start_tuning_button.clicked.connect(self.run_pid_tuning)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        layout.addWidget(self.start_tuning_button)
        results_group = QGroupBox("Results & Log")
        results_layout = QVBoxLayout()
        best_gains_layout = QFormLayout()
        self.best_kp_label = QLabel("N/A"); self.best_ki_label = QLabel("N/A"); self.best_kd_label = QLabel("N/A")
        best_gains_layout.addRow("Best Kp:", self.best_kp_label)
        best_gains_layout.addRow("Best Ki:", self.best_ki_label)
        best_gains_layout.addRow("Best Kd:", self.best_kd_label)
        self.tuning_log_text = QTextEdit(); self.tuning_log_text.setReadOnly(True)
        results_layout.addLayout(best_gains_layout)
        results_layout.addWidget(self.tuning_log_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

    def run_pid_tuning(self):
        self.start_tuning_button.setEnabled(False)
        self.tuning_log_text.clear()
        tuning_params = {
            'kp_min': self.kp_min_input.value(), 'kp_max': self.kp_max_input.value(),
            'ki_min': self.ki_min_input.value(), 'ki_max': self.ki_max_input.value(),
            'kd_min': self.kd_min_input.value(), 'kd_max': self.kd_max_input.value(),
            'iterations': self.tuning_iterations_input.value(),
            'objective': self.tuning_objective_input.currentText()
        }
        sim_params = {key: widget.value() for key, widget in self.param_inputs.items()}
        self.tuner_thread = QThread()
        self.tuner_worker = TunerWorker(tuning_params, sim_params)
        self.tuner_worker.moveToThread(self.tuner_thread)
        self.tuner_thread.started.connect(self.tuner_worker.run)
        self.tuner_worker.progress.connect(self.tuning_log_text.append)
        self.tuner_worker.best_gains.connect(self.update_best_gains)
        self.tuner_worker.finished.connect(self.on_tuning_finished)
        self.tuner_worker.finished.connect(self.tuner_thread.quit)
        self.tuner_worker.finished.connect(self.tuner_worker.deleteLater)
        self.tuner_thread.finished.connect(self.tuner_thread.deleteLater)
        self.tuner_thread.start()

    def on_tuning_finished(self):
        self.start_tuning_button.setEnabled(True)
        self.tuning_log_text.append("\n--- Tuning Finished ---")

    def update_best_gains(self, gains):
        self.best_kp_label.setText(f"{gains['Kp']:.4f}")
        self.best_ki_label.setText(f"{gains['Ki']:.4f}")
        self.best_kd_label.setText(f"{gains['Kd']:.4f}")

    def create_range_widget(self, min_box, max_box, default_range=(0.0, 1.0)):
        min_box.setRange(-100, 100); min_box.setValue(default_range[0])
        max_box.setRange(-100, 100); max_box.setValue(default_range[1])
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(min_box)
        layout.addWidget(QLabel("to"))
        layout.addWidget(max_box)
        layout.setContentsMargins(0,0,0,0)
        return widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
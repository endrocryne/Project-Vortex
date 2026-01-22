"""
GUI for Flight Dynamics Simulation
Tkinter interface with parameter configuration and plotting
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default (keeps CLI/testing headless)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Imports for embedding interactive plots in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from datetime import datetime
import csv
import os
import threading

from simulation import SuicideBurnSimulation


class SimulationGUI:
    """GUI for configuring and running simulations"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Suicide Burn Simulation - Project Vortex")
        self.root.geometry("900x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_rocket_tab()
        self.create_environment_tab()
        self.create_simulation_tab()
        self.create_run_tab()
        # Plots tab for embedded interactive figures
        self.create_plots_tab()
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Simulation object
        self.simulation = None
        self.running = False
        
    def create_rocket_tab(self):
        """Create rocket configuration tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Rocket")
        
        # Scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mass parameters
        frame = ttk.LabelFrame(scrollable_frame, text="Mass Parameters", padding=10)
        frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Dry Mass (kg):").grid(row=0, column=0, sticky='w')
        self.dry_mass = tk.DoubleVar(value=50.0)
        ttk.Entry(frame, textvariable=self.dry_mass).grid(row=0, column=1)
        
        ttk.Label(frame, text="Propellant Mass (kg):").grid(row=1, column=0, sticky='w')
        self.propellant_mass = tk.DoubleVar(value=10.0)
        ttk.Entry(frame, textvariable=self.propellant_mass).grid(row=1, column=1)
        
        # Geometry
        frame = ttk.LabelFrame(scrollable_frame, text="Geometry", padding=10)
        frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Length (m):").grid(row=0, column=0, sticky='w')
        self.length = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.length).grid(row=0, column=1)
        
        ttk.Label(frame, text="Diameter (m):").grid(row=1, column=0, sticky='w')
        self.diameter = tk.DoubleVar(value=0.3)
        ttk.Entry(frame, textvariable=self.diameter).grid(row=1, column=1)
        
        ttk.Label(frame, text="Use Dynamic CG/Inertia:").grid(row=2, column=0, sticky='w')
        self.use_dynamic_inertia = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, variable=self.use_dynamic_inertia).grid(row=2, column=1, sticky='w')
        
        # Thrust curve
        frame = ttk.LabelFrame(scrollable_frame, text="Thrust Curve", padding=10)
        frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Thrust Curve (t,T pairs):").grid(row=0, column=0, sticky='w')
        self.thrust_curve = tk.Text(frame, height=5, width=40)
        self.thrust_curve.insert('1.0', "0.0,0\n0.1,1000\n3.0,1000\n3.1,0")
        self.thrust_curve.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Label(frame, text="Burn Time (s):").grid(row=2, column=0, sticky='w')
        self.burn_time = tk.DoubleVar(value=3.0)
        ttk.Entry(frame, textvariable=self.burn_time).grid(row=2, column=1)
        
        # TVC parameters
        frame = ttk.LabelFrame(scrollable_frame, text="Thrust Vector Control", padding=10)
        frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Max Gimbal Angle (deg):").grid(row=0, column=0, sticky='w')
        self.tvc_max_angle = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.tvc_max_angle).grid(row=0, column=1)
        
        ttk.Label(frame, text="Response Time (s):").grid(row=1, column=0, sticky='w')
        self.tvc_response_time = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_response_time).grid(row=1, column=1)
        
        ttk.Label(frame, text="--- Pitch Control (Y-axis) ---").grid(row=2, column=0, columnspan=2, pady=(10,5))
        
        ttk.Label(frame, text="Kp (Proportional):").grid(row=3, column=0, sticky='w')
        self.tvc_kp_pitch = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.tvc_kp_pitch).grid(row=3, column=1)
        
        ttk.Label(frame, text="Ki (Integral):").grid(row=4, column=0, sticky='w')
        self.tvc_ki_pitch = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.tvc_ki_pitch).grid(row=4, column=1)
        
        ttk.Label(frame, text="Kd (Derivative):").grid(row=5, column=0, sticky='w')
        self.tvc_kd_pitch = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_kd_pitch).grid(row=5, column=1)
        
        ttk.Label(frame, text="--- Yaw Control (X-axis) ---").grid(row=6, column=0, columnspan=2, pady=(10,5))
        
        ttk.Label(frame, text="Kp (Proportional):").grid(row=7, column=0, sticky='w')
        self.tvc_kp_yaw = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.tvc_kp_yaw).grid(row=7, column=1)
        
        ttk.Label(frame, text="Ki (Integral):").grid(row=8, column=0, sticky='w')
        self.tvc_ki_yaw = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.tvc_ki_yaw).grid(row=8, column=1)
        
        ttk.Label(frame, text="Kd (Derivative):").grid(row=9, column=0, sticky='w')
        self.tvc_kd_yaw = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_kd_yaw).grid(row=9, column=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_environment_tab(self):
        """Create environment configuration tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Environment")
        
        # Scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Atmospheric parameters
        frame = ttk.LabelFrame(scrollable_frame, text="Atmosphere", padding=10)
        frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Gravity (m/s²):").grid(row=0, column=0, sticky='w')
        self.gravity = tk.DoubleVar(value=9.81)
        ttk.Entry(frame, textvariable=self.gravity).grid(row=0, column=1)
        
        ttk.Label(frame, text="Air Density (kg/m³):").grid(row=1, column=0, sticky='w')
        self.air_density = tk.DoubleVar(value=1.225)
        ttk.Entry(frame, textvariable=self.air_density).grid(row=1, column=1)
        
        ttk.Label(frame, text="Temperature (K):").grid(row=2, column=0, sticky='w')
        self.temperature = tk.DoubleVar(value=288.15)
        ttk.Entry(frame, textvariable=self.temperature).grid(row=2, column=1)
        
        ttk.Label(frame, text="Drag Coefficient:").grid(row=3, column=0, sticky='w')
        self.drag_coefficient = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.drag_coefficient).grid(row=3, column=1)
        
        # Wind parameters
        frame = ttk.LabelFrame(scrollable_frame, text="Wind", padding=10)
        frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Wind Model:").grid(row=0, column=0, sticky='w')
        self.wind_model = tk.StringVar(value="constant")
        wind_combo = ttk.Combobox(frame, textvariable=self.wind_model, 
                                  values=["constant", "altitude_varying", "gusts"])
        wind_combo.grid(row=0, column=1)
        
        ttk.Label(frame, text="Wind Speed (m/s):").grid(row=1, column=0, sticky='w')
        self.wind_speed = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.wind_speed).grid(row=1, column=1)
        
        ttk.Label(frame, text="Wind Direction (deg):").grid(row=2, column=0, sticky='w')
        self.wind_direction = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.wind_direction).grid(row=2, column=1)
        
        # Initial conditions
        frame = ttk.LabelFrame(scrollable_frame, text="Initial Conditions", padding=10)
        frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(frame, text="Initial Altitude (m):").grid(row=0, column=0, sticky='w')
        self.initial_altitude = tk.DoubleVar(value=1000.0)
        ttk.Entry(frame, textvariable=self.initial_altitude).grid(row=0, column=1)
        
        ttk.Label(frame, text="Initial Velocity (m/s):").grid(row=1, column=0, sticky='w')
        self.initial_velocity = tk.DoubleVar(value=-50.0)
        ttk.Entry(frame, textvariable=self.initial_velocity).grid(row=1, column=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_simulation_tab(self):
        """Create simulation parameters tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Simulation")
        
        frame = ttk.LabelFrame(tab, text="Monte Carlo Parameters", padding=10)
        frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame, text="Monte Carlo Runs:").grid(row=0, column=0, sticky='w')
        self.num_monte_carlo = tk.IntVar(value=100)
        ttk.Entry(frame, textvariable=self.num_monte_carlo).grid(row=0, column=1)
        
        ttk.Label(frame, text="Altitude Search Range (m):").grid(row=1, column=0, sticky='w')
        self.altitude_search_range = tk.DoubleVar(value=10.0)
        ttk.Entry(frame, textvariable=self.altitude_search_range).grid(row=1, column=1)
        
        ttk.Label(frame, text="Altitude Step (m):").grid(row=2, column=0, sticky='w')
        self.altitude_step = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.altitude_step).grid(row=2, column=1)

        # Toggle to show plots after run
        self.show_plots = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Show plots after run", variable=self.show_plots).grid(row=3, column=0, columnspan=2, sticky='w', pady=(5,0))
        
        # Variation parameters
        frame = ttk.LabelFrame(tab, text="Monte Carlo Variations", padding=10)
        frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame, text="Thrust Variation (±):").grid(row=0, column=0, sticky='w')
        self.thrust_variation = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.thrust_variation).grid(row=0, column=1)
        
        ttk.Label(frame, text="Drag Variation (±):").grid(row=1, column=0, sticky='w')
        self.drag_variation = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.drag_variation).grid(row=1, column=1)
        
        ttk.Label(frame, text="Air Density Variation (±):").grid(row=2, column=0, sticky='w')
        self.air_density_variation = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.air_density_variation).grid(row=2, column=1)
        
        ttk.Label(frame, text="TVC Response Variation (±):").grid(row=3, column=0, sticky='w')
        self.tvc_response_variation = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_response_variation).grid(row=3, column=1)
        
        ttk.Label(frame, text="Mass Variation (±):").grid(row=4, column=0, sticky='w')
        self.mass_variation = tk.DoubleVar(value=0.02)
        ttk.Entry(frame, textvariable=self.mass_variation).grid(row=4, column=1)
        
        ttk.Label(frame, text="Altimeter Error (±):").grid(row=5, column=0, sticky='w')
        self.altimeter_error = tk.DoubleVar(value=0.01)
        ttk.Entry(frame, textvariable=self.altimeter_error).grid(row=5, column=1)
        
        ttk.Label(frame, text="Velocity Sensor Error (±):").grid(row=6, column=0, sticky='w')
        self.velocity_sensor_error = tk.DoubleVar(value=0.01)
        ttk.Entry(frame, textvariable=self.velocity_sensor_error).grid(row=6, column=1)
        
    def create_run_tab(self):
        """Create run simulation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Run")

        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Optimization", 
                                      command=self.run_optimization)
        self.run_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Run Single Simulation", 
                  command=self.run_single).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Save Configuration", 
                  command=self.save_config).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Load Configuration", 
                  command=self.load_config).pack(side='left', padx=5)
        
        # Progress bar for optimization runs
        progress_frame = ttk.Frame(tab)
        progress_frame.pack(fill='x', padx=5, pady=(5,0))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode='determinate', maximum=100)
        self.progress_bar.pack(fill='x', side='left', expand=True, padx=(0,5))
        self.progress_label = ttk.Label(progress_frame, text="0/0")
        self.progress_label.pack(side='left')
        
        # Output text
        frame = ttk.LabelFrame(tab, text="Output", padding=5)
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.output_text = tk.Text(frame, wrap='word', height=20)
        self.output_text.pack(fill='both', expand=True, side='left')
        
        scrollbar = ttk.Scrollbar(frame, command=self.output_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.output_text.config(yscrollcommand=scrollbar.set)

    def create_plots_tab(self):
        """Create Plots tab to embed interactive matplotlib canvases"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Plots")
        self.plots_tab = tab

        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Clear Plots", command=self._clear_plots).pack(side='left')
        ttk.Label(control_frame, text="  Interactive plots appear here after a run").pack(side='left', padx=10)

        # Frame where figures will be embedded
        self.plots_frame = ttk.Frame(tab)
        self.plots_frame.pack(fill='both', expand=True)

        # Track canvas and toolbar widgets for cleanup
        self._plot_widgets = []
        self._plot_toolbars = []
        # Track mpl connection ids for interactive events so we can disconnect them
        self._plot_cids = []

    def _clear_plots(self):
        """Remove embedded plot canvases and toolbars"""
        for w in self._plot_widgets:
            try:
                w.get_tk_widget().destroy()
            except Exception:
                try:
                    w.destroy()
                except Exception:
                    print("Mishra Python Runtime: Error when [AI] destroying plot widget. [AI]")
        # Disconnect any mpl event connections
        for canvas, cid in list(self._plot_cids):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                print("Mishra Python Runtime: Error when [AI] disconnecting mpl event. [AI]")
        self._plot_cids = []

        for tb in self._plot_toolbars:
            try:
                tb.destroy()
            except Exception:
                print("Mishra Python Runtime: Error when [AI] destroying plot toolbar. [AI]")
        self._plot_widgets = []
        self._plot_toolbars = []
        # Also clear any remaining children in the plots_frame
        for child in list(self.plots_frame.winfo_children()):
            child.destroy()        
        # (Run tab widgets moved to create_run_tab)
        
    def parse_thrust_curve(self):
        """Parse thrust curve from text widget"""
        text = self.thrust_curve.get('1.0', 'end')
        lines = text.strip().split('\n')
        curve = []
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    t = float(parts[0].strip())
                    T = float(parts[1].strip())
                    curve.append([t, T])
        return curve
    
    def get_configs(self):
        """Get configuration dictionaries from GUI"""
        rocket_config = {
            'dry_mass': self.dry_mass.get(),
            'propellant_mass': self.propellant_mass.get(),
            'length': self.length.get(),
            'diameter': self.diameter.get(),
            'use_dynamic_inertia': self.use_dynamic_inertia.get(),
            'thrust_curve': self.parse_thrust_curve(),
            'burn_time': self.burn_time.get(),
            'tvc_max_angle': self.tvc_max_angle.get(),
            'tvc_response_time': self.tvc_response_time.get(),
            'tvc_kp_pitch': self.tvc_kp_pitch.get(),
            'tvc_ki_pitch': self.tvc_ki_pitch.get(),
            'tvc_kd_pitch': self.tvc_kd_pitch.get(),
            'tvc_kp_yaw': self.tvc_kp_yaw.get(),
            'tvc_ki_yaw': self.tvc_ki_yaw.get(),
            'tvc_kd_yaw': self.tvc_kd_yaw.get(),
            'thrust_variation': self.thrust_variation.get(),
            'tvc_response_variation': self.tvc_response_variation.get(),
            'mass_variation': self.mass_variation.get(),
        }
        
        environment_config = {
            'gravity': self.gravity.get(),
            'air_density': self.air_density.get(),
            'temperature': self.temperature.get(),
            'drag_coefficient': self.drag_coefficient.get(),
            'reference_area': np.pi * (self.diameter.get() / 2) ** 2,
            'wind_model': self.wind_model.get(),
            'wind_speed': self.wind_speed.get(),
            'wind_direction': self.wind_direction.get(),
            'drag_variation': self.drag_variation.get(),
            'air_density_variation': self.air_density_variation.get(),
        }
        
        simulation_config = {
            'num_monte_carlo': self.num_monte_carlo.get(),
            'altitude_search_range': self.altitude_search_range.get(),
            'altitude_step': self.altitude_step.get(),
            'altimeter_error': self.altimeter_error.get(),
            'velocity_sensor_error': self.velocity_sensor_error.get(),
            'show_plots': self.show_plots.get(),
        }
        
        return rocket_config, environment_config, simulation_config
    
    def log(self, message):
        """Log message to output text widget"""
        self.output_text.insert('end', message + '\n')
        self.output_text.see('end')
        self.root.update()

    def _progress_callback(self, completed, total):
        """Thread-safe progress callback passed to simulation.optimize_ignition_altitude.
        Called from simulation thread; schedules UI update on the main thread."""
        # Use after to ensure GUI updates happen on main thread
        try:
            self.root.after(0, lambda: self._update_progress(completed, total))
        except Exception:
            print("Mishra Python Runtime: Error when [AI] scheduling progress update. [AI]")

    def _update_progress(self, completed, total):
        """Update progress bar and label on the GUI thread."""
        try:
            if total <= 0:
                # Reset to zero state
                self.progress_bar.config(maximum=1)
                self.progress_var.set(0)
                self.progress_label.config(text=f"{completed}/{total}")
                self.status_bar.config(text="Optimization progress: 0/0")
                return

            self.progress_bar.config(maximum=total)
            # progress_var is bound to the bar; set numeric completed value
            self.progress_var.set(completed)
            self.progress_label.config(text=f"{completed}/{total}")
            self.status_bar.config(text=f"Optimization progress: {completed}/{total}")
        except Exception:
            print("Mishra Python Runtime: Error when [AI] scheduling progress update. [AI]")
        
    def run_optimization(self):
        """Run Monte Carlo optimization in a separate thread"""
        if self.running:
            messagebox.showwarning("Warning", "Simulation already running")
            return
        
        self.running = True
        self.run_button.config(state='disabled')
        
        thread = threading.Thread(target=self._run_optimization_thread)
        thread.start()
        
    def _run_optimization_thread(self):
        """Thread function for running optimization"""
        try:
            self.output_text.delete('1.0', 'end')
            self.log("Starting optimization...")
            
            # Get configurations
            rocket_config, environment_config, simulation_config = self.get_configs()
            
            # Create simulation
            self.simulation = SuicideBurnSimulation(
                rocket_config, environment_config, simulation_config
            )
            
            # Initial state
            initial_altitude = self.initial_altitude.get()
            initial_velocity = self.initial_velocity.get()
            
            initial_state = np.array([
                0, 0, initial_altitude,  # position
                0, 0, initial_velocity,  # velocity
                1, 0, 0, 0,  # quaternion (identity)
                0, 0, 0,  # angular velocity
                rocket_config['dry_mass'] + rocket_config['propellant_mass']  # mass
            ])
            
            # Reset progress UI
            self._update_progress(0, 0)

            # Create results folder up-front so per-trial outputs can be written during optimization
            results_folder, ts = self._make_results_subfolder('optimization')

            # Run optimization (pass GUI progress callback), save per-trial CSVs and PNGs
            optimal_altitude, success_rates, best_history = self.simulation.optimize_ignition_altitude(
                initial_state,
                simulation_config['num_monte_carlo'],
                simulation_config['altitude_search_range'],
                simulation_config['altitude_step'],
                progress_callback=self._progress_callback,
                save_each_trial=True,
                results_folder=results_folder,
                save_plots_per_trial=True
            )
            
            self.log(f"\nOptimal ignition altitude: {optimal_altitude:.2f} m")
            
            # Save summary results into the same folder
            self.save_results(success_rates, best_history, folder=results_folder)
            
            # Generate plots (overall) into the same folder
            self.generate_plots(success_rates, best_history, results_folder)
            
            self.log("\nOptimization complete! Check the 'results' directory for outputs.")
            self.status_bar.config(text="Optimization complete")
            
            messagebox.showinfo("Complete", "Optimization complete! Check results directory.")
            
        except Exception as e:
            self.log(f"\nError: {str(e)}")
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            
        finally:
            self.running = False
            self.run_button.config(state='normal')
            
    def run_single(self):
        """Run a single simulation"""
        try:
            self.output_text.delete('1.0', 'end')
            self.log("Running single simulation...")
            
            # Get configurations
            rocket_config, environment_config, simulation_config = self.get_configs()
            
            # Create simulation
            self.simulation = SuicideBurnSimulation(
                rocket_config, environment_config, simulation_config
            )
            
            # Initial state
            initial_altitude = self.initial_altitude.get()
            initial_velocity = self.initial_velocity.get()
            
            initial_state = np.array([
                0, 0, initial_altitude,
                0, 0, initial_velocity,
                1, 0, 0, 0,
                0, 0, 0,
                rocket_config['dry_mass'] + rocket_config['propellant_mass']
            ])
            
            # Calculate ignition altitude
            ignition_altitude = self.simulation.calculate_ignition_altitude(
                initial_velocity, initial_altitude
            )
            
            self.log(f"Calculated ignition altitude: {ignition_altitude:.2f} m")
            
            # Run simulation
            success, final_state, history = self.simulation.run_simulation(
                initial_state, ignition_altitude
            )
            
            self.log(f"Success: {success}")
            self.log(f"Final altitude: {history['final_altitude']:.2f} m")
            self.log(f"Final velocity: {history['final_velocity']:.2f} m/s")
            
            # Save and plot
            results_folder = self.save_single_result(history)
            self.generate_single_plots(history, results_folder)
            
            self.log("\nSimulation complete! Check the 'results' directory.")
            
            messagebox.showinfo("Complete", "Simulation complete! Check results directory.")
            
        except Exception as e:
            self.log(f"\nError: {str(e)}")
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            
    def _make_results_subfolder(self, prefix):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join('results', f'{prefix}_{timestamp}')
        os.makedirs(folder, exist_ok=True)
        return folder, timestamp

    def save_results(self, success_rates, history, folder=None):
        """Save optimization results to CSV inside a timestamped subfolder
        If folder is provided, use it; otherwise create a new timestamped folder."""
        if folder is None:
            folder, timestamp = self._make_results_subfolder('optimization')
        
        # Save success rates
        filename = os.path.join(folder, 'optimization.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Ignition Altitude (m)', 'Success Rate'])
            for altitude, rate in sorted(success_rates.items()):
                writer.writerow([altitude, rate])
        
        self.log(f"Saved optimization results to {filename}")
        
        # Save best trajectory
        if history:
            filename = os.path.join(folder, 'trajectory.csv')
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass'])
                for i in range(len(history['t'])):
                    writer.writerow([
                        history['t'][i],
                        history['x'][i],
                        history['y'][i],
                        history['z'][i],
                        history['vx'][i],
                        history['vy'][i],
                        history['vz'][i],
                        history['qw'][i],
                        history['qx'][i],
                        history['qy'][i],
                        history['qz'][i],
                        history['mass'][i]
                    ])
            
            self.log(f"Saved trajectory to {filename}")
        
        # Return the folder where results were saved for further processing
        return folder
            
    def save_single_result(self, history, folder=None):
        """Save single simulation result to CSV inside a timestamped subfolder
        If folder is provided, save into it; otherwise create one."""
        if folder is None:
            folder, timestamp = self._make_results_subfolder('single_run')

        filename = os.path.join(folder, 'single_run.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass'])
            for i in range(len(history['t'])):
                writer.writerow([
                    history['t'][i],
                    history['x'][i],
                    history['y'][i],
                    history['z'][i],
                    history['vx'][i],
                    history['vy'][i],
                    history['vz'][i],
                    history['qw'][i],
                    history['qx'][i],
                    history['qy'][i],
                    history['qz'][i],
                    history['mass'][i]
                ])
        
        self.log(f"Saved trajectory to {filename}")
        return folder
        
    def generate_plots(self, success_rates, history, folder):
        """Generate plots for optimization results and save them into `folder`"""
        # Success rate vs altitude
        altitudes = sorted(success_rates.keys())
        rates = [success_rates[a] for a in altitudes]

        saved_files = []
        success_rate_path = os.path.join(folder, 'success_rate.png')
        plt.figure(figsize=(10, 6))
        plt.plot(altitudes, rates, 'b-', linewidth=2)
        plt.xlabel('Ignition Altitude (m)')
        plt.ylabel('Success Rate')
        plt.title('Success Rate vs Ignition Altitude')
        plt.grid(True)
        plt.savefig(success_rate_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(success_rate_path)

        if history:
            traj_files = self.generate_trajectory_plots(history, folder)
            saved_files.extend(traj_files)

        # Open interactive windows if requested
        if hasattr(self, 'show_plots') and self.show_plots.get():
            try:
                self.display_interactive_plots(success_rates, history)
            except Exception as e:
                self.log(f"Could not open interactive plots: {e}")
                # Fallback to opening saved images
                for fpath in saved_files:
                    try:
                        os.startfile(fpath)
                    except Exception as ex:
                        self.log(f"Could not open {fpath}: {ex}")
            
    def generate_single_plots(self, history, folder):
        """Generate plots for single simulation and save into `folder`"""
        saved_files = self.generate_trajectory_plots(history, folder)

        # Open interactive windows if requested
        if hasattr(self, 'show_plots') and self.show_plots.get():
            try:
                self.display_interactive_plots(None, history)
            except Exception as e:
                self.log(f"Could not open interactive plots: {e}")
                # Fallback to opening saved images
                for fpath in saved_files:
                    try:
                        os.startfile(fpath)
                    except Exception as ex:
                        self.log(f"Could not open {fpath}: {ex}")
        
    def generate_trajectory_plots(self, history, folder):
        """Generate trajectory plots and save into `folder`"""
        t = history['t']
        
        # 2D trajectory (altitude vs time)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t, history['z'], 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title('Altitude vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(t, history['vz'], 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Vertical Velocity (m/s)')
        plt.title('Vertical Velocity vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        speed = np.sqrt(history['vx']**2 + history['vy']**2 + history['vz']**2)
        plt.plot(t, speed, 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Total Speed vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(t, history['mass'], 'k-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Mass (kg)')
        plt.title('Mass vs Time')
        plt.grid(True)
        
        saved_files = []

        plt.tight_layout()
        path_2d = os.path.join(folder, 'trajectory_2d.png')
        plt.savefig(path_2d, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path_2d)
        
        # 3D trajectory
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(history['x'], history['y'], history['z'], 'b-', linewidth=2)
        ax.scatter([history['x'][0]], [history['y'][0]], [history['z'][0]], 
                  c='g', s=100, marker='o', label='Start')
        ax.scatter([history['x'][-1]], [history['y'][-1]], [history['z'][-1]], 
                  c='r', s=100, marker='x', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
        
        path_3d = os.path.join(folder, 'trajectory_3d.png')
        plt.savefig(path_3d, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(path_3d)
        
        self.log(f"Saved plots to {folder}")

        return saved_files

    def display_interactive_plots(self, success_rates, history):
        """Embed interactive matplotlib figures into the `Plots` tab using TkAgg.
        This keeps the figures inside the GUI and uses toolbar controls."""
        # Try to switch backend to TkAgg for embedding (force if necessary)
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception as e:
            self.log(f"Could not switch matplotlib backend to TkAgg: {e}")
            return

        # Ensure the plots tab exists
        if not hasattr(self, 'plots_tab'):
            self.log("Plots tab not available")
            return

        # Clear previous embedded plots
        self._clear_plots()

        try:
            # Success rate plot
            if success_rates:
                altitudes = sorted(success_rates.keys())
                rates = [success_rates[a] for a in altitudes]
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.plot(altitudes, rates, 'b-', linewidth=2)
                ax.set_xlabel('Ignition Altitude (m)')
                ax.set_ylabel('Success Rate')
                ax.set_title('Success Rate vs Ignition Altitude')
                ax.grid(True)

                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.pack(fill='both', expand=True, pady=(5, 5))
                self._plot_widgets.append(canvas)

                toolbar = NavigationToolbar2Tk(canvas, self.plots_frame)
                toolbar.update()
                toolbar.pack(fill='x')
                self._plot_toolbars.append(toolbar)

            # Trajectory 2D
            if history:
                fig2 = Figure(figsize=(12, 8))
                ax1 = fig2.add_subplot(2, 2, 1)
                t = history['t']
                ax1.plot(t, history['z'], 'b-', linewidth=2)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Altitude (m)')
                ax1.set_title('Altitude vs Time')
                ax1.grid(True)

                ax2 = fig2.add_subplot(2, 2, 2)
                ax2.plot(t, history['vz'], 'r-', linewidth=2)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Vertical Velocity (m/s)')
                ax2.set_title('Vertical Velocity vs Time')
                ax2.grid(True)

                ax3 = fig2.add_subplot(2, 2, 3)
                speed = np.sqrt(history['vx']**2 + history['vy']**2 + history['vz']**2)
                ax3.plot(t, speed, 'g-', linewidth=2)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Speed (m/s)')
                ax3.set_title('Total Speed vs Time')
                ax3.grid(True)

                ax4 = fig2.add_subplot(2, 2, 4)
                ax4.plot(t, history['mass'], 'k-', linewidth=2)
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Mass (kg)')
                ax4.set_title('Mass vs Time')
                ax4.grid(True)

                canvas2 = FigureCanvasTkAgg(fig2, master=self.plots_frame)
                canvas2.draw()
                widget2 = canvas2.get_tk_widget()
                widget2.pack(fill='both', expand=True, pady=(5, 5))
                self._plot_widgets.append(canvas2)

                toolbar2 = NavigationToolbar2Tk(canvas2, self.plots_frame)
                toolbar2.update()
                toolbar2.pack(fill='x')
                self._plot_toolbars.append(toolbar2)

                # 3D trajectory
                fig3 = Figure(figsize=(10, 10))
                ax3d = fig3.add_subplot(111, projection='3d')
                ax3d.plot(history['x'], history['y'], history['z'], 'b-', linewidth=2)
                ax3d.scatter([history['x'][0]], [history['y'][0]], [history['z'][0]],
                             c='g', s=100, marker='o', label='Start')
                ax3d.scatter([history['x'][-1]], [history['y'][-1]], [history['z'][-1]],
                             c='r', s=100, marker='x', label='End')
                ax3d.set_xlabel('X (m)')
                ax3d.set_ylabel('Y (m)')
                ax3d.set_zlabel('Z (m)')
                ax3d.set_title('3D Trajectory')
                ax3d.legend()

                canvas3 = FigureCanvasTkAgg(fig3, master=self.plots_frame)
                canvas3.draw()
                widget3 = canvas3.get_tk_widget()
                widget3.pack(fill='both', expand=True, pady=(5, 5))
                self._plot_widgets.append(canvas3)

                toolbar3 = NavigationToolbar2Tk(canvas3, self.plots_frame)
                toolbar3.update()
                toolbar3.pack(fill='x')
                self._plot_toolbars.append(toolbar3)

                # Add interactive handlers: rotate on drag, zoom on scroll
                press = {'x': None, 'y': None, 'elev': None, 'azim': None}

                def on_press(event):
                    try:
                        if event.inaxes == ax3d and event.button == 1:
                            press['x'] = event.x
                            press['y'] = event.y
                            press['elev'] = ax3d.elev
                            press['azim'] = ax3d.azim
                    except Exception:
                        print("Mishra Python Runtime: Error in on_press event. \n Mishra Runtime Integrity has detected the following libraries to be corrupt. Check your installation. \n 1. Vortex Simulation Desktop 0.6 \n 2. TUI Beta 0.2")

                def on_release(event):
                    press['x'] = None
                    press['y'] = None
                    press['elev'] = None
                    press['azim'] = None

                def on_motion(event):
                    if press['x'] is None or event.inaxes != ax3d:
                        return
                    try:
                        dx = event.x - press['x']
                        dy = event.y - press['y']
                        az = press['azim'] - dx * 0.5
                        el = press['elev'] - dy * 0.5
                        ax3d.view_init(elev=el, azim=az)
                        canvas3.draw_idle()
                    except Exception:
                        print("Mishra Python Runtime: Error in on_motion event. \n Mishra Runtime Integrity has detected the following libraries to be corrupt. Check your installation. \n 1. Vortex Simulation 0.6 \n 2. TUI Beta 0.2")

                def on_scroll(event):
                    if event.inaxes != ax3d:
                        return
                    try:
                        # 'up'/'down' indicate scroll direction
                        if getattr(event, 'button', None) == 'up':
                            ax3d.dist = max(getattr(ax3d, 'dist', 1) - 1, 1)
                        else:
                            ax3d.dist = getattr(ax3d, 'dist', 10) + 1
                        canvas3.draw_idle()
                    except Exception:
                        print("Mishra Python Runtime: Error in on_scroll event. \n Mishra Runtime Integrity has detected the following libraries to be corrupt. Check your installation. \n 1. Vortex Simulation 0.6 \n 2. TUI Beta 0.2")

                # Register connections and keep track for cleanup
                try:
                    cid1 = canvas3.mpl_connect('button_press_event', on_press)
                    cid2 = canvas3.mpl_connect('button_release_event', on_release)
                    cid3 = canvas3.mpl_connect('motion_notify_event', on_motion)
                    cid4 = canvas3.mpl_connect('scroll_event', on_scroll)
                    self._plot_cids.append((canvas3, cid1))
                    self._plot_cids.append((canvas3, cid2))
                    self._plot_cids.append((canvas3, cid3))
                    self._plot_cids.append((canvas3, cid4))
                except Exception:
                    print("Mishra Python Runtime: Error when [AI] connecting mpl events. [AI]")

            # Switch to Plots tab so user sees embedded figures
            try:
                self.notebook.select(self.plots_tab)
            except Exception:
                pass

        except Exception as e:
            self.log(f"Error embedding interactive plots: {e}")

    def save_config(self):
        """Save configuration to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement config save
            messagebox.showinfo("Info", "Configuration save not yet implemented")
            
    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement config load
            messagebox.showinfo("Info", "Configuration load not yet implemented")


def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

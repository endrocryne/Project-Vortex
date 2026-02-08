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
        self.root.title("HexaKinetic Simulation - Vortex Desktop")
        self.root.geometry("1000x900") # Slightly larger to accommodate banner
        
        # Set Icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'hexakinetic_icon.png')
            if os.path.exists(icon_path):
                self.icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, self.icon_img)
                # Fix taskbar icon on Windows
                import ctypes
                myappid = 'vortex.hexakinetic.gui.v0.4'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
        
        # Menu Bar
        self.create_menu()
        
        # Banner Frame (Top)
        self.banner_frame = tk.Frame(root, bg="#d9534f", pady=5)
        self.banner_label = tk.Label(self.banner_frame, text=" Warning from [Mishra Physics Engine]: Current configuration is impossible and cannot land.", 
                                     fg="white", bg="#d9534f", font=("Arial", 11, "bold"))
        self.banner_label.pack(side='left', expand=True, padx=(20, 0))
        self.btn_dismiss_banner = tk.Button(self.banner_frame, text="Dismiss", 
                                            command=self.dismiss_banner, bg="#d9534f", fg="white", 
                                            relief='flat', font=("Arial", 9, "bold"), cursor="hand2")
        self.btn_dismiss_banner.pack(side='right', padx=10)
        self.banner_frame.pack_forget() # Hidden by default
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_rocket_tab()
        self.create_environment_tab()
        self.create_simulation_tab()
        self.create_fault_injection_tab()
        self.create_run_tab()
        # Plots tab for embedded interactive figures
        self.create_plots_tab()
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # State
        self.simulation = None
        self.running = False
        self.banner_dismissed_manually = False
        self.last_feasibility_state = True # True = Possible
        self.last_result_csv = None # Track the latest result CSV for HexaVisual
        
        # Setup continuous monitoring
        self.setup_monitoring()

    def create_menu(self):
        """Create the top menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Preferences", command=self.open_preferences)
        file_menu.add_separator()
        file_menu.add_command(label="About", command=self.open_about)
        file_menu.add_separator()
        file_menu.add_command(label="Quit App", command=self.root.quit)
        
        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences", command=self.open_preferences)
        edit_menu.add_separator()
        edit_menu.add_command(label="HexaVisual", command=self.launch_hexavisual)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=lambda: self.log("Zoom In (Placeholder)"))
        view_menu.add_command(label="Zoom Out", command=lambda: self.log("Zoom Out (Placeholder)"))
        view_menu.add_command(label="Fit to Screen", command=lambda: self.log("Fit to Screen (Placeholder)"))

    def open_preferences(self):
        """Open Preferences window"""
        top = tk.Toplevel(self.root)
        top.title("Preferences")
        top.geometry("400x300")
        
        ttk.Label(top, text="Results Directory:", font=("Arial", 10, "bold")).pack(anchor='w', padx=10, pady=(10,5))
        
        dir_frame = ttk.Frame(top)
        dir_frame.pack(fill='x', padx=10)
        self.res_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), 'results'))
        ttk.Entry(dir_frame, textvariable=self.res_dir_var).pack(side='left', fill='x', expand=True)
        ttk.Button(dir_frame, text="Browse", command=lambda: self.res_dir_var.set(filedialog.askdirectory())).pack(side='right', padx=5)
        
        ttk.Label(top, text="Plots to Save:", font=("Arial", 10, "bold")).pack(anchor='w', padx=10, pady=(15,5))
        
        self.chk_save_3d = tk.BooleanVar(value=True)
        self.chk_save_2d = tk.BooleanVar(value=True)
        self.chk_save_vel = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(top, text="3D Trajectory", variable=self.chk_save_3d).pack(anchor='w', padx=20)
        ttk.Checkbutton(top, text="2D Altitude/Time", variable=self.chk_save_2d).pack(anchor='w', padx=20)
        ttk.Checkbutton(top, text="Velocity Profiles", variable=self.chk_save_vel).pack(anchor='w', padx=20)
        
        ttk.Label(top, text="ML Model Paths:", font=("Arial", 10, "bold")).pack(anchor='w', padx=10, pady=(15,5))
        
        # Model
        m_frame = ttk.Frame(top)
        m_frame.pack(fill='x', padx=10)
        ttk.Label(m_frame, text="Model (.keras):").pack(side='left')
        self.ml_model_path_var = tk.StringVar(value='ignition_model_improved.keras')
        ttk.Entry(m_frame, textvariable=self.ml_model_path_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(m_frame, text="...", width=3, command=lambda: self.ml_model_path_var.set(filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")]))).pack(side='right')

        # Scaler
        s_frame = ttk.Frame(top)
        s_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(s_frame, text="Scaler (.pkl):  ").pack(side='left')
        self.ml_scaler_path_var = tk.StringVar(value='scaler_improved.pkl')
        ttk.Entry(s_frame, textvariable=self.ml_scaler_path_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(s_frame, text="...", width=3, command=lambda: self.ml_scaler_path_var.set(filedialog.askopenfilename(filetypes=[("Pickle Scaler", "*.pkl"), ("All Files", "*.*")]))).pack(side='right')

        ttk.Button(top, text="Close", command=top.destroy).pack(pady=20)

    def open_about(self):
        """Open About window"""
        messagebox.showinfo("About", "HexaKinetic Simulation\nVersion 0.4\n\nThis software is part of the Vortex Desktop package. Powered by Python. \n\nPowered by Mishra Physics Engine\n\n©️ 2026 Agastya Mishra")

    def launch_hexavisual(self):
        """Launch the Advanced Visualizer (HexaVisual)"""
        import subprocess
        try:
            # Use same python executable
            import sys
            subprocess.Popen([sys.executable, "advanced_visualizer.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch HexaVisual: {e}")

    def setup_monitoring(self):
        """Setup traces and timers to monitor config changes"""
        # Traces for numeric variables
        vars_to_trace = [
            self.dry_mass, self.propellant_mass, self.gravity,
            self.initial_altitude, self.initial_velocity
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *args: self.root.after(100, self.check_live_feasibility))
            
        # Periodic check for text-based controls (thrust curve)
        self.periodic_check()

    def periodic_check(self):
        """Check things that don't have built-in traces every second"""
        self.check_live_feasibility()
        self.root.after(1000, self.periodic_check)

    def dismiss_banner(self):
        """Manually hide the banner until next restart"""
        self.banner_dismissed_manually = True
        self.banner_frame.pack_forget()

    def check_live_feasibility(self):
        """Non-blocking feasibility check for the UI banner"""
        try:
            # Lightweight check
            rocket_config, env_config, sim_config, fault_config, ml_config = self.get_configs()
            sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
            
            initial_v = self.initial_velocity.get()
            initial_h = self.initial_altitude.get()
            
            is_possible, r = sim.check_feasibility(initial_v, initial_h)
            
            if is_possible:
                # If it became possible, we reset manual dismissal
                if not self.last_feasibility_state: 
                    self.banner_dismissed_manually = False
                self.banner_frame.pack_forget()
            else:
                # Show banner if not manually dismissed
                if not self.banner_dismissed_manually:
                    self.banner_frame.pack(side='top', fill='x', before=self.notebook)
            
            self.last_feasibility_state = is_possible
            
        except Exception:
            # Silence errors during live typing/parsing
            pass
        
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
        
        # Ascent Motor (New)
        ttk.Label(frame, text="Ascent Motor Dry Mass (kg):", font=("Arial", 9)).grid(row=2, column=0, sticky='w')
        self.ascent_motor_casing_mass = tk.DoubleVar(value=2.0)
        ttk.Entry(frame, textvariable=self.ascent_motor_casing_mass).grid(row=2, column=1)
        ttk.Label(frame, text="(Only used if 'Simulate Ascent' is checked)", font=("Arial", 8, "italic"), foreground="gray").grid(row=3, column=0, columnspan=2, sticky='w', padx=10)
        
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
        
        ttk.Label(frame, text="TVC Control Mode:").grid(row=2, column=0, sticky='w')
        self.tvc_mode = tk.StringVar(value='orientation')
        self.cb_tvc_mode = ttk.Combobox(frame, textvariable=self.tvc_mode, 
                                        values=['orientation', 'velocity'], state='readonly')
        self.cb_tvc_mode.grid(row=2, column=1)
        self.cb_tvc_mode.bind("<<ComboboxSelected>>", self._update_tvc_mode_visibility)
        
        self.lbl_drift_gain = ttk.Label(frame, text="Drift Correction Gain:")
        self.lbl_drift_gain.grid(row=3, column=0, sticky='w')
        self.tvc_drift_gain = tk.DoubleVar(value=0.1)
        self.ent_drift_gain = ttk.Entry(frame, textvariable=self.tvc_drift_gain)
        self.ent_drift_gain.grid(row=3, column=1)
        
        ttk.Label(frame, text="--- Pitch Control (Y-axis) ---").grid(row=4, column=0, columnspan=2, pady=(10,5))
        
        ttk.Label(frame, text="Kp (Proportional):").grid(row=5, column=0, sticky='w')
        self.tvc_kp_pitch = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.tvc_kp_pitch).grid(row=5, column=1)
        
        ttk.Label(frame, text="Ki (Integral):").grid(row=6, column=0, sticky='w')
        self.tvc_ki_pitch = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.tvc_ki_pitch).grid(row=6, column=1)
        
        ttk.Label(frame, text="Kd (Derivative):").grid(row=7, column=0, sticky='w')
        self.tvc_kd_pitch = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_kd_pitch).grid(row=7, column=1)
        
        ttk.Label(frame, text="--- Yaw Control (X-axis) ---").grid(row=8, column=0, columnspan=2, pady=(10,5))
        
        ttk.Label(frame, text="Kp (Proportional):").grid(row=9, column=0, sticky='w')
        self.tvc_kp_yaw = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.tvc_kp_yaw).grid(row=9, column=1)
        
        ttk.Label(frame, text="Ki (Integral):").grid(row=10, column=0, sticky='w')
        self.tvc_ki_yaw = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.tvc_ki_yaw).grid(row=10, column=1)
        
        ttk.Label(frame, text="Kd (Derivative):").grid(row=11, column=0, sticky='w')
        self.tvc_kd_yaw = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.tvc_kd_yaw).grid(row=11, column=1)

        # Initialize visibility
        self._update_tvc_mode_visibility()
        
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
        
        self.lbl_altitude_step = ttk.Label(frame, text="Altitude Step (m):")
        self.lbl_altitude_step.grid(row=2, column=0, sticky='w')
        self.altitude_step = tk.DoubleVar(value=0.1)
        self.ent_altitude_step = ttk.Entry(frame, textvariable=self.altitude_step)
        self.ent_altitude_step.grid(row=2, column=1)

        ttk.Label(frame, text="Ignition Percent Offset (%):").grid(row=3, column=0, sticky='w')
        self.ignition_percent_offset = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.ignition_percent_offset).grid(row=3, column=1)

        ttk.Label(frame, text="Ignition Hard Offset (m):").grid(row=4, column=0, sticky='w')
        self.ignition_hard_offset = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.ignition_hard_offset).grid(row=4, column=1)

        # Adaptive Optimization Settings (New)
        ttk.Label(frame, text="Optimization Mode:").grid(row=5, column=0, sticky='w', pady=(10,0))
        self.opt_mode = tk.StringVar(value="grid")
        self.cb_opt_mode = ttk.Combobox(frame, textvariable=self.opt_mode, values=["grid", "adaptive"], state="readonly")
        self.cb_opt_mode.grid(row=5, column=1, pady=(10,0))
        self.cb_opt_mode.bind("<<ComboboxSelected>>", self._update_optimization_mode_visibility)

        self.lbl_max_iters = ttk.Label(frame, text="Max Iterations (Adaptive):")
        self.lbl_max_iters.grid(row=6, column=0, sticky='w')
        self.opt_max_iterations = tk.IntVar(value=3)
        self.ent_max_iters = ttk.Entry(frame, textvariable=self.opt_max_iterations)
        self.ent_max_iters.grid(row=6, column=1)

        self.lbl_samples = ttk.Label(frame, text="Samples per Step (Adaptive):")
        self.lbl_samples.grid(row=7, column=0, sticky='w')
        self.opt_samples_per_step = tk.IntVar(value=10)
        self.ent_samples = ttk.Entry(frame, textvariable=self.opt_samples_per_step)
        self.ent_samples.grid(row=7, column=1)

        self.lbl_target_step = ttk.Label(frame, text="Target Step (m) (Adaptive):")
        self.lbl_target_step.grid(row=8, column=0, sticky='w')
        self.opt_target_step = tk.DoubleVar(value=0.01)
        self.ent_target_step = ttk.Entry(frame, textvariable=self.opt_target_step)
        self.ent_target_step.grid(row=8, column=1)
        
        # Initial visibility update
        self._update_optimization_mode_visibility()

        # Ascent & Orientation
        frame_ascent = ttk.LabelFrame(tab, text="Flight Phase & Orientation", padding=10)
        frame_ascent.pack(fill='x', padx=5, pady=5)
        
        self.simulate_ascent = tk.BooleanVar(value=False)
        self.chk_ascent = ttk.Checkbutton(frame_ascent, text="Simulate Ascent Phase (Launch -> Apogee -> Descent)", 
                                          variable=self.simulate_ascent, command=self.update_orientation_labels)
        self.chk_ascent.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))
        
        self.lbl_orientation = ttk.Label(frame_ascent, text="Burn Start Orientation (deg):", font=("Arial", 9, "bold"))
        self.lbl_orientation.grid(row=1, column=0, columnspan=2, sticky='w')
        
        ttk.Label(frame_ascent, text="Pitch (deg):").grid(row=2, column=0, sticky='w')
        self.start_pitch = tk.DoubleVar(value=0.0)
        ttk.Entry(frame_ascent, textvariable=self.start_pitch).grid(row=2, column=1, sticky='w')
        
        ttk.Label(frame_ascent, text="Yaw (deg):").grid(row=3, column=0, sticky='w')
        self.start_yaw = tk.DoubleVar(value=0.0)
        ttk.Entry(frame_ascent, textvariable=self.start_yaw).grid(row=3, column=1, sticky='w')
        
        ttk.Label(frame_ascent, text="Roll (deg):").grid(row=4, column=0, sticky='w')
        self.start_roll = tk.DoubleVar(value=0.0)
        ttk.Entry(frame_ascent, textvariable=self.start_roll).grid(row=4, column=1, sticky='w')

        # Toggle to show plots after run
        self.show_plots = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Show plots after run", variable=self.show_plots).grid(row=9, column=0, columnspan=2, sticky='w', pady=(5,0))
        
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

        # Fault Injection Settings removed from Simulation tab; use the 'Fault Injection' tab instead

    def create_faults_frame(self, tab):
        """Legacy random in-flight faults removed.
        Use the dedicated 'Fault Injection' tab for defining precise, ordered fault scenarios.
        """
        pass  # removed legacy UI to avoid duplication with Fault Injection tab
        self.drag_change_prob = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.drag_change_prob).grid(row=3, column=1)
        
        ttk.Label(frame, text="Drag Multiplier (x):").grid(row=4, column=0, sticky='w')
        self.drag_multiplier_fault = tk.DoubleVar(value=1.5)
        ttk.Entry(frame, textvariable=self.drag_multiplier_fault).grid(row=4, column=1)
        
        # Thrust Anomaly
        ttk.Label(frame, text="Thrust Anomaly Prob (0-1):").grid(row=5, column=0, sticky='w')
        self.thrust_anomaly_prob = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.thrust_anomaly_prob).grid(row=5, column=1)
        
        self.thrust_multiplier_fault = tk.DoubleVar(value=0.8)
        ttk.Entry(frame, textvariable=self.thrust_multiplier_fault).grid(row=6, column=1)
        
        # ML Adaptive Toggle
        self.use_ml_adaptive = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Use Adaptive ML Guidance (requires model)", variable=self.use_ml_adaptive).grid(row=7, column=0, columnspan=2, sticky='w', pady=(10,0))
    
    def create_fault_injection_tab(self):
        """Create fault injection configuration tab with list-based ordering"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Fault Injection")
        
        # Initialize fault storage
        self.fault_list = []  # List of fault dicts
        self._group_colors = {}  # Map concurrent_group id -> color hex
        self.fault_id_counter = 1
        
        # Top frame for enable/disable and ML model selection
        top_frame = ttk.LabelFrame(tab, text="Fault Injection Settings", padding=10)
        top_frame.pack(fill='x', padx=5, pady=5)
        
        self.faults_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_frame, text="Enable Fault Injection", variable=self.faults_enabled).grid(row=0, column=0, columnspan=2, sticky='w')
        
        # ML Flight Computer section
        ml_frame = ttk.LabelFrame(top_frame, text="ML Flight Computer", padding=5)
        ml_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10,0))
        
        self.ml_fc_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(ml_frame, text="Enable ML Flight Computer", variable=self.ml_fc_enabled).grid(row=0, column=0, columnspan=3, sticky='w')
        
        ttk.Label(ml_frame, text="Model Path:").grid(row=1, column=0, sticky='w', pady=2)
        self.ml_fc_model_path = tk.StringVar(value="ML/run_2/correction_model.keras")
        ttk.Entry(ml_frame, textvariable=self.ml_fc_model_path, width=30).grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(ml_frame, text="Browse", command=self._browse_ml_model).grid(row=1, column=2)
        
        ttk.Label(ml_frame, text="Scaler Path:").grid(row=2, column=0, sticky='w', pady=2)
        self.ml_fc_scaler_path = tk.StringVar(value="ML/run_2/correction_scalers.pkl")
        ttk.Entry(ml_frame, textvariable=self.ml_fc_scaler_path, width=30).grid(row=2, column=1, sticky='ew', padx=5)
        ttk.Button(ml_frame, text="Browse", command=self._browse_ml_scaler).grid(row=2, column=2)
        
        ttk.Label(ml_frame, text="Update Interval (s):").grid(row=3, column=0, sticky='w', pady=2)
        self.ml_fc_update_interval = tk.DoubleVar(value=0.5)
        ttk.Entry(ml_frame, textvariable=self.ml_fc_update_interval, width=15).grid(row=3, column=1, sticky='w', padx=5)
        
        ml_frame.columnconfigure(1, weight=1)
        
        # Fault list frame
        list_frame = ttk.LabelFrame(tab, text="Fault Sequence", padding=10)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side='right', fill='y')
        
        # Multi-select enabled so users can group or operate on multiple faults
        self.fault_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10, selectmode=tk.EXTENDED, exportselection=False)
        self.fault_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.fault_listbox.yview)
        
        # Bind selection
        self.fault_listbox.bind('<<ListboxSelect>>', self._on_fault_select)
        # Double-click an item to select its entire group or open edit for single faults
        self.fault_listbox.bind('<Double-Button-1>', self._on_fault_double_click)
        
        # Control buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(side='right', fill='y', padx=(10,0))
        
        ttk.Button(btn_frame, text="Add Fault", command=self._add_fault_dialog).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Edit Fault", command=self._edit_fault_dialog).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Remove Fault", command=self._remove_fault).pack(fill='x', pady=2)
        ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(btn_frame, text="Move Up", command=self._move_fault_up).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Move Down", command=self._move_fault_down).pack(fill='x', pady=2)
        ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(btn_frame, text="Group as Concurrent", command=self._group_concurrent).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Select Group", command=self._select_group).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all_faults).pack(fill='x', pady=2)
    
    def _browse_ml_model(self):
        """Browse for ML model file"""
        filename = filedialog.askopenfilename(
            title="Select ML Model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("TFLite Model", "*.tflite"), ("All Files", "*.*")]
        )
        if filename:
            self.ml_fc_model_path.set(filename)
    
    def _browse_ml_scaler(self):
        """Browse for scaler file"""
        filename = filedialog.askopenfilename(
            title="Select Scaler File",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")]
        )
        if filename:
            self.ml_fc_scaler_path.set(filename)
    
    def _add_fault_dialog(self):
        """Open dialog to add a new fault"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Fault")
        dialog.geometry("450x500")
        
        # Fault type
        ttk.Label(dialog, text="Fault Type:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        fault_type = tk.StringVar(value="mass_loss")
        fault_types = ["mass_loss", "thrust_var", "drag_change", "wind_gust"]
        ttk.Combobox(dialog, textvariable=fault_type, values=fault_types, state='readonly').grid(row=0, column=1, sticky='ew', padx=10)
        
        # Trigger mode
        ttk.Label(dialog, text="Trigger Mode:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        trigger_mode = tk.StringVar(value="absolute_time")
        trigger_modes = ["absolute_time", "time_since_apogee", "altitude_threshold", "manual"]
        ttk.Combobox(dialog, textvariable=trigger_mode, values=trigger_modes, state='readonly').grid(row=1, column=1, sticky='ew', padx=10)
        
        # Trigger value
        ttk.Label(dialog, text="Trigger Value (s or m):").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        trigger_value = tk.DoubleVar(value=5.0)
        ttk.Entry(dialog, textvariable=trigger_value).grid(row=2, column=1, sticky='ew', padx=10)
        
        # Magnitude
        ttk.Label(dialog, text="Magnitude:").grid(row=3, column=0, sticky='w', padx=10, pady=5)
        magnitude = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=magnitude).grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(dialog, text="(mass_loss: kg to lose (negative),\nthrust/drag: multiplier,\nwind_gust: m/s)", 
                 font=("Arial", 8), foreground="gray").grid(row=4, column=0, columnspan=2, sticky='w', padx=10)
        
        # Duration
        ttk.Label(dialog, text="Duration (s, 0=permanent):").grid(row=5, column=0, sticky='w', padx=10, pady=5)
        duration = tk.DoubleVar(value=0.0)
        ttk.Entry(dialog, textvariable=duration).grid(row=5, column=1, sticky='ew', padx=10)
        
        # Target
        ttk.Label(dialog, text="Target:").grid(row=6, column=0, sticky='w', padx=10, pady=5)
        target = tk.StringVar(value="simulated_state")
        targets = ["simulated_state", "sensor_only"]
        ttk.Combobox(dialog, textvariable=target, values=targets, state='readonly').grid(row=6, column=1, sticky='ew', padx=10)
        
        # Randomize magnitude
        randomize = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="Randomize Magnitude", variable=randomize).grid(row=7, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        
        # Magnitude range
        ttk.Label(dialog, text="Magnitude Range (min):").grid(row=8, column=0, sticky='w', padx=10, pady=2)
        mag_min = tk.DoubleVar(value=0.8)
        ttk.Entry(dialog, textvariable=mag_min).grid(row=8, column=1, sticky='ew', padx=10)
        
        ttk.Label(dialog, text="Magnitude Range (max):").grid(row=9, column=0, sticky='w', padx=10, pady=2)
        mag_max = tk.DoubleVar(value=1.2)
        ttk.Entry(dialog, textvariable=mag_max).grid(row=9, column=1, sticky='ew', padx=10)
        
        # Probability
        ttk.Label(dialog, text="Probability (0-1):").grid(row=10, column=0, sticky='w', padx=10, pady=5)
        probability = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=probability).grid(row=10, column=1, sticky='ew', padx=10)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=11, column=0, columnspan=2, pady=20)
        
        def add_fault():
            fault = {
                'fault_type': fault_type.get(),
                'trigger_mode': trigger_mode.get(),
                'trigger_value': trigger_value.get(),
                'magnitude': magnitude.get(),
                'duration': duration.get(),
                'target': target.get(),
                'randomize_magnitude': randomize.get(),
                'magnitude_range': [mag_min.get(), mag_max.get()],
                'probability': probability.get(),
                'fault_id': self.fault_id_counter,
                'concurrent_group': None
            }
            self.fault_list.append(fault)
            self.fault_id_counter += 1
            self._refresh_fault_listbox()
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Add", command=add_fault).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
        
        dialog.columnconfigure(1, weight=1)
    
    def _edit_fault_dialog(self):
        """Edit selected fault"""
        selection = self.fault_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a fault to edit")
            return
        if len(selection) != 1:
            messagebox.showwarning("Selection Error", "Please select exactly one fault to edit")
            return
        
        idx = selection[0]
        fault = self.fault_list[idx]
        
        # Similar dialog to add, but pre-populated
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Fault")
        dialog.geometry("450x500")
        
        # Fault type
        ttk.Label(dialog, text="Fault Type:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        fault_type = tk.StringVar(value=fault['fault_type'])
        fault_types = ["mass_loss", "thrust_var", "drag_change", "wind_gust"]
        ttk.Combobox(dialog, textvariable=fault_type, values=fault_types, state='readonly').grid(row=0, column=1, sticky='ew', padx=10)
        
        # Trigger mode
        ttk.Label(dialog, text="Trigger Mode:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        trigger_mode = tk.StringVar(value=fault['trigger_mode'])
        trigger_modes = ["absolute_time", "time_since_apogee", "altitude_threshold", "manual"]
        ttk.Combobox(dialog, textvariable=trigger_mode, values=trigger_modes, state='readonly').grid(row=1, column=1, sticky='ew', padx=10)
        
        # Trigger value
        ttk.Label(dialog, text="Trigger Value (s or m):").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        trigger_value = tk.DoubleVar(value=fault['trigger_value'])
        ttk.Entry(dialog, textvariable=trigger_value).grid(row=2, column=1, sticky='ew', padx=10)
        
        # Magnitude
        ttk.Label(dialog, text="Magnitude:").grid(row=3, column=0, sticky='w', padx=10, pady=5)
        magnitude = tk.DoubleVar(value=fault['magnitude'])
        ttk.Entry(dialog, textvariable=magnitude).grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(dialog, text="(mass_loss: kg to lose (negative),\nthrust/drag: multiplier,\nwind_gust: m/s)", 
                 font=("Arial", 8), foreground="gray").grid(row=4, column=0, columnspan=2, sticky='w', padx=10)
        
        # Duration
        ttk.Label(dialog, text="Duration (s, 0=permanent):").grid(row=5, column=0, sticky='w', padx=10, pady=5)
        duration = tk.DoubleVar(value=fault['duration'])
        ttk.Entry(dialog, textvariable=duration).grid(row=5, column=1, sticky='ew', padx=10)
        
        # Target
        ttk.Label(dialog, text="Target:").grid(row=6, column=0, sticky='w', padx=10, pady=5)
        target = tk.StringVar(value=fault['target'])
        targets = ["simulated_state", "sensor_only"]
        ttk.Combobox(dialog, textvariable=target, values=targets, state='readonly').grid(row=6, column=1, sticky='ew', padx=10)
        
        # Randomize magnitude
        randomize = tk.BooleanVar(value=fault.get('randomize_magnitude', False))
        ttk.Checkbutton(dialog, text="Randomize Magnitude", variable=randomize).grid(row=7, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        
        # Magnitude range
        ttk.Label(dialog, text="Magnitude Range (min):").grid(row=8, column=0, sticky='w', padx=10, pady=2)
        mag_min = tk.DoubleVar(value=fault.get('magnitude_range', [0.8, 1.2])[0])
        ttk.Entry(dialog, textvariable=mag_min).grid(row=8, column=1, sticky='ew', padx=10)
        
        ttk.Label(dialog, text="Magnitude Range (max):").grid(row=9, column=0, sticky='w', padx=10, pady=2)
        mag_max = tk.DoubleVar(value=fault.get('magnitude_range', [0.8, 1.2])[1])
        ttk.Entry(dialog, textvariable=mag_max).grid(row=9, column=1, sticky='ew', padx=10)
        
        # Probability
        ttk.Label(dialog, text="Probability (0-1):").grid(row=10, column=0, sticky='w', padx=10, pady=5)
        probability = tk.DoubleVar(value=fault.get('probability', 1.0))
        ttk.Entry(dialog, textvariable=probability).grid(row=10, column=1, sticky='ew', padx=10)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=11, column=0, columnspan=2, pady=20)
        
        def save_fault():
            self.fault_list[idx] = {
                'fault_type': fault_type.get(),
                'trigger_mode': trigger_mode.get(),
                'trigger_value': trigger_value.get(),
                'magnitude': magnitude.get(),
                'duration': duration.get(),
                'target': target.get(),
                'randomize_magnitude': randomize.get(),
                'magnitude_range': [mag_min.get(), mag_max.get()],
                'probability': probability.get(),
                'fault_id': fault['fault_id'],
                'concurrent_group': fault.get('concurrent_group')
            }
            self._refresh_fault_listbox()
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Save", command=save_fault).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
        
        dialog.columnconfigure(1, weight=1)
    
    def _remove_fault(self):
        """Remove selected fault(s)"""
        selection = self.fault_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select one or more faults to remove")
            return
        
        # Delete from highest index to lowest to avoid reindexing issues
        for idx in sorted(selection, reverse=True):
            del self.fault_list[idx]
        self._refresh_fault_listbox()
    
    def _move_fault_up(self):
        """Move selected fault or contiguous block up in the list"""
        selection = self.fault_listbox.curselection()
        if not selection:
            return
        indices = sorted(selection)
        if indices[0] == 0:
            return  # Can't move past top
        # Require contiguous block for multi-move
        if len(indices) > 1 and (indices[-1] - indices[0] + 1) != len(indices):
            messagebox.showwarning("Selection Error", "Please select a contiguous block to move")
            return
        # Extract block
        block = [self.fault_list[i] for i in indices]
        # Remove existing entries from highest index to lowest
        for i in reversed(indices):
            del self.fault_list[i]
        insert_at = indices[0] - 1
        for j, item in enumerate(block):
            self.fault_list.insert(insert_at + j, item)
        self._refresh_fault_listbox()
        # Restore selection on moved block
        for j in range(len(block)):
            self.fault_listbox.selection_set(insert_at + j)
    
    def _move_fault_down(self):
        """Move selected fault or contiguous block down in the list"""
        selection = self.fault_listbox.curselection()
        if not selection:
            return
        indices = sorted(selection)
        if indices[-1] >= len(self.fault_list) - 1:
            return  # Can't move past bottom
        # Require contiguous block for multi-move
        if len(indices) > 1 and (indices[-1] - indices[0] + 1) != len(indices):
            messagebox.showwarning("Selection Error", "Please select a contiguous block to move")
            return
        block = [self.fault_list[i] for i in indices]
        for i in reversed(indices):
            del self.fault_list[i]
        insert_at = indices[0] + 1
        for j, item in enumerate(block):
            self.fault_list.insert(insert_at + j, item)
        self._refresh_fault_listbox()
        # Restore selection on moved block
        for j in range(len(block)):
            self.fault_listbox.selection_set(insert_at + j)
    
    def _group_concurrent(self):
        """Group selected faults as concurrent and assign a visual color"""
        selection = self.fault_listbox.curselection()
        if len(selection) < 2:
            messagebox.showwarning("Selection Required", "Please select 2 or more faults to group as concurrent")
            return
        
        # Assign same concurrent group ID to selected faults
        group_id = max([f.get('concurrent_group', 0) for f in self.fault_list], default=0) + 1
        
        for idx in selection:
            self.fault_list[idx]['concurrent_group'] = group_id
        
        # Assign a color for visual grouping if new
        palette = ['#FFD700','#ADFF2F','#87CEFA','#FFA07A','#DA70D6','#98FB98','#FFB6C1','#87CEEB']
        if group_id not in self._group_colors:
            self._group_colors[group_id] = palette[(group_id - 1) % len(palette)]
        
        self._refresh_fault_listbox()
        messagebox.showinfo("Grouped", f"Selected faults grouped as concurrent (Group {group_id})")
    
    def _clear_all_faults(self):
        """Clear all faults"""
        if messagebox.askyesno("Confirm", "Clear all faults?"):
            self.fault_list = []
            self._refresh_fault_listbox()
    
    def _on_fault_select(self, event):
        """Handle fault selection (placeholder)"""
        # Currently no special handling on selection; reserved for future UI updates
        return
    
    def _select_group(self):
        """Select all faults that belong to the same concurrent group as the current selection"""
        selection = self.fault_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a fault inside a group to select the entire group")
            return
        idx = selection[0]
        group_id = self.fault_list[idx].get('concurrent_group')
        if not group_id:
            messagebox.showwarning("Not in Group", "Selected fault is not part of a concurrent group.")
            return
        self._select_group_by_id(group_id)
    
    def _select_group_by_id(self, group_id):
        indices = [i for i, f in enumerate(self.fault_list) if f.get('concurrent_group') == group_id]
        if not indices:
            messagebox.showwarning("Group Not Found", f"No faults found for Group {group_id}")
            return
        self.fault_listbox.selection_clear(0, tk.END)
        for i in indices:
            self.fault_listbox.selection_set(i)
    
    def _on_fault_double_click(self, event):
        """Double-click to select whole group or edit single fault"""
        idx = self.fault_listbox.nearest(event.y)
        if idx is None or idx >= len(self.fault_list):
            return
        group_id = self.fault_list[idx].get('concurrent_group')
        if group_id:
            self._select_group_by_id(group_id)
        else:
            # Single fault - open edit dialog
            self.fault_listbox.selection_clear(0, tk.END)
            self.fault_listbox.selection_set(idx)
            self._edit_fault_dialog()
    
    def _refresh_fault_listbox(self):
        """Refresh the fault listbox display"""
        self.fault_listbox.delete(0, tk.END)
        
        for i, fault in enumerate(self.fault_list):
            # Format display string
            concurrent = f" [Group {fault['concurrent_group']}]" if fault.get('concurrent_group') else ""
            display = f"{i+1}. {fault['fault_type']} @ {fault['trigger_mode']}={fault['trigger_value']:.1f}{concurrent}"
            self.fault_listbox.insert(tk.END, display)
            # Color grouped faults for visual clarity
            group_id = fault.get('concurrent_group')
            if group_id:
                color = self._group_colors.get(group_id, '#f0f0f0')
                try:
                    self.fault_listbox.itemconfig(i, bg=color)
                except Exception:
                    # Some Tk versions may not support itemconfig/bg; ignore gracefully
                    pass
    
    def _get_fault_config(self):
        """Build fault configuration from GUI fault list"""
        from faults import FaultGroup
        
        if not self.faults_enabled.get() or not self.fault_list:
            return {
                'enabled': False,
                'fault_groups': []
            }
        
        # Group faults by concurrent_group
        groups = {}
        for fault in self.fault_list:
            group_id = fault.get('concurrent_group', None)
            if group_id is None:
                # Individual fault (sequential)
                group_id = f"solo_{fault['fault_id']}"
                concurrent = False
            else:
                concurrent = True
            
            if group_id not in groups:
                groups[group_id] = {'faults': [], 'concurrent': concurrent}
            groups[group_id]['faults'].append(fault)
        
        # Build fault groups
        fault_groups = []
        for group_id, group_data in groups.items():
            fault_groups.append({
                'faults': group_data['faults'],
                'concurrent': group_data['concurrent'],
                'group_id': hash(group_id) % 10000  # Simple numeric ID
            })
        
        return {
            'enabled': True,
            'fault_groups': fault_groups
        }
    
    def _get_ml_flight_computer_config(self):
        """Build ML flight computer configuration from GUI"""
        return {
            'enabled': self.ml_fc_enabled.get(),
            'model_path': self.ml_fc_model_path.get() if self.ml_fc_enabled.get() else None,
            'scaler_path': self.ml_fc_scaler_path.get() if self.ml_fc_enabled.get() else None,
            'update_interval': self.ml_fc_update_interval.get()
        }
    
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

        ttk.Button(button_frame, text="Check Feasibility", 
                  command=self.check_feasibility).pack(side='left', padx=5)
        
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

    def check_feasibility(self):
        """Check if landing is physically possible"""
        try:
            # Get configurations
            rocket_config, environment_config, simulation_config, fault_config, ml_config = self.get_configs()
            
            # Create simulation (lightweight)
            sim = SuicideBurnSimulation(rocket_config, environment_config, simulation_config)
            
            initial_velocity = self.initial_velocity.get()
            initial_altitude = self.initial_altitude.get()
            
            is_possible, r = sim.check_feasibility(initial_velocity, initial_altitude)
            
            msg = f"Feasibility: {'POSSIBLE' if is_possible else 'IMPOSSIBLE'}\n\n"
            msg += f"Initial Impact Speed (Unpowered): {r['v_impact_unpowered']:.1f} m/s\n"
            msg += f"Delta-V Capacity (Gravity Losses Included): {r['dv_capacity']:.1f} m/s\n"
            msg += f"  - Gross Delta-V: {r['dv_gross']:.1f} m/s\n"
            msg += f"  - Gravity Loss: {r['dv_gravity_loss']:.1f} m/s\n\n"
            msg += f"Margin (Capacity - Impact): {r['margin']:.1f} m/s\n\n"
            msg += f"Max TWR (at burnout): {r['max_twr']:.2f}"
            
            if is_possible:
                messagebox.showinfo("Feasibility Report", msg)
            else:
                messagebox.showwarning("Feasibility Report", msg)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_plots_tab(self):
        """Create Plots tab to embed interactive matplotlib canvases"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Plots")
        self.plots_tab = tab

        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Clear Plots", command=self._clear_plots).pack(side='left')
        ttk.Button(control_frame, text="View in HexaVisual", command=self.launch_hexavisual_with_data).pack(side='left', padx=5)
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
            'ascent_motor_casing_mass': self.ascent_motor_casing_mass.get(),
            'tvc_mode': self.tvc_mode.get(),
            'tvc_drift_gain': self.tvc_drift_gain.get()
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
            'initial_altitude': self.initial_altitude.get(),
            'initial_velocity': self.initial_velocity.get(),
        }
        
        sim_config = {
            'num_monte_carlo': self.num_monte_carlo.get(),
            'altitude_search_range': self.altitude_search_range.get(),
            'altitude_step': self.altitude_step.get(),
            'altimeter_error': self.altimeter_error.get(),
            'velocity_sensor_error': self.velocity_sensor_error.get(),
            'ignition_percent_offset': self.ignition_percent_offset.get() / 100.0,
            'ignition_hard_offset': self.ignition_hard_offset.get(),
            'show_plots': self.show_plots.get(),
            'simulate_ascent': self.simulate_ascent.get(),
            'optimization_mode': self.opt_mode.get(),
            'opt_max_iterations': self.opt_max_iterations.get(),
            'opt_samples_per_step': self.opt_samples_per_step.get(),
            'opt_target_step': self.opt_target_step.get(),
            'opt_samples_per_step': self.opt_samples_per_step.get(),
            'opt_target_step': self.opt_target_step.get(),
            'use_ml_adaptive': getattr(self, 'use_ml_adaptive', tk.BooleanVar(value=False)).get(),
            'ml_model_path': getattr(self, 'ml_model_path_var', tk.StringVar(value='ignition_model_improved.keras')).get(),
            'ml_scaler_path': getattr(self, 'ml_scaler_path_var', tk.StringVar(value='scaler_improved.pkl')).get(),
        }

        # Legacy simple random faults removed — use the 'Fault Injection' tab. The
        # detailed fault configuration is returned separately as `fault_config` below.
        
        # Map GUI orientation fields to correct config based on mode
        if self.simulate_ascent.get():
            sim_config['ascent_initial_pitch'] = self.start_pitch.get()
            sim_config['ascent_initial_yaw'] = self.start_yaw.get()
            sim_config['ascent_initial_roll'] = self.start_roll.get()
            # Zero out descent ones or leave default? It doesn't matter, logic uses ascent->apogee->descent
        else:
            sim_config['descent_initial_pitch'] = self.start_pitch.get()
            sim_config['descent_initial_yaw'] = self.start_yaw.get()
            sim_config['descent_initial_roll'] = self.start_roll.get()
        
        # Add fault configuration from new tab
        fault_config = self._get_fault_config() if hasattr(self, '_get_fault_config') else {'enabled': False, 'fault_groups': []}
        
        # Add ML flight computer configuration from new tab
        ml_config = self._get_ml_flight_computer_config() if hasattr(self, '_get_ml_flight_computer_config') else {'enabled': False}
        
        return rocket_config, environment_config, sim_config, fault_config, ml_config
        
    def update_orientation_labels(self):
        """Update labels based on ascent checkbox"""
        if self.simulate_ascent.get():
            self.lbl_orientation.config(text="Launch Pad Orientation (deg):")
        else:
            self.lbl_orientation.config(text="Burn Start Orientation (deg):")

    def _update_optimization_mode_visibility(self, event=None):
        """Show/hide fields based on optimization mode"""
        mode = self.opt_mode.get()
        if mode == "adaptive":
            # Show adaptive, hide grid step
            self.lbl_altitude_step.grid_remove()
            self.ent_altitude_step.grid_remove()
            
            self.lbl_max_iters.grid()
            self.ent_max_iters.grid()
            self.lbl_samples.grid()
            self.ent_samples.grid()
            self.lbl_target_step.grid()
            self.ent_target_step.grid()
        else:
            # Show grid step, hide adaptive
            self.lbl_altitude_step.grid()
            self.ent_altitude_step.grid()
            
            self.lbl_max_iters.grid_remove()
            self.ent_max_iters.grid_remove()
            self.lbl_samples.grid_remove()
            self.ent_samples.grid_remove()
            self.lbl_target_step.grid_remove()
            self.ent_target_step.grid_remove()

    def _update_tvc_mode_visibility(self, event=None):
        mode = self.tvc_mode.get()
        if mode == 'velocity':
            self.lbl_drift_gain.grid()
            self.ent_drift_gain.grid()
        else:
            self.lbl_drift_gain.grid_remove()
            self.ent_drift_gain.grid_remove()
    
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
            rocket_config, environment_config, simulation_config, fault_config, ml_config = self.get_configs()
            
            # Create simulation (use EnhancedSimulation if faults or ML enabled)
            from simulation_wrapper import EnhancedSimulation
            if fault_config.get('enabled') or ml_config.get('enabled'):
                self.log("Using enhanced simulation with fault injection and/or ML flight computer...")
                self.simulation = EnhancedSimulation(
                    rocket_config, environment_config, simulation_config,
                    fault_config, ml_config
                ).simulation  # Access wrapped simulation for compatibility
            else:
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
            best_history = None

            # Create results folder up-front so per-trial outputs can be written during optimization
            results_folder, ts = self._make_results_subfolder('optimization')

            # Save the configuration for the entire optimization run
            self._save_config_to_folder(results_folder)

            # Run optimization (pass GUI progress callback), save per-trial CSVs and PNGs
            if simulation_config.get('optimization_mode', 'grid') == 'adaptive':
                optimal_altitude, convergence_history, best_history = self.simulation.optimize_ignition_altitude_adaptive(
                    initial_state,
                    num_monte_carlo=simulation_config['num_monte_carlo'],
                    altitude_search_range=simulation_config['altitude_search_range'],
                    max_iterations=simulation_config.get('opt_max_iterations', 3),
                    samples_per_step=simulation_config.get('opt_samples_per_step', 10),
                    target_step=simulation_config.get('opt_target_step', 0.01),
                    progress_callback=self._progress_callback,
                    save_each_trial=True,
                    results_folder=results_folder,
                    save_plots_per_trial=True
                )
                # For summary plotting/saving, we create a dict with at least the best result
                success_rates = {optimal_altitude: 1.0 if (best_history and best_history.get('success')) else 0.0}
            else:
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
                best_history = best_history # For clarity
            
            self.log(f"\nOptimal ignition altitude: {optimal_altitude:.2f} m")
            
            # Save summary results into the same folder
            self.save_results(success_rates, best_history, folder=results_folder)
            
            # Generate plots (overall) into the same folder
            self.generate_plots(success_rates, best_history, results_folder)
            
            # Track the latest best trajectory for HexaVisual
            self.last_result_csv = os.path.join(results_folder, 'trajectory.csv')
            
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
            self.log("Note: Your HexaKinetic window will be unresponsive for the duration of the simulation.")
            self.log("Running single simulation...")
            self.status_bar.config(text="Running single simulation")

            
            # Get configurations
            rocket_config, environment_config, simulation_config, fault_config, ml_config = self.get_configs()
            
            # Create simulation (use EnhancedSimulation if faults or ML enabled)
            from simulation_wrapper import EnhancedSimulation
            if fault_config.get('enabled') or ml_config.get('enabled'):
                self.log("Using enhanced simulation with fault injection and/or ML flight computer...")
                enhanced_sim = EnhancedSimulation(
                    rocket_config, environment_config, simulation_config,
                    fault_config, ml_config
                )
                self.simulation = enhanced_sim.simulation  # For compatibility
                use_enhanced = True
            else:
                self.simulation = SuicideBurnSimulation(
                    rocket_config, environment_config, simulation_config
                )
                enhanced_sim = None
                use_enhanced = False
            
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
            
            # Run simulation
            # Passing ignition_altitude=None allows the simulation to calculate it 
            # dynamically based on apogee if simulate_ascent is True.
            if use_enhanced:
                assert enhanced_sim is not None, "Enhanced simulation should be initialized"
                history = enhanced_sim.run_simulation(initial_state, None)
                success = history.get('success', False)
                final_state = None  # Not needed for enhanced sim
            else:
                success, final_state, history = self.simulation.run_simulation(
                    initial_state, None
                )
            
            # Ignition altitude used (can be retrieved from history)
            actual_ignition_altitude = history.get('ignition_altitude', 0.0) if isinstance(history, dict) else 0.0
            self.log(f"Calculated ignition altitude: {actual_ignition_altitude:.2f} m")
            
            self.log(f"Success: {success}")
            self.log(f"Final altitude: {history['final_altitude']:.2f} m")
            self.log(f"Final velocity: {history['final_velocity']:.2f} m/s")
            
            # Save and plot
            results_folder = self.save_single_result(history)
            self._save_config_to_folder(results_folder)
            self.last_result_csv = os.path.join(results_folder, 'single_run.csv')
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

    def _save_config_to_folder(self, folder):
        """Save current GUI configuration to a JSON file in the specified folder"""
        try:
            import json
            rocket_config, env_config, sim_config, fault_config, ml_config = self.get_configs()
            full_config = {
                "rocket": rocket_config,
                "environment": env_config,
                "simulation": sim_config,
                "faults": fault_config,
                "ml_flight_computer": ml_config
            }
            config_path = os.path.join(folder, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            self.log(f"Saved configuration to {config_path}")
        except Exception as e:
            self.log(f"Warning: Could not save configuration to folder: {e}")

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

    def launch_hexavisual_with_data(self):
        """Launch HexaVisual with the current run's data"""
        if not self.last_result_csv:
            messagebox.showwarning("Warning", "No simulation result available. Run a simulation first.")
            return
        
        import subprocess
        import sys
        try:
            # Launch advanced_visualizer.py with the CSV path as an argument
            subprocess.Popen([sys.executable, "advanced_visualizer.py", self.last_result_csv])
            self.log(f"Launched HexaVisual with: {self.last_result_csv}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch HexaVisual: {e}")

    def save_config(self):
        """Save configuration to JSON file (compatible with CLI)"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                rocket_config, env_config, sim_config, fault_config, ml_config = self.get_configs()
                
                # Combine into one interchangeable config
                full_config = {
                    "rocket": rocket_config,
                    "environment": env_config,
                    "simulation": sim_config,
                    "faults": fault_config,
                    "ml_flight_computer": ml_config
                }
                
                with open(filename, 'w') as f:
                    json.dump(full_config, f, indent=2)
                
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")
            
    def load_config(self):
        """Load configuration from JSON file (compatible with CLI)"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                rocket = config.get('rocket', {})
                env = config.get('environment', {})
                sim = config.get('simulation', {})
                
                # Update Rocket parameters
                if 'dry_mass' in rocket: self.dry_mass.set(rocket['dry_mass'])
                if 'propellant_mass' in rocket: self.propellant_mass.set(rocket['propellant_mass'])
                if 'ascent_motor_casing_mass' in rocket: self.ascent_motor_casing_mass.set(rocket['ascent_motor_casing_mass'])
                if 'length' in rocket: self.length.set(rocket['length'])
                if 'diameter' in rocket: self.diameter.set(rocket['diameter'])
                if 'use_dynamic_inertia' in rocket: self.use_dynamic_inertia.set(rocket['use_dynamic_inertia'])
                if 'burn_time' in rocket: self.burn_time.set(rocket['burn_time'])
                if 'tvc_max_angle' in rocket: self.tvc_max_angle.set(rocket['tvc_max_angle'])
                if 'tvc_response_time' in rocket: self.tvc_response_time.set(rocket['tvc_response_time'])
                if 'tvc_kp_pitch' in rocket: self.tvc_kp_pitch.set(rocket['tvc_kp_pitch'])
                if 'tvc_ki_pitch' in rocket: self.tvc_ki_pitch.set(rocket['tvc_ki_pitch'])
                if 'tvc_kd_pitch' in rocket: self.tvc_kd_pitch.set(rocket['tvc_kd_pitch'])
                if 'tvc_kp_yaw' in rocket: self.tvc_kp_yaw.set(rocket['tvc_kp_yaw'])
                if 'tvc_ki_yaw' in rocket: self.tvc_ki_yaw.set(rocket['tvc_ki_yaw'])
                if 'tvc_kd_yaw' in rocket: self.tvc_kd_yaw.set(rocket['tvc_kd_yaw'])
                if 'thrust_variation' in rocket: self.thrust_variation.set(rocket['thrust_variation'])
                if 'tvc_response_variation' in rocket: self.tvc_response_variation.set(rocket['tvc_response_variation'])
                if 'mass_variation' in rocket: self.mass_variation.set(rocket['mass_variation'])
                if 'tvc_mode' in rocket: self.tvc_mode.set(rocket['tvc_mode'])
                if 'tvc_drift_gain' in rocket: self.tvc_drift_gain.set(rocket['tvc_drift_gain'])
                
                # Update TVC visibility
                self._update_tvc_mode_visibility()
                
                if 'thrust_curve' in rocket:
                    self.thrust_curve.delete('1.0', 'end')
                    curve_str = "\n".join([f"{t},{T}" for t, T in rocket['thrust_curve']])
                    self.thrust_curve.insert('1.0', curve_str)
                
                # Update Environment parameters
                if 'gravity' in env: self.gravity.set(env['gravity'])
                if 'air_density' in env: self.air_density.set(env['air_density'])
                if 'temperature' in env: self.temperature.set(env['temperature'])
                if 'drag_coefficient' in env: self.drag_coefficient.set(env['drag_coefficient'])
                if 'wind_model' in env: self.wind_model.set(env['wind_model'])
                if 'wind_speed' in env: self.wind_speed.set(env['wind_speed'])
                if 'wind_direction' in env: self.wind_direction.set(env['wind_direction'])
                if 'drag_variation' in env: self.drag_variation.set(env['drag_variation'])
                if 'air_density_variation' in env: self.air_density_variation.set(env['air_density_variation'])
                if 'initial_altitude' in env: self.initial_altitude.set(env['initial_altitude'])
                if 'initial_velocity' in env: self.initial_velocity.set(env['initial_velocity'])
                
                # Update Simulation parameters
                if 'num_monte_carlo' in sim: self.num_monte_carlo.set(sim['num_monte_carlo'])
                if 'altitude_search_range' in sim: self.altitude_search_range.set(sim['altitude_search_range'])
                if 'altitude_step' in sim: self.altitude_step.set(sim['altitude_step'])
                if 'altimeter_error' in sim: self.altimeter_error.set(sim['altimeter_error'])
                if 'velocity_sensor_error' in sim: self.velocity_sensor_error.set(sim['velocity_sensor_error'])
                if 'ignition_percent_offset' in sim: self.ignition_percent_offset.set(sim['ignition_percent_offset'] * 100.0)
                if 'ignition_hard_offset' in sim: self.ignition_hard_offset.set(sim['ignition_hard_offset'])
                if 'show_plots' in sim: self.show_plots.set(sim['show_plots'])
                if 'simulate_ascent' in sim: self.simulate_ascent.set(sim['simulate_ascent'])
                
                # Adaptive optimization parameters
                if 'optimization_mode' in sim: self.opt_mode.set(sim['optimization_mode'])
                if 'opt_max_iterations' in sim: self.opt_max_iterations.set(sim['opt_max_iterations'])
                if 'opt_samples_per_step' in sim: self.opt_samples_per_step.set(sim['opt_samples_per_step'])
                if 'opt_target_step' in sim: self.opt_target_step.set(sim['opt_target_step'])
                
                # Update ML paths if in config
                if 'ml_model_path' in sim: 
                    if not hasattr(self, 'ml_model_path_var'): self.ml_model_path_var = tk.StringVar()
                    self.ml_model_path_var.set(sim['ml_model_path'])
                if 'ml_scaler_path' in sim:
                    if not hasattr(self, 'ml_scaler_path_var'): self.ml_scaler_path_var = tk.StringVar()
                    self.ml_scaler_path_var.set(sim['ml_scaler_path'])

                # Update visibility
                self._update_optimization_mode_visibility()
                
                # Load faults configuration
                if 'faults' in config:
                    faults = config['faults']
                    if hasattr(self, 'faults_enabled'):
                        self.faults_enabled.set(faults.get('enabled', False))
                    if hasattr(self, 'fault_list') and 'fault_groups' in faults:
                        # Clear existing faults
                        self.fault_list = []
                        # Load faults from groups
                        for group in faults['fault_groups']:
                            for fault in group['faults']:
                                self.fault_list.append(fault)
                        if hasattr(self, '_refresh_fault_listbox'):
                            self._refresh_fault_listbox()
                
                # Load ML flight computer configuration
                if 'ml_flight_computer' in config:
                    ml = config['ml_flight_computer']
                    if hasattr(self, 'ml_fc_enabled'):
                        self.ml_fc_enabled.set(ml.get('enabled', False))
                    if hasattr(self, 'ml_fc_model_path') and 'model_path' in ml and ml['model_path']:
                        self.ml_fc_model_path.set(ml['model_path'])
                    if hasattr(self, 'ml_fc_scaler_path') and 'scaler_path' in ml and ml['scaler_path']:
                        self.ml_fc_scaler_path.set(ml['scaler_path'])
                    if hasattr(self, 'ml_fc_update_interval') and 'update_interval' in ml:
                        self.ml_fc_update_interval.set(ml['update_interval'])
                
                # Start angles
                if sim.get('simulate_ascent'):
                    if 'ascent_initial_pitch' in sim: self.start_pitch.set(sim['ascent_initial_pitch'])
                    if 'ascent_initial_yaw' in sim: self.start_yaw.set(sim['ascent_initial_yaw'])
                    if 'ascent_initial_roll' in sim: self.start_roll.set(sim['ascent_initial_roll'])
                else:
                    if 'descent_initial_pitch' in sim: self.start_pitch.set(sim['descent_initial_pitch'])
                    if 'descent_initial_yaw' in sim: self.start_yaw.set(sim['descent_initial_yaw'])
                    if 'descent_initial_roll' in sim: self.start_roll.set(sim['descent_initial_roll'])
                
                self.update_orientation_labels()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                self.log(f"Loaded config: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")


def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

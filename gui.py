"""
GUI for Flight Dynamics Simulation
Tkinter interface with parameter configuration and plotting
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        
        # Output text
        frame = ttk.LabelFrame(tab, text="Output", padding=5)
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.output_text = tk.Text(frame, wrap='word', height=20)
        self.output_text.pack(fill='both', expand=True, side='left')
        
        scrollbar = ttk.Scrollbar(frame, command=self.output_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.output_text.config(yscrollcommand=scrollbar.set)
        
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
        }
        
        return rocket_config, environment_config, simulation_config
    
    def log(self, message):
        """Log message to output text widget"""
        self.output_text.insert('end', message + '\n')
        self.output_text.see('end')
        self.root.update()
        
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
            
            # Run optimization
            optimal_altitude, success_rates, best_history = self.simulation.optimize_ignition_altitude(
                initial_state,
                simulation_config['num_monte_carlo'],
                simulation_config['altitude_search_range'],
                simulation_config['altitude_step']
            )
            
            self.log(f"\nOptimal ignition altitude: {optimal_altitude:.2f} m")
            
            # Save results
            self.save_results(success_rates, best_history)
            
            # Generate plots
            self.generate_plots(success_rates, best_history)
            
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
            self.save_single_result(history)
            self.generate_single_plots(history)
            
            self.log("\nSimulation complete! Check the 'results' directory.")
            
            messagebox.showinfo("Complete", "Simulation complete! Check results directory.")
            
        except Exception as e:
            self.log(f"\nError: {str(e)}")
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            
    def save_results(self, success_rates, history):
        """Save optimization results to CSV"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save success rates
        filename = f'results/optimization_{timestamp}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Ignition Altitude (m)', 'Success Rate'])
            for altitude, rate in sorted(success_rates.items()):
                writer.writerow([altitude, rate])
        
        self.log(f"Saved optimization results to {filename}")
        
        # Save best trajectory
        if history:
            filename = f'results/trajectory_{timestamp}.csv'
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Mass'])
                for i in range(len(history['t'])):
                    writer.writerow([
                        history['t'][i],
                        history['x'][i],
                        history['y'][i],
                        history['z'][i],
                        history['vx'][i],
                        history['vy'][i],
                        history['vz'][i],
                        history['mass'][i]
                    ])
            
            self.log(f"Saved trajectory to {filename}")
            
    def save_single_result(self, history):
        """Save single simulation result to CSV"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f'results/single_run_{timestamp}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Mass'])
            for i in range(len(history['t'])):
                writer.writerow([
                    history['t'][i],
                    history['x'][i],
                    history['y'][i],
                    history['z'][i],
                    history['vx'][i],
                    history['vy'][i],
                    history['vz'][i],
                    history['mass'][i]
                ])
        
        self.log(f"Saved trajectory to {filename}")
        
    def generate_plots(self, success_rates, history):
        """Generate plots for optimization results"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Success rate vs altitude
        altitudes = sorted(success_rates.keys())
        rates = [success_rates[a] for a in altitudes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(altitudes, rates, 'b-', linewidth=2)
        plt.xlabel('Ignition Altitude (m)')
        plt.ylabel('Success Rate')
        plt.title('Success Rate vs Ignition Altitude')
        plt.grid(True)
        plt.savefig(f'results/success_rate_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        if history:
            self.generate_trajectory_plots(history, timestamp)
            
    def generate_single_plots(self, history):
        """Generate plots for single simulation"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.generate_trajectory_plots(history, timestamp)
        
    def generate_trajectory_plots(self, history, timestamp):
        """Generate trajectory plots"""
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
        
        plt.tight_layout()
        plt.savefig(f'results/trajectory_2d_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
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
        
        plt.savefig(f'results/trajectory_3d_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"Saved plots with timestamp {timestamp}")
        
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

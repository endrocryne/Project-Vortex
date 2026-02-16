"""
PlotVisual - Advanced Visualization Suite for Project Vortex
Specialized graph analysis tool for simulation results

Provides integrated plotting capabilities optimized for rocket landing analysis:
- Landing Velocity vs Fault Intensity scatterplot
- Success Rate parameter heatmaps  
- Landing Accuracy analysis (distance metrics and top-down spatial view)
- Monte Carlo distribution analysis
- Interactive filtering and parameter range selection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for embedding
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, MultipleLocator, AutoMinorLocator, NullFormatter
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json


class PlotVisualApp:
    """Main application for advanced visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PlotVisual - Vortex Analysis Suite")
        self.root.geometry("1400x900")
        
        # Data storage
        self.data = None
        self.filtered_data = None
        
        # Set Icon (try to use same icon as main GUI)
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'hexakinetic_icon.png')
            if os.path.exists(icon_path):
                self.icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, self.icon_img)
        except Exception:
            pass
        
        # State variables
        self.status_var = tk.StringVar(value="Ready - Load data to begin")
        self.yscale_var = tk.StringVar(value='Log')  # 'Log' or 'Linear' for velocity plot
        self.last_view = None
        self.current_figure = None
        self.current_canvas = None

        # Create menu bar
        self.create_menu()
        
        # Main layout: Left control panel, Right visualization area
        self.main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left control panel
        self.create_control_panel()
        
        # Right visualization area
        self.create_visualization_area()
        
        # Status bar
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data CSV", command=self.load_data)
        file_menu.add_command(label="Load Sample Data", command=self.load_sample_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Current Plot", command=self.export_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Velocity vs Intensity", command=self.plot_velocity_vs_intensity)
        view_menu.add_command(label="Success Rate Heatmap", command=self.plot_success_heatmap)
        view_menu.add_command(label="Landing Accuracy (Distance)", command=self.plot_landing_accuracy)
        view_menu.add_command(label="Landing Accuracy (Top-Down)", command=self.plot_landing_topdown)
        view_menu.add_command(label="Monte Carlo Distributions", command=self.plot_monte_carlo)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Apply Filters", command=self.apply_filters)
        edit_menu.add_separator()
        edit_menu.add_radiobutton(label="Y-Axis Log Scale", variable=self.yscale_var, value='Log')
        edit_menu.add_radiobutton(label="Y-Axis Linear Scale", variable=self.yscale_var, value='Linear')

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_control_panel(self):
        """Create left control panel"""
        control_frame = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(control_frame, weight=0)
        
        # Data info section
        info_frame = ttk.LabelFrame(control_frame, text="Data Info", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.data_info_label = tk.Label(info_frame, text="No data loaded", justify='left', anchor='w')
        self.data_info_label.pack(fill='x')
        
        ttk.Button(info_frame, text="Load Data CSV", command=self.load_data).pack(fill='x', pady=2)
        ttk.Button(info_frame, text="Load Sample Data", command=self.load_sample_data).pack(fill='x', pady=2)
        
        # Visualization selection
        viz_frame = ttk.LabelFrame(control_frame, text="Visualization Type", padding=10)
        viz_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(viz_frame, text="🎯 Velocity vs Intensity", 
                   command=self.plot_velocity_vs_intensity).pack(fill='x', pady=2)
        ttk.Button(viz_frame, text="📊 Success Rate Heatmap", 
                   command=self.plot_success_heatmap).pack(fill='x', pady=2)
        ttk.Button(viz_frame, text="📏 Landing Accuracy (Distance)", 
                   command=self.plot_landing_accuracy).pack(fill='x', pady=2)
        ttk.Button(viz_frame, text="🎯 Landing Accuracy (Top-Down)", 
                   command=self.plot_landing_topdown).pack(fill='x', pady=2)
        ttk.Button(viz_frame, text="📈 Monte Carlo Distributions", 
                   command=self.plot_monte_carlo).pack(fill='x', pady=2)

        # Trace scale changes (even if UI moved to menu)
        self.yscale_var.trace_add('write', self.on_scale_change)
        
        # Filters section
        filter_frame = ttk.LabelFrame(control_frame, text="Data Filters", padding=10)
        filter_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollable filter area
        filter_canvas = tk.Canvas(filter_frame, highlightthickness=0)
        filter_scrollbar = ttk.Scrollbar(filter_frame, orient="vertical", command=filter_canvas.yview)
        self.filter_content = ttk.Frame(filter_canvas)
        
        self.filter_content.bind(
            "<Configure>",
            lambda e: filter_canvas.configure(scrollregion=filter_canvas.bbox("all"))
        )
        
        filter_canvas.create_window((0, 0), window=self.filter_content, anchor="nw")
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        
        filter_canvas.pack(side="left", fill="both", expand=True)
        filter_scrollbar.pack(side="right", fill="y")
        
        self.filter_widgets = {}
        self.create_filter_widgets()
        
        # Apply filters button
        ttk.Button(filter_frame, text="Apply Filters", 
                   command=self.apply_filters, style='Accent.TButton').pack(fill='x', pady=5, side='bottom')
    
    def create_filter_widgets(self):
        """Create filter widgets (will be populated when data is loaded)"""
        tk.Label(self.filter_content, text="Load data to see available filters", 
                 fg='gray').pack(pady=20)
    
    def create_visualization_area(self):
        """Create right visualization area"""
        viz_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(viz_frame, weight=1)
        
        # Welcome message
        welcome_frame = ttk.Frame(viz_frame)
        welcome_frame.pack(expand=True)
        
        tk.Label(welcome_frame, text="PlotVisual", 
                 font=("Arial", 24, "bold")).pack(pady=10)
        tk.Label(welcome_frame, text="Advanced Visualization Suite for Project Vortex", 
                 font=("Arial", 12)).pack(pady=5)
        tk.Label(welcome_frame, text="\nLoad data to begin analysis", 
                 font=("Arial", 10), fg='gray').pack(pady=10)
        
        self.viz_container = viz_frame
    
    def load_data(self):
        """Load data from CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), 'results')
        )
        
        if not filename:
            return
        
        try:
            self.data = pd.read_csv(filename)
            self.filtered_data = self.data.copy()
            
            # Update info label
            info_text = f"Loaded: {len(self.data)} rows\n"
            if 'Type' in self.data.columns:
                info_text += f"ML: {len(self.data[self.data['Type']=='ML'])} | "
                info_text += f"Opt: {len(self.data[self.data['Type']=='Optimization'])}"
            
            self.data_info_label.config(text=info_text)
            self.status_var.set(f"Loaded {len(self.data)} records from {os.path.basename(filename)}")
            
            # Rebuild filter widgets
            self.rebuild_filter_widgets()
            
            messagebox.showinfo("Success", f"Loaded {len(self.data)} records successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
    
    def load_sample_data(self):
        """Load or generate sample data"""
        sample_dir = os.path.join(os.path.dirname(__file__), 'results', 'sample_data')
        os.makedirs(sample_dir, exist_ok=True)
        sample_file = os.path.join(sample_dir, 'sample_visualization_data.csv')
        
        if not os.path.exists(sample_file):
            # Generate sample data
            if messagebox.askyesno("Generate Sample Data", 
                                   "Sample data not found in results/sample_data/. Generate realistic sample data now?"):
                self.generate_sample_data(sample_file)
            else:
                return
        
        try:
            self.data = pd.read_csv(sample_file)
            self.filtered_data = self.data.copy()
            
            # Update info label
            info_text = f"Loaded: {len(self.data)} rows (Sample Data)\n"
            if 'Type' in self.data.columns:
                info_text += f"ML: {len(self.data[self.data['Type']=='ML'])} | "
                info_text += f"Opt: {len(self.data[self.data['Type']=='Optimization'])}"
            
            self.data_info_label.config(text=info_text)
            self.status_var.set(f"Loaded sample data: {len(self.data)} records")
            
            # Rebuild filter widgets
            self.rebuild_filter_widgets()
            
            messagebox.showinfo("Success", "Sample data loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data:\n{str(e)}")
    
    def generate_sample_data(self, output_file):
        """Generate realistic sample data for visualization"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        np.random.seed(42)  # Reproducible
        
        records = []
        
        # Generate paired runs (150 samples each)
        for i in range(150):
            # Shared Rocket Parameters
            dry_mass = np.random.uniform(45, 65)
            propellant_mass = np.random.uniform(8, 12)
            diameter = np.random.uniform(0.3, 0.4)
            thrust_avg = np.random.uniform(900, 1100)
            
            # Shared Environmental Conditions
            wind_speed = np.random.uniform(0, 12)
            drag_coef = np.random.uniform(0.45, 0.65)
            air_density = np.random.uniform(1.15, 1.25)
            
            # Shared Initial Conditions
            initial_alt = np.random.uniform(900, 1300)
            initial_vel = np.random.uniform(45, 65)
            
            # Shared Fault intensity (0 to 0.98)
            fault_intensity = np.random.beta(2, 4) * 0.98
            
            # Shared base performance (near ideal landing starting between 0.5 and 1.0 m/s)
            base_landing_vel = np.random.uniform(0.5, 1.0)
            mass_impact_shared = (dry_mass + propellant_mass - 55) * 0.04
            wind_impact_shared = wind_speed * 0.12

            # --- OPTIMIZATION AGENT CALCULATION ---
            # Optimization balloons aggressively as fault intensity increases
            # High exponent makes it stay low then jump to ~40-50 m/s at intensity 1.0
            opt_fault_impact = (fault_intensity ** 3.8) * 45.0
            
            opt_landing_vel = base_landing_vel + opt_fault_impact + mass_impact_shared + wind_impact_shared
            # Significant noise tied to fault intensity
            opt_landing_vel += np.random.normal(0, 0.1 + fault_intensity * 8)
            opt_landing_vel = max(0.1, opt_landing_vel)
            
            # Accuracy (Distance) scales poorly for Optimization
            opt_horiz_base = 4 + wind_speed * 0.5 + (fault_intensity ** 2.2) * 50
            opt_horiz_error = np.random.rayleigh(opt_horiz_base)
            opt_x, opt_y = np.random.normal(0, opt_horiz_error), np.random.normal(0, opt_horiz_error)
            
            records.append({
                'Type': 'Optimization',
                'Landing Velocity': opt_landing_vel,
                'Success': opt_landing_vel < 2.0,
                'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel, 'Landing X': opt_x,
                'Landing Y': opt_y, 'Landing Distance': np.sqrt(opt_x**2 + opt_y**2)
            })

            # --- ML AGENT CALCULATION ---
            # ML increases linearly from the initial ~0.5 m/s
            # Targeting ~5 m/s at intensity 0.7 (slope of ~6.4)
            ml_linear_base = base_landing_vel + (fault_intensity * 6.5)
            
            ml_landing_vel = ml_linear_base + (mass_impact_shared * 0.4) + (wind_impact_shared * 0.3)
            # Low noise
            ml_landing_vel += np.random.normal(0, 0.1 + fault_intensity * 0.3)
            
            # Application of requested clamping:
            # ~98% clamped to 2.0 m/s (perfectly meeting success)
            # ~2% follow the linear trend fully (outliers)
            if np.random.rand() < 0.98:
                # Still follow the linear trend but cap at the success line
                ml_landing_vel = min(ml_landing_vel, 2.0)
            else:
                # Outliers follow the linear trend reaching ~5m/s at 0.7
                ml_landing_vel = min(ml_landing_vel, 35.0) 
            
            ml_landing_vel = max(0.1, ml_landing_vel)
            
            # ML has superior horizontal accuracy
            ml_horiz_base = 2 + wind_speed * 0.15 + fault_intensity * 5
            ml_horiz_error = np.random.rayleigh(ml_horiz_base)
            ml_x, ml_y = np.random.normal(0, ml_horiz_error), np.random.normal(0, ml_horiz_error)

            records.append({
                'Type': 'ML',
                'Landing Velocity': ml_landing_vel,
                'Success': ml_landing_vel < 2.0,
                'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel, 'Landing X': ml_x,
                'Landing Y': ml_y, 'Landing Distance': np.sqrt(ml_x**2 + ml_y**2)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        
        messagebox.showinfo("Generated", f"Sample data generated:\n{output_file}\n{len(df)} records")
    
    def rebuild_filter_widgets(self):
        """Rebuild filter widgets based on loaded data"""
        # Clear existing widgets
        for widget in self.filter_content.winfo_children():
            widget.destroy()
        
        self.filter_widgets = {}
        
        if self.data is None:
            return
        
        # Numeric columns for range filtering
        numeric_cols = ['Dry Mass', 'Propellant Mass', 'Wind Speed', 'Total Fault Intensity',
                        'Initial Altitude', 'Initial Velocity', 'Thrust Average']
        
        row = 0
        for col in numeric_cols:
            if col in self.data.columns:
                tk.Label(self.filter_content, text=f"{col}:", anchor='w').grid(
                    row=row, column=0, sticky='w', padx=5, pady=2)
                
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                
                min_var = tk.DoubleVar(value=min_val)
                max_var = tk.DoubleVar(value=max_val)
                
                frame = ttk.Frame(self.filter_content)
                frame.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
                
                ttk.Entry(frame, textvariable=min_var, width=8).pack(side='left', padx=2)
                tk.Label(frame, text=" to ").pack(side='left')
                ttk.Entry(frame, textvariable=max_var, width=8).pack(side='left', padx=2)
                
                self.filter_widgets[col] = (min_var, max_var, min_val, max_val)
                row += 1
        
        # Type filter
        if 'Type' in self.data.columns:
            tk.Label(self.filter_content, text="Type:", anchor='w').grid(
                row=row, column=0, sticky='w', padx=5, pady=2)
            
            type_var = tk.StringVar(value="All")
            types = ['All'] + list(self.data['Type'].unique())
            ttk.Combobox(self.filter_content, textvariable=type_var, 
                         values=types, state='readonly', width=15).grid(
                row=row, column=1, sticky='w', padx=5, pady=2)
            
            self.filter_widgets['Type'] = type_var
            row += 1
        
        # Success filter
        if 'Success' in self.data.columns:
            tk.Label(self.filter_content, text="Success:", anchor='w').grid(
                row=row, column=0, sticky='w', padx=5, pady=2)
            
            success_var = tk.StringVar(value="All")
            ttk.Combobox(self.filter_content, textvariable=success_var, 
                         values=['All', 'True', 'False'], state='readonly', width=15).grid(
                row=row, column=1, sticky='w', padx=5, pady=2)
            
            self.filter_widgets['Success'] = success_var
            row += 1
        
        self.filter_content.columnconfigure(1, weight=1)
    
    def apply_filters(self):
        """Apply current filters to data"""
        if self.data is None:
            return
        
        self.filtered_data = self.data.copy()
        
        # Apply numeric range filters
        for col, widget_data in self.filter_widgets.items():
            if col in ['Type', 'Success']:
                continue
            
            min_var, max_var, orig_min, orig_max = widget_data
            min_val = min_var.get()
            max_val = max_var.get()
            
            if min_val > orig_min or max_val < orig_max:
                self.filtered_data = self.filtered_data[
                    (self.filtered_data[col] >= min_val) & 
                    (self.filtered_data[col] <= max_val)
                ]
        
        # Apply Type filter
        if 'Type' in self.filter_widgets:
            type_val = self.filter_widgets['Type'].get()
            if type_val != 'All':
                self.filtered_data = self.filtered_data[self.filtered_data['Type'] == type_val]
        
        # Apply Success filter
        if 'Success' in self.filter_widgets:
            success_val = self.filter_widgets['Success'].get()
            if success_val != 'All':
                success_bool = success_val == 'True'
                self.filtered_data = self.filtered_data[self.filtered_data['Success'] == success_bool]
        
        self.status_var.set(f"Filtered: {len(self.filtered_data)} / {len(self.data)} records")
        messagebox.showinfo("Filters Applied", f"Showing {len(self.filtered_data)} of {len(self.data)} records")

    def clear_visualization(self):
        """Clear current visualization area"""
        for widget in self.viz_container.winfo_children():
            widget.destroy()
        self.current_figure = None
        self.current_canvas = None

    def on_scale_change(self, *args):
        """Handle Y-scale changes by refreshing current view if applicable"""
        if self.last_view == 'velocity':
            self.plot_velocity_vs_intensity()
        elif self.last_view == 'accuracy':
            self.plot_landing_accuracy()

    def plot_velocity_vs_intensity(self):
        """Plot Landing Velocity vs Fault Intensity scatterplot"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            messagebox.showwarning("No Data", "Please load data first")
            return
            
        self.clear_visualization()
        self.last_view = 'velocity'
        
        # Create figure
        fig = Figure(figsize=(12, 7), dpi=100)
        ax = fig.add_subplot(111)
        
        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            ax.scatter(subset['Total Fault Intensity'], 
                      subset['Landing Velocity'],
                      c=colors[plot_type], 
                      s=50, 
                      alpha=0.6, 
                      label=plot_type,
                      edgecolors='black',
                      linewidth=0.5)
        
        ax.set_xlabel('Total Fault Intensity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('Landing Velocity vs Fault Intensity', fontsize=14, fontweight='bold')

        # Set y-scale according to user selection (Log/Linear)
        yscale = self.yscale_var.get()
        if yscale == 'Log':
            ax.set_yscale('log')
            # Richer log markings
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())
            
            # Ensure we show some range even if data is tight
            y_min = max(0.1, self.filtered_data['Landing Velocity'].min() * 0.5)
            y_max = self.filtered_data['Landing Velocity'].max() * 2.0
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_yscale('linear')
            # Dense linear markings
            y_max = max(5.0, self.filtered_data['Landing Velocity'].max() * 1.1)
            ax.set_ylim(0, y_max)
            
            # Add more markings: step size based on range
            if y_max > 50:
                major_step = 10
                minor_step = 2
            elif y_max > 20:
                major_step = 5
                minor_step = 1
            else:
                major_step = 2
                minor_step = 0.5
                
            ax.yaxis.set_major_locator(MultipleLocator(major_step))
            ax.yaxis.set_minor_locator(MultipleLocator(minor_step))

        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Add success threshold line at 2 m/s
        ax.axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Success Threshold (2 m/s)')
        
        ax.legend(fontsize=11, frameon=True, shadow=True)
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.viz_container)
        toolbar.update()
        
        self.current_figure = fig
        self.current_canvas = canvas
        
        self.status_var.set(f"Showing: Landing Velocity vs Fault Intensity ({yscale} scale)")
    
    def plot_success_heatmap(self):
        """Plot Success Rate heatmap across parameter ranges"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        if 'Success' not in self.filtered_data.columns:
            messagebox.showwarning("Missing Column", "Success column not found in data")
            return
        
        self.clear_visualization()
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8))
        
        # Heatmap 1: Wind Speed vs Fault Intensity
        ax1 = fig.add_subplot(221)
        self._create_heatmap(ax1, 'Wind Speed', 'Total Fault Intensity', 'Success')
        
        # Heatmap 2: Initial Altitude vs Fault Intensity
        ax2 = fig.add_subplot(222)
        self._create_heatmap(ax2, 'Initial Altitude', 'Total Fault Intensity', 'Success')
        
        # Heatmap 3: Dry Mass vs Wind Speed
        ax3 = fig.add_subplot(223)
        self._create_heatmap(ax3, 'Dry Mass', 'Wind Speed', 'Success')
        
        # Heatmap 4: Initial Velocity vs Fault Intensity
        ax4 = fig.add_subplot(224)
        self._create_heatmap(ax4, 'Initial Velocity', 'Total Fault Intensity', 'Success')
        
        fig.suptitle('Success Rate Heatmaps', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.viz_container)
        toolbar.update()
        
        self.current_figure = fig
        self.current_canvas = canvas
        
        self.status_var.set("Showing: Success Rate Heatmaps")
    
    def _create_heatmap(self, ax, x_col, y_col, z_col):
        """Helper to create a single heatmap"""
        if x_col not in self.filtered_data.columns or y_col not in self.filtered_data.columns:
            ax.text(0.5, 0.5, f'Missing: {x_col} or {y_col}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create bins
        x_bins = 10
        y_bins = 10
        
        x_edges = np.linspace(self.filtered_data[x_col].min(), 
                             self.filtered_data[x_col].max(), x_bins + 1)
        y_edges = np.linspace(self.filtered_data[y_col].min(), 
                             self.filtered_data[y_col].max(), y_bins + 1)
        
        # Calculate success rate in each bin
        heatmap = np.zeros((y_bins, x_bins))
        counts = np.zeros((y_bins, x_bins))
        
        for _, row in self.filtered_data.iterrows():
            x_idx = np.searchsorted(x_edges, row[x_col]) - 1
            y_idx = np.searchsorted(y_edges, row[y_col]) - 1
            
            x_idx = np.clip(x_idx, 0, x_bins - 1)
            y_idx = np.clip(y_idx, 0, y_bins - 1)
            
            counts[y_idx, x_idx] += 1
            if row[z_col]:
                heatmap[y_idx, x_idx] += 1
        
        # Calculate success rate
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', origin='lower',
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                      vmin=0, vmax=1)
        
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(f'{z_col} Rate', fontsize=10, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Success Rate')
    
    def plot_landing_accuracy(self):
        """Plot landing accuracy distance statistics"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        if 'Landing Distance' not in self.filtered_data.columns:
            messagebox.showwarning("Missing Column", "Landing Distance column not found")
            return
        
        self.last_view = 'accuracy'
        
        # Create figure
        fig = Figure(figsize=(12, 6))
        
        # Plot 1: Distance distribution by Type (Histogram)
        ax1 = fig.add_subplot(121)
        
        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            ax1.hist(subset['Landing Distance'], bins=30, alpha=0.6, 
                    color=colors[plot_type], label=plot_type, edgecolor='black')
        
        ax1.set_xlabel('Landing Distance from Target (m)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Landing Accuracy Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Distance vs Fault Intensity (Scatter)
        ax2 = fig.add_subplot(122)
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            ax2.scatter(subset['Total Fault Intensity'], 
                       subset['Landing Distance'],
                       c=colors[plot_type], 
                       s=40, 
                       alpha=0.6, 
                       label=plot_type,
                       edgecolors='black',
                       linewidth=0.5)
        
        ax2.set_xlabel('Total Fault Intensity', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Landing Distance from Target (m)', fontsize=11, fontweight='bold')
        ax2.set_title('Accuracy vs Fault Intensity', fontsize=12, fontweight='bold')
        
        # Apply Y-scale scaling
        yscale = self.yscale_var.get()
        if yscale == 'Log':
            ax2.set_yscale('log')
            ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
            ax2.yaxis.set_minor_formatter(NullFormatter())
            
            d_min = max(0.1, self.filtered_data['Landing Distance'].min() * 0.5)
            d_max = max(10, self.filtered_data['Landing Distance'].max() * 2.0)
            ax2.set_ylim(d_min, d_max)
        else:
            ax2.set_yscale('linear')
            d_max = max(10.0, self.filtered_data['Landing Distance'].max() * 1.1)
            ax2.set_ylim(0, d_max)
            # More linear markings
            if d_max > 100: major, minor = 20, 5
            elif d_max > 50: major, minor = 10, 2
            else: major, minor = 5, 1
            ax2.yaxis.set_major_locator(MultipleLocator(major))
            ax2.yaxis.set_minor_locator(MultipleLocator(minor))

        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--', which='both')
        
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.viz_container)
        toolbar.update()
        
        self.current_figure = fig
        self.current_canvas = canvas
        
        self.status_var.set(f"Showing: Landing Accuracy ({yscale} scale)")
    
    def plot_landing_topdown(self):
        """Plot top-down view of landing positions"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        if 'Landing X' not in self.filtered_data.columns or 'Landing Y' not in self.filtered_data.columns:
            messagebox.showwarning("Missing Columns", "Landing X/Y columns not found")
            return
        
        self.clear_visualization()
        
        # Create figure
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # Plot landing target at origin
        target_circle = plt.Circle((0, 0), 2, color='red', fill=False, 
                                   linewidth=3, label='Target (2m radius)', zorder=10)
        ax.add_patch(target_circle)
        
        # Plot landing points
        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}
        markers = {'ML': 'o', 'Optimization': 's', 'All': 'o'}
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            if 'Type' in self.filtered_data.columns:
                subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            else:
                subset = self.filtered_data
            
            ax.scatter(subset['Landing X'], 
                      subset['Landing Y'],
                      c=colors[plot_type], 
                      marker=markers[plot_type],
                      s=50, 
                      alpha=0.6, 
                      label=plot_type,
                      edgecolors='black',
                      linewidth=0.5)
        
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax.set_title('Landing Positions - Top-Down View', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # Add crosshairs at origin
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.viz_container)
        toolbar.update()
        
        self.current_figure = fig
        self.current_canvas = canvas
        
        self.status_var.set("Showing: Landing Positions (Top-Down)")
    
    def plot_monte_carlo(self):
        """Plot Monte Carlo distribution analysis"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        self.clear_visualization()
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8))
        
        # Plot 1: Landing Velocity distribution
        ax1 = fig.add_subplot(231)
        self._plot_distribution(ax1, 'Landing Velocity', 'm/s')
        
        # Plot 2: Landing Distance distribution
        ax2 = fig.add_subplot(232)
        self._plot_distribution(ax2, 'Landing Distance', 'm')
        
        # Plot 3: Fault Intensity distribution
        ax3 = fig.add_subplot(233)
        self._plot_distribution(ax3, 'Total Fault Intensity', '')
        
        # Plot 4: Success rate by type
        ax4 = fig.add_subplot(234)
        self._plot_success_rate(ax4)
        
        # Plot 5: Dry Mass distribution
        ax5 = fig.add_subplot(235)
        self._plot_distribution(ax5, 'Dry Mass', 'kg')
        
        # Plot 6: Wind Speed distribution
        ax6 = fig.add_subplot(236)
        self._plot_distribution(ax6, 'Wind Speed', 'm/s')
        
        fig.suptitle('Monte Carlo Distribution Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.viz_container)
        toolbar.update()
        
        self.current_figure = fig
        self.current_canvas = canvas
        
        self.status_var.set("Showing: Monte Carlo Distributions")
    
    def _plot_distribution(self, ax, column, unit):
        """Helper to plot distribution for a column"""
        if column not in self.filtered_data.columns:
            ax.text(0.5, 0.5, f'Missing: {column}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        types = self.filtered_data['Type'].unique() if 'Type' in self.filtered_data.columns else ['All']
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db', 'All': '#95a5a6'}
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            if 'Type' in self.filtered_data.columns:
                subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            else:
                subset = self.filtered_data
            
            ax.hist(subset[column], bins=20, alpha=0.6, 
                   color=colors[plot_type], label=plot_type, edgecolor='black')
        
        xlabel = f'{column} ({unit})' if unit else column
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(column, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_success_rate(self, ax):
        """Helper to plot success rate comparison"""
        if 'Success' not in self.filtered_data.columns or 'Type' not in self.filtered_data.columns:
            ax.text(0.5, 0.5, 'Missing: Success or Type', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        types = self.filtered_data['Type'].unique()
        success_rates = []
        labels = []
        colors_list = []
        
        colors = {'ML': '#2ecc71', 'Optimization': '#3498db'}
        
        for plot_type in types:
            if plot_type not in colors:
                continue
            
            subset = self.filtered_data[self.filtered_data['Type'] == plot_type]
            rate = subset['Success'].sum() / len(subset) * 100
            
            success_rates.append(rate)
            labels.append(plot_type)
            colors_list.append(colors[plot_type])
        
        ax.bar(labels, success_rates, color=colors_list, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Success Rate (%)', fontsize=9)
        ax.set_title('Success Rate by Type', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add percentage labels on bars
        for i, (label, rate) in enumerate(zip(labels, success_rates)):
            ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    def export_plot(self):
        """Export current plot to file"""
        if self.current_figure is None:
            messagebox.showwarning("No Plot", "No plot to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """PlotVisual - Advanced Visualization Suite
        
Version: 1.0
Project: Vortex Desktop Suite

Specialized graph analysis tool for rocket landing simulations.

Features:
• Landing Velocity vs Fault Intensity analysis
• Success Rate heatmaps across parameters
• Landing Accuracy metrics and spatial views
• Monte Carlo distribution analysis
• Interactive filtering and visualization

© 2026 HexaKinetic Systems"""
        
        messagebox.showinfo("About PlotVisual", about_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PlotVisualApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

# Project Vortex - Suicide Burn Flight Dynamics Simulation

A comprehensive 6-degree-of-freedom (6DOF) flight dynamics simulation for testing the feasibility of landing a solid-fuel rocket using a suicide burn (hoverslam) technique.

## Features

### Physics Engine
- Full 6DOF dynamics with quaternion-based attitude representation
- Gravity model
- Air resistance using drag equation with altitude-varying air density
- Wind models: constant, altitude-varying (power law), or random gusts
- Time-varying vehicle mass as fuel burns

### Solid Motor Model
- Non-throttleable motor with user-definable thrust curve
- Thrust Vector Control (TVC) system with gimbal limits
- First-order lag model for TVC actuator response
- Burns to completion once ignited

### Suicide Burn Controller
- Analytical calculation of optimal ignition timing
- PD control for attitude stabilization via TVC
- Targets: v=0 and altitude=0 at motor burnout

### Monte Carlo Optimization
- Automated search for optimal ignition altitude
- Varies multiple parameters:
  - Thrust curve
  - TVC response time
  - Sensor accuracy (altimeter, velocity)
  - Wind speed and direction
  - Air density and temperature
  - Drag coefficient
  - Motor mass and depletion rate
- Iterative refinement around analytical estimate (±10m in 0.1m steps)

### GUI Interface
- Tkinter-based interface (lightweight and fast)
- Fully configurable rocket parameters:
  - Mass (dry and propellant)
  - Geometry (length, diameter)
  - Thrust curve (custom time-thrust pairs)
  - TVC parameters (max angle, response time, control gains)
- Configurable environment:
  - Atmospheric properties
  - Wind model selection and parameters
  - Initial conditions (altitude, velocity)
- Monte Carlo configuration:
  - Number of runs per altitude
  - Search range and step size
  - Variation percentages for all parameters
- Real-time output logging
- Automated plot generation

### Visualization
All plots saved as PNG files to avoid GUI crashes:
- Success rate vs ignition altitude
- 2D trajectory plots (altitude, velocity, speed, mass vs time)
- 3D trajectory visualization with start/end markers
- Attitude plots (quaternion components, angular velocity)

### Data Export
- Timestamped CSV files with complete trajectory data
- Optimization results (altitude vs success rate)
- Full state history (position, velocity, attitude, mass)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/endrocryne/Project-Vortex.git
cd Project-Vortex
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0

## Usage

### Quick Start

```bash
python main.py
```

This will attempt to launch the GUI. If Tkinter is not available, it will automatically run a demo simulation using the command-line interface.

### Running the GUI

```bash
python gui.py
```

**Note**: The GUI requires Tkinter. If not available, use the CLI instead.

### Using the Command-Line Interface (CLI)

The CLI is always available and doesn't require Tkinter:

```bash
# Run a single simulation with default parameters
python cli.py --mode single

# Run Monte Carlo optimization
python cli.py --mode optimize

# Use a custom configuration file
python cli.py --mode single --config my_config.json
python cli.py --mode optimize --config my_config.json
```

### Configuration File Format (JSON)

```json
{
  "rocket": {
    "dry_mass": 50.0,
    "propellant_mass": 10.0,
    "length": 5.0,
    "diameter": 0.3,
    "thrust_curve": [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
    "burn_time": 3.0,
    "tvc_max_angle": 5.0,
    "tvc_response_time": 0.1,
    "tvc_kp": 0.5,
    "tvc_kd": 0.1
  },
  "environment": {
    "gravity": 9.81,
    "air_density": 1.225,
    "wind_model": "altitude_varying",
    "wind_speed": 5.0
  },
  "simulation": {
    "altimeter_error": 0.01,
    "velocity_sensor_error": 0.01
  }
}
```

### Using the GUI

1. **Rocket Tab**: Configure rocket parameters
   - Set mass properties (dry mass, propellant mass)
   - Define geometry (length, diameter)
   - Enter thrust curve as comma-separated time,thrust pairs
   - Configure TVC parameters and control gains

2. **Environment Tab**: Configure environment and initial conditions
   - Set atmospheric properties (gravity, air density, temperature)
   - Choose wind model and parameters
   - Set initial altitude and velocity

3. **Simulation Tab**: Configure Monte Carlo parameters
   - Set number of Monte Carlo runs
   - Define altitude search range and step size
   - Configure variation percentages for uncertainty analysis

4. **Run Tab**: Execute simulations
   - Click "Run Optimization" for full Monte Carlo search
   - Click "Run Single Simulation" for one run with analytical estimate
   - View real-time output in the text area
   - Results saved automatically to `results/` directory

### Output Files

All results are saved to the `results/` directory with timestamps:

- `optimization_YYYYMMDD_HHMMSS.csv` - Success rate vs altitude data
- `trajectory_YYYYMMDD_HHMMSS.csv` - Full trajectory state history
- `single_run_YYYYMMDD_HHMMSS.csv` - Single simulation trajectory
- `success_rate_YYYYMMDD_HHMMSS.png` - Success rate plot
- `trajectory_2d_YYYYMMDD_HHMMSS.png` - 2D trajectory plots
- `trajectory_3d_YYYYMMDD_HHMMSS.png` - 3D trajectory visualization

## Technical Details

### State Vector

The simulation uses a 14-element state vector:
```
[x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz, mass]
```

Where:
- `x, y, z`: Position in inertial frame (m)
- `vx, vy, vz`: Velocity in inertial frame (m/s)
- `qw, qx, qy, qz`: Attitude quaternion (scalar-first convention)
- `ωx, ωy, ωz`: Angular velocity in body frame (rad/s)
- `mass`: Current vehicle mass (kg)

### Numerical Integration

- Method: RK45 (Runge-Kutta 4(5) with adaptive step size)
- Relative tolerance: 1e-6
- Absolute tolerance: 1e-9
- Maximum step size: 0.01 s

### Success Criteria

A landing is considered successful if:
- Final altitude within ±0.5 m of target (0 m)
- Final vertical velocity < 2 m/s
- Final total velocity < 3 m/s

### Wind Models

1. **Constant**: Uniform wind at all altitudes
2. **Altitude-varying**: Power law profile `v(h) = v_ref * (h/h_ref)^α`
3. **Gusts**: Constant wind with sinusoidal gusts

### Monte Carlo Variations

Each simulation run applies random variations to:
- Thrust magnitude (±5% default)
- Drag coefficient (±10% default)
- Air density (±5% default)
- TVC response time (±10% default)
- Mass flow rate (±2% default)
- Altimeter reading (±1% default)
- Velocity sensor reading (±1% default)

## Example Configuration

### Standard Test Case

- **Rocket**:
  - Dry mass: 50 kg
  - Propellant mass: 10 kg
  - Length: 5 m
  - Diameter: 0.3 m
  - Thrust: 1000 N for 3 seconds
  - TVC: ±5° gimbal, 0.1 s response time

- **Environment**:
  - Initial altitude: 1000 m
  - Initial velocity: -50 m/s (falling)
  - Wind: 5 m/s constant
  - Standard atmosphere

- **Monte Carlo**:
  - 100 runs per altitude
  - Search ±10 m around estimate
  - 0.1 m step size

Expected optimal ignition altitude: ~40-50 m

## Architecture

```
main.py              # Entry point
gui.py               # Tkinter GUI interface
simulation.py        # Main simulation class
physics_engine.py    # 6DOF physics and atmosphere models
solid_motor.py       # Solid motor and TVC models
requirements.txt     # Python dependencies
```

## Contributing

This is an educational/research project. Feel free to fork and experiment!

## License

MIT License - See LICENSE file for details

## References

- Quaternion kinematics for spacecraft dynamics
- Barometric formula for atmosphere modeling
- Power law wind profile for boundary layer
- Euler's rotation equations for rigid body dynamics
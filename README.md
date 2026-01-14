# Project-Vortex

High-Fidelity 6-DOF TVC Rocket Simulation

## Overview

Project Vortex is a comprehensive physics-based simulation of a Thrust Vector Control (TVC) model rocket with full six degrees of freedom (6-DOF) dynamics. The simulation accurately models:

- **Non-linear Equations of Motion** in 3D space
- **Variable mass properties** with shifting Center of Gravity (CG)
- **Hollow cylinder fuel grain geometry** burning from inside out
- **Active thrust vector control** with PID-based attitude stabilization
- **Aerodynamic drag and lift** with center of pressure effects
- **Logarithmic wind shear** and turbulence
- **Standard atmosphere model** for variable air density

## Features

### Physics Implementation

- **State Vector (13 DOF)**: Position, Velocity, Quaternion, Angular Velocity
- **Translational Dynamics**: F_total = m(t) · dV/dt
- **Rotational Dynamics**: Euler's equations with gyroscopic effects
- **Quaternion Kinematics**: Drift-free attitude representation
- **Dynamic Inertia Tensor**: Updates with fuel consumption

### Software Architecture

The simulation is built with a modular, object-oriented design:

1. **`Rocket` Class** (`tvc_simulation/rocket.py`)
   - Variable mass and inertia properties
   - Hollow cylinder fuel grain model
   - TVC thrust vector computation
   - Gimbal angle limits and actuation

2. **`Environment` Class** (`tvc_simulation/environment.py`)
   - Standard atmosphere model
   - Logarithmic wind shear
   - Turbulence and gust modeling

3. **`GNC` Class** (`tvc_simulation/gnc.py`)
   - PID controller for attitude stabilization
   - Quaternion-based error computation
   - Rate damping for oscillation reduction

4. **`Simulation` Class** (`tvc_simulation/simulation.py`)
   - Integration using `scipy.integrate.solve_ivp`
   - Event detection for ground impact
   - State history logging

5. **`Visualization` Module** (`tvc_simulation/visualization.py`)
   - 3D trajectory plotting
   - Time-series analysis plots
   - Flight summary statistics

## Installation

### Requirements

- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/endrocryne/Project-Vortex.git
cd Project-Vortex

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the simulation with default parameters (Estes F15-like motor):

```bash
python main.py
```

This will:
1. Initialize a model rocket with TVC capability
2. Simulate the flight with active attitude control
3. Generate comprehensive visualization plots
4. Display flight summary statistics

### Output

The simulation produces:
- **3D trajectory plot** showing the full flight path
- **Altitude vs time** graph
- **Velocity vs time** (magnitude and components)
- **Euler angles** (roll, pitch, yaw) vs time
- **Gimbal actuation angles** showing TVC activity
- **Thrust and mass** evolution during flight
- **Flight summary** with key metrics

Results are saved as `tvc_simulation_results.png`.

## Customization

### Modifying Rocket Parameters

Edit the `rocket_config` dictionary in `main.py`:

```python
rocket_config = {
    'mass_dry': 0.150,        # Dry mass (kg)
    'mass_fuel': 0.025,       # Fuel mass (kg)
    'length': 0.5,            # Rocket length (m)
    'diameter': 0.04,         # Body diameter (m)
    'cd': 0.5,                # Drag coefficient
    'cp_position': 0.4,       # Center of pressure (m from nose)
    'max_gimbal_angle_deg': 5.0,  # TVC deflection limit (degrees)
    # ... more parameters
}
```

### Tuning the Controller

Adjust PID gains in `gnc_config`:

```python
gnc_config = {
    'kp_pitch': 3.0,  # Proportional gain
    'ki_pitch': 0.2,  # Integral gain
    'kd_pitch': 1.0,  # Derivative gain
    # ... similar for yaw
}
```

### Custom Thrust Curve

Replace the `create_estes_f15_thrust_curve()` function with your own motor data:

```python
def create_custom_thrust_curve():
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    thrust = np.array([0.0, 30.0, 25.0, 15.0, 0.0])
    return time, thrust
```

## Mathematical Details

### Coordinate Frames

- **Inertial Frame (I)**: NED convention (North-East-Down)
- **Body Frame (B)**: Origin at instantaneous CG, X-axis points forward

### Key Equations

**Translational Dynamics:**
```
F_total^(I) = m(t) · dV^(I)/dt
```

**Rotational Dynamics (Euler's Equations):**
```
M_total^(B) = I(t) · dω^(B)/dt + ω^(B) × (I(t) · ω^(B))
```

**Quaternion Kinematics:**
```
dq/dt = (1/2) q ⊗ ω
```

**Thrust Vector Control:**
```
F_thrust^(B) = [cos(δp)cos(δy), sin(δy), sin(δp)]ᵀ · T(t)
M_TVC^(B) = r_pivot × F_thrust^(B)
```

## Project Structure

```
Project-Vortex/
├── tvc_simulation/
│   ├── __init__.py
│   ├── rocket.py           # Rocket dynamics and properties
│   ├── environment.py      # Atmospheric and wind models
│   ├── gnc.py             # Guidance, Navigation, Control
│   ├── simulation.py      # Main simulation engine
│   ├── utils.py           # Quaternion math utilities
│   └── visualization.py   # Plotting and analysis
├── main.py                # Main executable script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technical Notes

- **Integration Method**: RK45 (Runge-Kutta 4th/5th order adaptive)
- **Time Step**: Adaptive, with max step of 10ms
- **Quaternion Normalization**: Applied at each step to prevent drift
- **Event Detection**: Automatic ground impact detection
- **Mass Flow**: Calculated from thrust and specific impulse

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Acknowledgments

This simulation implements aerospace engineering principles from:
- Classical Mechanics and Rigid Body Dynamics
- Rocket Propulsion Elements
- Spacecraft Attitude Determination and Control
- Model Rocketry Best Practices

---

**Author**: endrocryne  
**Version**: 1.0.0
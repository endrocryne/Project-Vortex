# Examples

This directory contains example simulations demonstrating different scenarios and configurations for the 6-DOF TVC rocket simulation.

## Available Examples

### 1. High Altitude Flight (`high_altitude_flight.py`)

Demonstrates a high-power rocket flight with a G-class motor.

**Features:**
- Larger rocket (80cm, 400g dry mass)
- High-power motor (~80N average thrust)
- Extended flight time
- Higher apogee

**Run:**
```bash
cd examples
python high_altitude_flight.py
```

### 2. Windy Conditions (`windy_conditions.py`)

Tests the TVC control system under strong wind and turbulence.

**Features:**
- 10 m/s wind from East
- 25% turbulence intensity
- Higher PID gains for aggressive control
- Wind drift analysis

**Run:**
```bash
cd examples
python windy_conditions.py
```

### 3. Passive Flight (`passive_flight.py`)

Demonstrates rocket behavior without active thrust vector control.

**Features:**
- TVC disabled (zero PID gains)
- Initial 10° pitch and 5° yaw tilt
- Ballistic trajectory
- Shows importance of active control

**Run:**
```bash
cd examples
python passive_flight.py
```

## Creating Your Own Examples

To create a custom simulation:

1. Import the necessary modules:
```python
from tvc_simulation.rocket import Rocket
from tvc_simulation.environment import Environment
from tvc_simulation.gnc import GNC
from tvc_simulation.simulation import Simulation
from tvc_simulation.visualization import plot_results, print_summary
```

2. Configure your rocket, environment, and GNC parameters
3. Create the simulation object and set initial conditions
4. Run the simulation with `simulation.run()`
5. Visualize results with `plot_results()` and `print_summary()`

## Parameter Tuning Tips

### Rocket Configuration
- **mass_dry/mass_fuel**: Affects thrust-to-weight ratio and flight duration
- **cd**: Drag coefficient (0.4-0.6 typical for model rockets)
- **cp_position**: Center of pressure (should be behind CG for stability)
- **max_gimbal_angle_deg**: Mechanical limit for TVC deflection

### Environment
- **wind_reference_speed**: Surface wind speed in m/s
- **turbulence_intensity**: Fraction (0.1 = 10% turbulence)
- **wind_direction**: Radians from North (0=N, π/2=E, π=S, 3π/2=W)

### GNC (PID Tuning)
- **kp**: Proportional gain (2-4 typical, higher = more aggressive)
- **ki**: Integral gain (0.1-0.3 typical, eliminates steady-state error)
- **kd**: Derivative gain (0.5-1.5 typical, reduces oscillations)
- Start with low gains and increase gradually
- If oscillating, reduce kp and increase kd
- If slow response, increase kp
- If steady-state error, increase ki slightly

## Expected Results

All examples generate:
- 3D trajectory plot
- Altitude vs time
- Velocity vs time
- Euler angles (roll, pitch, yaw)
- Gimbal actuation angles
- Thrust and mass curves
- Flight summary statistics

Output images are saved in the examples directory.

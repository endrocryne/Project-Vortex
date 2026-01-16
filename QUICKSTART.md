# Project Vortex - Quick Start Guide

## Installation

```bash
git clone https://github.com/endrocryne/Project-Vortex.git
cd Project-Vortex
pip install -r requirements.txt
```

## Quick Usage

### Option 1: Auto-Run (Recommended for First Time)
```bash
python main.py
```
This will try to launch the GUI. If Tkinter is unavailable, it automatically runs a demo simulation.

### Option 2: Command-Line Interface (Always Works)

**Single Simulation:**
```bash
python cli.py --mode single
```

**Monte Carlo Optimization:**
```bash
python cli.py --mode optimize
```

**Using Configuration Files:**
```bash
# Ideal conditions (no variations)
python cli.py --mode single --config config_ideal.json

# Realistic conditions (5-10% variations)
python cli.py --mode single --config config_realistic.json

# Challenging conditions (10-20% variations, high wind)
python cli.py --mode single --config config_challenging.json
```

## Understanding the Results

### Output Files (in `results/` directory)

1. **CSV Files**: Complete trajectory data
   - `single_run_TIMESTAMP.csv` - Single simulation trajectory
   - `optimization_TIMESTAMP.csv` - Success rates vs altitude
   - `trajectory_TIMESTAMP.csv` - Best trajectory from optimization

2. **Plot Files**: All plots saved as PNG
   - `trajectory_2d_TIMESTAMP.png` - 4-panel plot showing:
     - Altitude vs Time
     - Vertical Velocity vs Time
     - Total Speed vs Time
     - Mass vs Time
   - `trajectory_3d_TIMESTAMP.png` - 3D trajectory with start/end markers
   - `success_rate_TIMESTAMP.png` - Success rate vs ignition altitude (optimization only)

### Interpreting the Plots

**Altitude Plot**: Shows the rocket descending from initial altitude. You should see:
- Smooth descent during freefall
- Change in slope when motor ignites
- Final touchdown at altitude ≈ 0

**Velocity Plot**: Shows vertical velocity (negative = falling). You should see:
- Increasing negative velocity during freefall (accelerating downward)
- Decrease in negative velocity when motor fires (deceleration)
- Target: velocity near 0 at touchdown

**Mass Plot**: Shows vehicle mass decreasing as fuel burns:
- Constant mass during freefall
- Linear decrease during motor burn
- Final mass = dry mass

**Success Criteria**:
- Final altitude: within ±0.5 m of 0
- Final vertical velocity: < 2 m/s
- Final total velocity: < 3 m/s

## Configuration Examples

### Creating Custom Configurations

Create a JSON file with your parameters:

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
    "tvc_kd": 0.1,
    "thrust_variation": 0.05,
    "tvc_response_variation": 0.1,
    "mass_variation": 0.02
  },
  "environment": {
    "gravity": 9.81,
    "air_density": 1.225,
    "temperature": 288.15,
    "drag_coefficient": 0.5,
    "wind_model": "altitude_varying",
    "wind_speed": 5.0,
    "wind_direction": 0.0,
    "drag_variation": 0.1,
    "air_density_variation": 0.05
  },
  "simulation": {
    "altimeter_error": 0.01,
    "velocity_sensor_error": 0.01
  }
}
```

### Parameter Descriptions

**Rocket Parameters:**
- `dry_mass`: Vehicle mass without propellant (kg)
- `propellant_mass`: Fuel mass (kg)
- `length`: Vehicle length (m)
- `diameter`: Vehicle diameter (m)
- `thrust_curve`: List of [time, thrust] pairs defining motor performance
- `tvc_max_angle`: Maximum gimbal angle (degrees)
- `tvc_response_time`: TVC actuator time constant (seconds)
- `tvc_kp_pitch`, `tvc_ki_pitch`, `tvc_kd_pitch`: PID gains for pitch control (Y-axis motor)
- `tvc_kp_yaw`, `tvc_ki_yaw`, `tvc_kd_yaw`: PID gains for yaw control (X-axis motor)

**Environment Parameters:**
- `wind_model`: "constant", "altitude_varying", or "gusts"
- `wind_speed`: Wind speed at reference height (m/s)
- `wind_direction`: Wind direction (degrees, 0 = East, 90 = North)
- `drag_coefficient`: Aerodynamic drag coefficient

**Monte Carlo Variations** (fraction of nominal value, e.g., 0.05 = ±5%):
- `thrust_variation`: Thrust uncertainty
- `drag_variation`: Drag coefficient uncertainty
- `air_density_variation`: Atmospheric density uncertainty
- `tvc_response_variation`: TVC response time uncertainty
- `mass_variation`: Mass flow rate uncertainty
- `altimeter_error`: Altitude sensor error
- `velocity_sensor_error`: Velocity sensor error

## Typical Workflow

1. **Start with ideal conditions** to understand the basic behavior:
   ```bash
   python cli.py --mode single --config config_ideal.json
   ```

2. **Add realistic variations** to test robustness:
   ```bash
   python cli.py --mode single --config config_realistic.json
   ```

3. **Run optimization** to find the best ignition altitude:
   ```bash
   python cli.py --mode optimize --config config_realistic.json
   ```
   Note: This can take 5-10 minutes depending on settings.

4. **Review results** in the `results/` directory:
   - Check CSV files for detailed data
   - View PNG plots for visual analysis
   - Look at success rates from optimization

## Troubleshooting

**Problem**: Simulation says "Success: False"
- **Solution**: The calculated ignition altitude may not be optimal. Try running optimization mode to find the best altitude.

**Problem**: Rocket crashes at high speed
- **Solution**: Increase thrust or decrease initial velocity/altitude. The motor may not have enough impulse for the given conditions.

**Problem**: Simulation takes too long
- **Solution**: Reduce Monte Carlo runs or search range. Default is 100 runs × 200 altitudes = 20,000 simulations!

**Problem**: Large drift from landing point
- **Solution**: This is expected with wind. Increase TVC controller gains (`tvc_kp`, `tvc_kd`) for better attitude control.

## Tips for Success

1. **Thrust-to-Weight Ratio**: For suicide burn to work, the motor must provide > 1 TWR at burnout.
   - Check: `avg_thrust / (dry_mass × gravity) > 1`
   
2. **Burn Time vs Fall Time**: The motor must have enough burn time to decelerate.
   - Longer burn = more control authority
   - Shorter burn = more efficient but harder to time

3. **TVC Authority**: The gimbal angle affects attitude control.
   - Larger angle = more control power
   - But also reduces effective thrust

4. **Monte Carlo Settings**: Balance between accuracy and computation time.
   - 10 runs: Quick test, rough estimate
   - 100 runs: Good for optimization
   - 1000 runs: High confidence, very slow

## Example Outputs

All simulations automatically generate:
- CSV file with complete state history (time, position, velocity, mass)
- 2D plots showing altitude, velocity, speed, mass vs time
- 3D trajectory plot with start/end markers
- Success rate plot (optimization mode only)

Files are timestamped to prevent overwriting.

## Next Steps

- Modify thrust curves to test different motors
- Adjust initial conditions for different scenarios
- Experiment with wind models and strengths
- Use optimization to find robust ignition altitudes
- Analyze CSV data for detailed performance metrics

## Support

See `PHYSICS_REFERENCE.md` for detailed equations and assumptions.
See `README.md` for complete documentation.

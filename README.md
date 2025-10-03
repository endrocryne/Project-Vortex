# 6-DOF Rocket Simulation with Environmental & Hardware Imperfections (Phase 4)

This project is a 6-Degrees-of-Freedom (6DOF) rocket simulation written in Python. It builds upon a simpler model by incorporating more realistic environmental conditions and hardware imperfections, providing a more challenging and accurate simulation of a rocket's flight dynamics under thrust vector control (TVC).

The simulation uses a Monte Carlo approach to analyze the rocket's performance and stability across a range of randomized parameters, including atmospheric conditions, hardware characteristics, and initial state disturbances.

## Features

*   **6-DOF Dynamics**: Simulates the rocket's position, velocity, orientation (using quaternions), and angular velocity in 3D space.
*   **Monte Carlo Analysis**: Runs hundreds or thousands of simulations with randomized parameters to generate statistical data on performance metrics like apogee, landing distance, and flight stability.
*   **Variable Wind Model**: Implements a wind power law (`V_wind(h) = V_ground * (h / 10.0)**0.15`) to simulate increasing wind speed with altitude, affecting the rocket's trajectory.
*   **Noisy Sensor Model**: The PID controller's error calculations are contaminated with Gaussian noise to simulate imperfections in the gyroscope. Apogee detection is also affected by noisy barometer readings.
*   **Actuator Dynamics**: Models the gimbaled engine mount (actuators) as a first-order system with a time delay (tau) and precision noise, meaning the control system's commands are not executed instantly or perfectly.
*   **Detailed Output**: Generates summary statistics, raw data CSV files, aggregate distribution plots (apogee, landing distance, stability), and an interactive GUI showing detailed plots for a single flight with mean parameters.

## Quick Start

Follow these steps to get the simulation running.

### 1. Installation

The simulation relies on a few common scientific computing libraries for Python. You can install them using pip:

```bash
pip install numpy scipy matplotlib
```

### 2. Running the Simulation

To run the full Monte Carlo simulation and view the results, execute the script from your terminal:

```bash
python phase4_simulation.py
```

This will:
1.  Run a number of simulations defined by the `N_RUNS` variable in the script.
2.  Print summary statistics for apogee, landing distance, and stability to the console.
3.  Save the raw data for all runs to `phase4_monte_carlo_results.csv`.
4.  Save aggregate distribution plots for apogee, landing distance, and stability as PNG images (`phase4_*.png`).
5.  Launch a Tkinter GUI window with detailed, interactive plots from a final simulation run that uses the mean (average) of all parameters.

## Usage and Configuration

The primary way to configure the simulation is by modifying the global parameters at the top of the `phase4_simulation.py` script.

*   `N_RUNS`: Set the number of Monte Carlo simulations to run. Higher numbers give better statistical results but take longer to complete.
*   **Parameter Distributions**: You can adjust the `_MEAN` and `_STD` (or `_STD_PERCENT`) values for various parameters to see how they affect the rocket's performance. This includes:
    *   Thrust and propellant mass.
    *   Aerodynamic coefficients.
    *   Initial orientation disturbances.
    *   **Ground wind speed**.
    *   **Sensor noise levels**.
    *   **Actuator response time and precision**.

## Debugging and Troubleshooting

*   **Low Stability / Success Rate**: If most flights are unstable, the control system may be having trouble compensating for the disturbances.
    *   **Check `CP_Z_OFFSET_MEAN`**: This value represents the distance between the Center of Pressure (CP) and the Center of Mass (CoM). It **must be negative** for the rocket to be passively stable. A value closer to zero makes the rocket more agile but harder to control.
    *   **Reduce Disturbances**: Try lowering the `_STD` values for `GROUND_WIND_SPEED`, `YAW_INIT`, and `PITCH_INIT` to make the conditions less challenging.
    *   **Tune PID Gains**: The `KP`, `KI`, and `KD` constants may need to be adjusted to better handle the sensor noise and actuator lag.

*   **Import Errors**: If you get an error like `ModuleNotFoundError`, ensure you have installed all the required libraries as described in the **Installation** section.

*   **Slow Performance**: The simulation can be computationally intensive. For quicker tests, reduce `N_RUNS` to a small number (e.g., 10 or 20).
# Project Vortex: 6-DOF Rocket Simulation

Welcome to Project Vortex! This project is a Python-based 6-Degrees-of-Freedom (6DOF) rocket simulation. This README file provides all the necessary information to understand, run, and contribute to the project.

## Phase 1: Vertical Flight Simulation

This initial phase simulates the flight of a simple rocket flying straight up in a vacuum. It does not include aerodynamics or control systems, focusing purely on the core physics of thrust and gravity.

## Quick Start

Follow these steps to run the simulation and see the results.

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository-url>
    cd project-vortex
    ```

2.  **Install the required Python libraries:**
    A `requirements.txt` file is not yet included, but you can install the necessary packages directly using pip:
    ```bash
    pip install numpy scipy matplotlib
    ```

### Running the Simulation

Execute the main simulation script from your terminal:
```bash
python phase1_simulation.py
```

After running, the script will output a message to the console:
```
Simulation complete. Trajectory plot saved as 'phase1_trajectory.png'
```
You will find a new file named `phase1_trajectory.png` in the project directory. This image shows a plot of the rocket's altitude over time.

## Usage and Configuration

The `phase1_simulation.py` script is designed to be run directly. However, you can modify the constants at the top of the file to see how they affect the rocket's trajectory.

### Key Constants

-   `G`: Gravitational acceleration (m/s²).
-   `DRY_MASS`: The mass of the rocket without any propellant (kg).
-   `PROPELLANT_MASS`: The initial mass of the propellant (kg).
-   `ENGINE_BURN_TIME`: The duration for which the engine provides thrust (seconds).
-   `THRUST_FORCE`: The constant force produced by the engine during its burn time (Newtons).
-   `INERTIA_TENSOR`: A diagonal matrix representing the rocket's rotational inertia. In this phase, it's not critical as there are no torques.

### Simulation Time

You can adjust the simulation duration by changing the `t_span` and `t_eval` variables in the main execution block:
```python
# Simulation time
t_span = [0, 30]  # Start and end time in seconds
t_eval = np.linspace(t_span[0], t_span[1], 1001) # Timestamps for evaluation
```

## Debugging

If you encounter issues, here are a few things to check:

1.  **"ModuleNotFoundError"**:
    This error means one of the required libraries (`numpy`, `scipy`, `matplotlib`) is not installed. Make sure you have run the installation command:
    ```bash
    pip install numpy scipy matplotlib
    ```

2.  **Incorrect Trajectory Plot**:
    The expected plot should be a parabolic arc, showing the rocket ascending while the engine is firing and then descending under gravity. If the plot looks different:
    -   Verify that the initial state `initial_state` is set correctly, with `qw = 1` and all other elements as zero.
    -   Check the force calculations inside `rocket_dynamics`. Ensure gravity is always applied and thrust is only applied during the `ENGINE_BURN_TIME`.

3.  **Simulation Fails to Run**:
    The `scipy.integrate.solve_ivp` function can sometimes fail if the differential equations are stiff or contain errors.
    -   Add print statements inside `rocket_dynamics` to inspect the values of `current_mass`, `F_total`, and the state derivatives at different time steps (`t`). This can help identify issues like division by zero or non-numeric values.

## How It Works

The simulation is built around a state vector and a function that describes how this state changes over time (`rocket_dynamics`).

### The State Vector

The state of the rocket at any given time `t` is represented by a 13-element NumPy array:
-   **Position** `[x, y, z]`
-   **Velocity** `[vx, vy, vz]`
-   **Orientation Quaternion** `[qw, qx, qy, qz]`
-   **Angular Velocity** `[ωx, ωy, ωz]`

### The Dynamics Function

The `rocket_dynamics(t, state)` function calculates the derivative of the state vector. It does this by:
1.  Calculating the rocket's current mass, which decreases as propellant is burned.
2.  Determining the forces acting on the rocket: gravity (in the world frame) and thrust (in the rocket's body frame).
3.  Rotating the thrust vector into the world frame using quaternions.
4.  Summing the forces and using Newton's second law (`F=ma`) to find the linear acceleration.
5.  Calculating the derivative of the quaternion based on the angular velocity (which is zero in this phase).
6.  Returning the derivatives of position, velocity, and orientation.

This derivative function is passed to `scipy.integrate.solve_ivp`, which numerically integrates the equations of motion to find the rocket's state at each time step.
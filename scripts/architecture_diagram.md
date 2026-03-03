# Project Vortex — Detailed Text Architecture Description

Generated: 2026-03-01  
This document describes the internal backend layers of the Vortex simulation suite.  It omits any GUI or user-facing code (PlotVisual, HexaVisual, `gui.py`).

## Entry Layer

- **Configuration files** (`config_ideal.json`, `config_realistic.json`, `config_challenging.json`) contain sections for rocket parameters, environment settings, simulation options, fault definitions and ML setup.
- **main.py** acts as the process entry point. It attempts to import `tkinter` and if successful delegates to the GUI; failing that it falls back to the CLI.
- **cli.py** parses command‑line arguments (`--mode single|optimize`, `--config`, `--mc-runs`, `--search-range`, `--tvc-mode`, etc.) and loads the chosen JSON config.  Two primary operations are provided:
  - `run_single_simulation()`: create a simulation object (plain or enhanced) and execute a single flight, saving results to `results/single_run_<ts>/`.
  - `run_optimization()`: build a simulation and run a Monte Carlo search/optimization for ignition altitude, writing trial data to `results/optimization_<ts>/`.

The CLI also exposes an interactive menu for editing config sections and launching runs.

## Orchestration Layer (`simulation_wrapper.py`)

- **EnhancedSimulation** wraps a `SuicideBurnSimulation` instance. During initialization it may construct a `FaultInjectionManager` (from fault groups) and an `MLFlightComputer` (from ML config).
- Two core methods of the underlying simulation are monkey‑patched before each run:
  - `state_derivative()`: the wrapper checks for active faults, modifies the state (mass, drag, thrust, wind) accordingly, then calls the original derivative.
  - `calculate_ignition_altitude()`: the wrapper obtains the analytic baseline and then applies any ML‑predicted correction from the flight computer.
- `run_simulation()` handles patching, resets fault/ML state, invokes the core simulation, restores the original methods, and appends `fault_log` and `ml_correction_log` to the returned history.
- The class also exposes `check_feasibility()` (pass‑through) and factory methods (`from_config_file`).

The wrapper enables fault injection and ML assistance without modifying core physics code.

## Core Simulation (`simulation.py`)

- **SuicideBurnSimulation** encapsulates the 6‑degree‑of‑freedom landing model.
- The **state vector** comprises 14 elements:
  `[x,y,z, vx,vy,vz, qw,qx,qy,qz, ωx,ωy,ωz, mass]`.
- Initialization builds a `PhysicsEngine` and a `SolidMotor`, loads configuration for rocket mass, inertia, PID gains, TVC mode, sensor error models, ascent/descent settings, and Monte Carlo variation parameters.
- Key routines:
  - `check_feasibility()`: compares ΔV and thrust-to-weight ratio to initial conditions.
  - `calculate_ignition_altitude()`: analytic estimate using terminal velocity and iterative refinement based on expected drag.
  - `run_ascent_phase()`: optional ascent integration to apogee, uses an event to detect `vz ≤ 0`.
  - `tvc_controller()`: stateless PID producing pitch/yaw commands for gimbal actuation (orientation or velocity mode).
  - `state_derivative()`: computes forces (gravity, drag, thrust), rotational EOM, quaternion kinematics, and mass depletion; called by `solve_ivp`.
  - `run_simulation()`: orchestrates the full flight: freefall with `ignition_event`, active burn, and ground contact with `ground_event`. It returns success flag, final state, and history time series.
  - `optimize_ignition_altitude()` and `optimize_ignition_altitude_adaptive()`: two Monte Carlo search routines for finding the best ignition altitude.
- **Events** used by RK45:
  - `ignition_event`: triggers when nozzle altitude equals sensed ignition altitude.
  - `ground_event`: when nozzle altitude ≤ 0.
  - `apogee_event`: vertical velocity crosses zero (used in ascent phase).
- **Success metrics**: final altitude < 1 m, vertical speed magnitude < 2 m/s, total speed < 3 m/s.
- The simulation emits a history dictionary with state components and meta fields (`success`, `final_velocity`, etc.) used later by CLI and visualization.

## Physics Layer

- **PhysicsEngine** (`physics_engine.py`) is a stand‑alone utility providing atmospheric models and quaternion mathematics.
  - Air density: exponential barometric formula with optional random variation.
  - Wind velocity: constant, altitude‑varying power law, or gusts sin/cos wave.
  - Drag force: ½ ρ v_rel² Cd A_ref.
  - Quaternion operations: multiplication, normalization, conversion to/from Euler, rotation matrix.

- **SolidMotor** (`solid_motor.py`) represents the propulsion module.
  - Contains a thrust curve interpolator (built from config data) and samples Monte Carlo variation factors at init.
  - Methods to ignite and check burning status.
  - Computes thrust magnitude and mass flow rate, both subject to stochastic variation.
  - TVC (thrust vector control) logic: command clamping, first‑order lag, and transformation from body to inertial frame for thrust vector and moment.

The simulation repeatedly queries both objects each integrator step.

## Fault Injection Layer (`faults.py`)

- Defines enumerations for `FaultType`, `TriggerMode`, and `FaultTarget` and dataclasses `FaultConfig` and `FaultGroup`.
- **FaultInjectionManager** tracks fault schedules and active effects.
  - `detect_apogee()` marks the time of apogee.
  - `check_triggers()` evaluates triggers (absolute time, time since apogee, altitude threshold, manual) and activates faults probabilistically.
  - Activated faults may randomize their applied magnitude; deactivation occurs after configured duration.
  - `apply_faults_to_state()` adjusts the simulation state (mass loss) and provides multipliers for thrust/drag and wind modifications.
  - Sensor‑only faults can be applied to measurement dictionaries.
  - `get_status_summary()` returns bookkeeping data used by the CLI/GUI for diagnostics.
- Standalone functions compute **fault intensity** using Hill functions and combine multiple intensities with probabilistic OR logic.

The manager is created by `EnhancedSimulation` and invoked inside the patched state derivative.

## ML Flight Computer Layer

- **StateEstimator** is a simple 2‑state EKF estimating vehicle mass and drag coefficient from acceleration, velocity, altitude, and thrust.
- **MLFlightComputer** loads a neural network (Keras or TFLite) and an optional scaler.
  - Periodically (configurable `update_interval`) it extracts a 26‑feature vector from the current state, physics and motor objects, and apogee time. Features include baseline ignition altitude, TWR, velocities, attitude, inferred mass/drag, air density, wind speed, dynamic pressure, time since apogee, and remaining burn time.
  - The network predicts a correction to the ignition altitude. The prediction is cached in `correction_history`.
  - `update()` returns the computed correction when due; `reset()` clears history.

`EnhancedSimulation` wraps the ignition altitude calculation to add this ML correction.

## Output / Results Layer

- Simulation results are written under `results/` with timestamped directories containing CSV logs and plotted PNGs.
- **VortexDataStore** (`plugins/data_store.py`) maintains an in‑memory registry of pandas DataFrames keyed by dataset name (e.g. `legacy`, `trajectory`, `optimization`, `monte_carlo`, or plugin‑specific keys). It can load entire directories, automatically detecting file types.
- The data store is the bus used by the visualization application (PlotVisual) and by any extension graphs.

## Plugin and Extension Layer

- **PlotVisualPlugin** (abstract class in `plugins/base.py`) defines the interface for visualization plugins: graph definitions, rendering, and sample data generation.
- **plugins/loader.py** discovers directory‑based plugins, imports them, and instantiates any `PlotVisualPlugin` subclasses.
- The newer extension framework is defined under `extensions/`:
  - **VortexExtension** abstract base class declares activation, settings schema, and metadata.
  - **ExtensionManager** scans `extensions/installed/` for manifest.json files, handles installs/uninstalls (including `.vortexext` archives), toggles enable/disable state, persists state to `extensions/state.json`, and loads extension classes (including wrapping legacy plugins via `LegacyPluginAdapter`).

Plugins/extensions receive a reference to the `VortexDataStore` during activation.

---

This plain‑text description replaces the original Mermaid diagram and is suitable for inclusion in documentation or code comments.

## Layer Summary

| Layer | Files | Role |
|---|---|---|
| Entry | `main.py`, `cli.py`, JSON configs | Argument parsing, config loading, run dispatch |
| Orchestration | `simulation_wrapper.py` | Non-invasive monkey-patching; wires faults + ML onto core |
| Core Simulation | `simulation.py` | 6DOF EOM, RK45 integration, ignition logic, Monte Carlo |
| Physics | `physics_engine.py`, `solid_motor.py` | Aerodynamics, atmosphere, quaternion math, TVC dynamics |
| Fault Injection | `faults.py` | Trigger/deactivation engine, Hill-function intensity model |
| ML Flight Computer | `ml_flight_computer.py`, `state_estimator.py` | EKF mass/Cd estimation, Keras/TFLite ignition-alt correction |
| Output | `results/`, `plugins/data_store.py` | CSV/PNG persistence, in-memory data bus |
| Plugin/Extension | `plugins/`, `extensions/` | Discoverable graph plugins, manifest-based extension system |

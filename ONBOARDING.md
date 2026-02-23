# Project Vortex — Onboarding Document

**Authors:** Agastya Mishra & Kabir Singh, duPont Manual High School
**Category:** ETSD (Engineering Technology, Statics & Dynamics)
**Competition:** ISEF Regional → State → International
**Last Updated:** February 2026

---

## What Is This Project?

Project Vortex is a **Science Fair project** aiming to solve a problem nobody has publicly solved: **propulsively landing a model rocket using solid rocket motors (SRMs).**

Solid motors are cheap, simple, and reliable — but unlike SpaceX's Merlin engines, they **cannot throttle**. Once ignited, they burn at a fixed thrust until empty. This makes propulsive landing a pure **timing problem**: ignite the retro-burn too early, you hover and tip over; too late, you crater. There is a single optimal ignition altitude with very little margin for error.

The project has two major components:

1. **A high-fidelity 6DOF simulation framework** ("digital twin") that models the complete flight dynamics of a solid-motor rocket — from launch through ballistic descent and powered landing.
2. **A machine learning correction model** that adjusts the pre-computed ignition altitude in real-time during descent, compensating for wind, mass uncertainty, drag variations, and sensor noise.

The ultimate goal is to determine the optimal retro-burn ignition timing under uncertainty and validate the approach through physical flight tests.

---

## The Core Problem

**Why is this hard?**

A liquid-engine rocket (Falcon 9) can throttle down to ~40% thrust, restart multiple times, and adjust in real-time. A solid motor gives you ONE shot: full thrust for a fixed duration. The entire landing problem reduces to:

> *At what altitude do I ignite the retro-burn motor so that the rocket decelerates to zero velocity exactly at ground level?*

This is complicated by:
- **Motor-to-motor thrust variation** (±5% is typical for commercial SRMs)
- **Wind** (changes descent trajectory and velocity)
- **Mass uncertainty** (propellant residuals, manufacturing tolerances)
- **Drag variation** (atmospheric density, angle of attack changes)
- **Sensor noise** (altimeter and IMU errors)
- **TVC actuator lag** (servo response time affects attitude control)

A few meters of error in ignition altitude can mean the difference between a soft landing and a crash.

---

## System Architecture

### 6DOF Simulation (`simulation.py`, `physics_engine.py`, `solid_motor.py`)

**State vector (14 elements):** `[x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, mass]`

**Physics modeled:**
- Gravity (constant, flat Earth)
- Aerodynamic drag: `F_d = 0.5 * rho * v^2 * Cd * A` with altitude-dependent density
- Wind: constant, altitude-varying (power law), or gust models
- Solid motor thrust with TVC (gimbal pitch/yaw, ±5° default, first-order actuator lag)
- Time-varying mass (propellant consumption)
- Optional dynamic CG and inertia (parallel axis theorem)
- Quaternion attitude representation (no gimbal lock)
- Euler's equation for angular dynamics: `I*w_dot + w x (I*w) = M`

**Integration:** RK45 adaptive (scipy `solve_ivp`), rtol=1e-6, atol=1e-9, max step 0.01s

**Control:** PID controller with separate pitch/yaw channels, anti-windup, velocity-hold mode for lateral drift correction during descent.

### Monte Carlo Optimization (`simulation.py`)

Sweeps ignition altitudes over a grid (typically ±10m around analytical estimate, 0.1m steps), running N simulations at each altitude with randomized parameters:

| Parameter | Default Variation |
|-----------|------------------|
| Thrust magnitude | ±5% |
| TVC response time | ±10% |
| Drag coefficient | ±10% |
| Air density | ±5% |
| Mass flow rate | ±2% |
| Altimeter error | ±1% |

**Success criteria:** Final altitude ≈ 0m, vertical velocity < 2 m/s, total velocity < 3 m/s.

### ML Correction Model (`ml_flight_computer.py`)

**Purpose:** During descent, periodically (every 0.5s) read the current flight state and output a correction (in meters) to the baseline ignition altitude.

**Input features (26):** baseline ignition altitude, ascent TWR, descent velocity, altitude, vertical acceleration, lateral velocity (x/y), lateral position (x/y), Euler angles (pitch/roll/yaw), angular rates (wx/wy/wz), EKF-inferred mass, EKF-inferred drag coefficient, estimated drag area, air density, temperature, wind speed, dynamic pressure, time since apogee, thrust available, burn time remaining, predicted ignition altitude.

**Output:** Single scalar — correction in meters to add to baseline ignition altitude.

**Model:** Keras neural network (70KB), also exported to TFLite (9KB) for embedded deployment on Teensy 4.1.

**Training data:** Generated from simulation runs with various fault conditions (mass loss, thrust variation, drag change, wind gust). ~200KB CSV with thousands of training samples.

### State Estimator (`state_estimator.py`)

Extended Kalman Filter tracking `[mass, drag_coefficient]` in real-time using accelerometer measurements. Provides the ML model with inferred physical parameters that can't be directly measured.

### Fault Injection (`faults.py`, `simulation_wrapper.py`)

Injects realistic faults (mass loss, thrust variation, drag change, wind gust) during simulation to test robustness. Supports absolute time, relative time, altitude threshold, and manual triggers. Faults can be probabilistic, temporary, and chained.

---

## Hardware

- **Airframe:** Model rocket kit (66-74mm tube)
- **Flight computer:** Teensy 4.1
- **IMU:** BNO055 (9-axis)
- **Altimeter:** MPL3115A2
- **Data logging:** I2C FRAM (vibration-resistant)
- **TVC:** 2-axis gimbal with servos
- **Parachute deployment:** MOSFET switch
- **Power:** LiPo battery
- **Telemetry:** Radio transmitter/receiver
- **GPS:** Module TBD

One ascent-only flight test has been completed and used to validate the simulation's ascent phase.

---

## Codebase Layout

```
Project-Vortex/
├── simulation.py          # Main sim engine + Monte Carlo optimization (1218 lines)
├── physics_engine.py      # 6DOF physics (136 lines)
├── solid_motor.py         # Motor + TVC model (226 lines)
├── ml_flight_computer.py  # ML correction wrapper (377 lines)
├── state_estimator.py     # EKF for mass/drag (118 lines)
├── faults.py              # Fault injection system (600+ lines)
├── simulation_wrapper.py  # Fault-aware sim wrapper (304 lines)
├── PlotVisual.py          # Visualization app with plugin system
├── plugins/               # PlotVisual plugin directory
│   └── claude_graphs/     # ISEF-focused graph plugin
├── gui.py                 # Tkinter GUI (680+ lines)
├── cli.py                 # CLI interface (380+ lines)
├── ML/run_1/              # Trained ML model + scalers + training data
├── configs/               # JSON configuration files (ideal/realistic/challenging)
├── results/               # Timestamped simulation outputs (CSV + PNG)
└── scripts/               # Analysis utilities
```

---

## Data Formats

**Single run CSV** (`results/single_run_*/single_run.csv`):
```
Time, X, Y, Z, VX, VY, VZ, QW, QX, QY, QZ, Mass
```
High-frequency trajectory data (~500-2000 RK45 steps per simulation).

**Optimization CSV** (`results/optimization_*/optimization.csv`):
```
Ignition Altitude (m), Success Rate
```
One row per tested altitude, success rate from 0.0 to 1.0.

**Trial CSVs** (`results/optimization_*/trials/*.csv`):
Individual trajectory data for each Monte Carlo trial within an optimization run.

**ML training data** (`ML/run_1/correction_training_data.csv`):
```
rocket_class, ascent_twr, descent_velocity, current_altitude, inferred_mass,
inferred_drag_coeff, wind_speed, baseline_ignition_altitude, fault_type, TARGET_correction
```

---

## What's Been Done

- [x] Full 6DOF simulation engine with quaternion dynamics
- [x] Solid motor model with TVC (proper 3D rotation matrices)
- [x] PID attitude controller with anti-windup
- [x] Analytical ignition altitude calculator
- [x] Monte Carlo optimization (grid search over altitudes)
- [x] ML correction model (Keras + TFLite, 26 features → 1 correction)
- [x] Extended Kalman Filter for mass/drag estimation
- [x] Fault injection system (4 fault types, multiple trigger modes)
- [x] GUI (Tkinter) and CLI interfaces
- [x] Physical rocket built and flown (ascent only, parachute recovery)
- [x] Avionics partially tested (Teensy, Radio, IMU, voltage reg, temp sensor)

## What's Remaining

- [ ] Full simulation validation plot (sim vs. flight test data overlay)
- [ ] Monte Carlo cliff plot (success rate vs. ignition altitude) for presentation
- [ ] ML before/after comparison (quantified improvement)
- [ ] Sensitivity analysis (tornado chart of parameter importance)
- [ ] CFD analysis (Kabir's responsibility)
- [ ] Static fire TVC test with data
- [ ] Science fair board and presentation materials
- [ ] Regional competition (~March 2026)

---

## Key Design Decisions

1. **Quaternions over Euler angles** — avoids gimbal lock, essential for 6DOF
2. **RK45 adaptive integration** — handles the stiff dynamics near ground impact
3. **Velocity-hold TVC mode** — tilts into lateral velocity to correct drift, not just maintain orientation
4. **EKF for state estimation** — provides real-time mass/drag estimates that feed the ML model
5. **TFLite export** — enables eventual deployment on Teensy 4.1 microcontroller
6. **Grid search over ignition altitudes** — brute-force but robust; the search space is small enough (~200 altitudes × 100 trials = 20,000 sims)

---

---

## For AI Assistants

When working on this project, key context:
- The simulation outputs CSV files in `results/` with timestamped directories
- Single runs produce trajectory data (Time, X, Y, Z, VX, VY, VZ, QW, QX, QY, QZ, Mass)
- Optimization runs produce `optimization.csv` (Ignition Altitude, Success Rate) plus individual trial CSVs
- The ML model is a regression model: 26 features → 1 scalar correction (meters)
- All physics are in SI units (meters, seconds, kilograms, Newtons)
- The coordinate system is: X=East, Y=North, Z=Up (right-handed)
- "Success" means: landed within 0.5m altitude, <2 m/s vertical velocity, <3 m/s total velocity

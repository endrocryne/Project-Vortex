# RocketPy Integration Extension for Project Vortex

> **Universal Vortex Extension** — works across HexaKinetic (GUI), PlotVisual (graphs), HexaVisual (3D), and TextKinetic (CLI).

## Overview

This extension replaces the built-in Vortex `SuicideBurnSimulation` with a full RocketPy-backed simulation that uses the same 14-element state vector, the same CSV output format, and the same results directory conventions.  It adds Monte Carlo, multi-parameter grid search, advanced graph types, and 3D overlays.

## Features

| Feature | Description |
|---------|-------------|
| **Single Run** | Full RocketPy `Flight` with state conversion to Vortex format |
| **Monte Carlo** | N‑iteration batch with random perturbations (thrust, mass, drag, density) |
| **Grid Search** | Cartesian sweep over any combination of config parameters |
| **Optimization** | Ignition-altitude sweep (coarse → fine adaptive) |
| **PlotVisual Graphs** | 6 new graph types (trajectory, stability, dispersion, comparison, velocity, heatmap) |
| **HexaVisual Overlays** | 4 overlays (apogee marker, trajectory comparison, rail departure, landing zone) + flight stats HUD |
| **Config Validation** | Pre-flight config checker with actionable error/warning messages |
| **CSV Compatibility** | Output identical to native Vortex: `Time,X,Y,Z,VX,VY,VZ,QW,QX,QY,QZ,Mass` |

## Installation

```bash
# From the Vortex Extension Manager (GUI):
#   Settings → Extensions → Install from file → select rocketpy_integration.vortexext

# Or via Python:
python -c "
from extensions.manager import ExtensionManager
mgr = ExtensionManager()
mgr.install_from_vortexext('rocketpy_integration.vortexext')
"
```

### Prerequisites

```
pip install rocketpy numpy pandas matplotlib
# Optional for HexaVisual overlays:
pip install pyvista
```

## File Structure

```
rocketpy_integration/
├── manifest.json          # Extension metadata (universal type)
├── extension.py           # Main entry point — VortexExtension subclass
├── cli_wrapper.py         # Standalone CLI (single / optimize / monte_carlo / grid_search / validate)
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── adapters.py        # Vortex config ↔ RocketPy object conversion
│   ├── data_converter.py  # Flight → CSV, DataFrame, results folders
│   ├── rocketpy_sim.py    # RocketPySimulation (drop-in for SuicideBurnSimulation)
│   ├── monte_carlo.py     # MonteCarloRunner & MonteCarloSweep
│   ├── grid_search.py     # GridSearch with nested-key parameter access
│   ├── validator.py       # Config validation rules
│   ├── plotvisual.py      # PlotVisual graph definitions + renderers
│   └── hexavisual.py      # HexaVisual overlay/HUD definitions + renderers
├── templates/
│   ├── rocketpy_default.json       # Standard config template
│   └── rocketpy_high_power.json    # High-power L-class with solid motor, fins, parachutes
├── tests/
│   ├── __init__.py
│   ├── test_adapters.py
│   ├── test_data_converter.py
│   ├── test_validator.py
│   └── test_grid_search.py
└── README.md              # This file
```

## Quick Start

### CLI — Single Run

```bash
cd scripts/build_rocketpy_ext/rocketpy_integration
python cli_wrapper.py --mode single --config templates/rocketpy_default.json
```

### CLI — Monte Carlo

```bash
python cli_wrapper.py --mode monte_carlo --config templates/rocketpy_default.json --num-mc 200
```

### CLI — Grid Search

```bash
python cli_wrapper.py --mode grid_search \
  --config templates/rocketpy_default.json \
  --grid '{"rocket.dry_mass": [40, 50, 60], "environment.air_density": [1.0, 1.225, 1.4]}'
```

### CLI — Validate Config

```bash
python cli_wrapper.py --mode validate --config templates/rocketpy_high_power.json
```

### Inside Vortex (after installing the .vortexext)

The extension registers automatically with all Vortex apps:

- **HexaKinetic**: New "RocketPy" motor profiles, "Run with RocketPy" option, grid search panel.
- **PlotVisual**: 6 new graph items in the sidebar.
- **HexaVisual**: 4 overlays and a flight-stats HUD widget.

## Parameter Mapping Reference

### Rocket Parameters

| Vortex Key | RocketPy Usage | Unit |
|---|---|---|
| `rocket.dry_mass` | `Rocket(mass=...)` | kg |
| `rocket.propellant_mass` | `GenericMotor(propellant_initial_mass=...)` | kg |
| `rocket.diameter` | `Rocket(radius=d/2)` | m |
| `rocket.length` | Nose cone position | m |
| `rocket.thrust_curve` | `GenericMotor(thrust_source=...)` | [[s, N], ...] |
| `rocket.burn_time` | `GenericMotor(burn_time=...)` | s |
| `rocket.drag_coefficient` | `Rocket.set_rail_buttons()` power-off/on drag | – |
| `rocket.ascent_motor_casing_mass` | `GenericMotor(dry_mass=...)` | kg |

### Environment Parameters

| Vortex Key | RocketPy Usage | Unit |
|---|---|---|
| `environment.gravity` | Default (RocketPy computes from coords) | m/s² |
| `environment.air_density` | Atmosphere model override | kg/m³ |
| `environment.temperature` | Atmosphere model override | K |
| `environment.wind_speed` | `Environment.set_atmospheric_model()` | m/s |
| `environment.wind_direction` | Constant wind bearing | deg |

### RocketPy-Specific Parameters (`rocketpy` block)

| Key | Description | Default |
|---|---|---|
| `rail_length` | Launch rail length | 5.0 m |
| `inclination` | Rail inclination from horizontal | 90° |
| `heading` | Rail heading (0 = North) | 0° |
| `atmospheric_model` | `"standard_atmosphere"`, `"Forecast"`, etc. | `"standard_atmosphere"` |
| `motor_type` | `"generic"` or `"solid"` | `"generic"` |
| `grains` | List of grain dicts for SolidMotor | — |
| `nose.kind` | Nose cone type (`"Von Karman"`, `"ogive"`, etc.) | `"Von Karman"` |
| `nose.length` | Nose cone length | 0.5 m |
| `fins.n` | Number of fins | 4 |
| `fins.root_chord` / `tip_chord` / `span` | Fin geometry | m |
| `parachutes` | List of `{name, cd_s, trigger, lag}` | — |
| `rail_buttons` | `{upper_position, lower_position, angular_position}` | — |
| `max_time` | Maximum simulation time | 600 s |

## State Vector Convention

Vortex 14-element:  `[x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz, mass]`

RocketPy Flight integration uses 6DOF when available, falling back to velocity-vector-based quaternion synthesis for 3DOF flights.

## CSV Output Format

All CSVs are fully compatible with the existing Vortex data pipeline:

```
Time,X,Y,Z,VX,VY,VZ,QW,QX,QY,QZ,Mass
0.0,0.0,0.0,1000.0,0.0,0.0,-50.0,1.0,0.0,0.0,0.0,60.0
...
```

## Known Limitations

1. **TVC (Thrust Vector Control)**: RocketPy does not natively support real-time TVC. The extension logs a warning when `tvc_enabled` is set. Gimbal dynamics from `solid_motor.py` are not modeled.
2. **3DOF → 6DOF approximation**: When RocketPy runs in 3DOF mode, quaternions are synthesised from the velocity vector.  Angular velocities are set to zero.
3. **Real-time wind data**: `"Forecast"` atmospheric model requires network access and may fail in offline/CI environments.
4. **Mass curve**: RocketPy tracks mass internally; the adapter reads it back for each timestep, but fuel grain regression detail is lost in the 14-element state vector.

## Running Tests

```bash
cd scripts/build_rocketpy_ext/rocketpy_integration
python -m pytest tests/ -v
# Or without pytest:
python -m unittest discover -s tests -v
```

Tests do **not** run any RocketPy simulations — they use mocking and synthetic data to verify conversion logic, CSV format compliance, validation rules, and grid enumeration.

## Building the .vortexext Package

```bash
cd scripts/build_rocketpy_ext
python build.py
# Creates: scripts/build_rocketpy_ext/rocketpy_integration.vortexext
```

## License

Same as Project Vortex.

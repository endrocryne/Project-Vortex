# Core Simulation Audit Findings

Scope reviewed:
- `/home/runner/work/Project-Vortex/Project-Vortex/physics_engine.py`
- `/home/runner/work/Project-Vortex/Project-Vortex/simulation.py`
- `/home/runner/work/Project-Vortex/Project-Vortex/solid_motor.py`
- `/home/runner/work/Project-Vortex/Project-Vortex/state_estimator.py`

## Confirmed issue fixed

### A) Non-deterministic drag/air-density sampling inside ODE force evaluations

**Observed behavior (before fix):**
- `PhysicsEngine.get_air_density()` sampled random density variation every call.
- `PhysicsEngine.get_drag_force()` sampled random drag coefficient variation every call.

Because RK45 evaluates the derivative function multiple times per step (and can revisit nearly identical states/times), this introduced stochastic noise directly into RHS evaluations within a single run. That is numerically inconsistent for deterministic ODE integration and can bias/unstabilize trajectories.

**Fix applied:**
- Added per-run Monte Carlo resampling in `PhysicsEngine.resample_monte_carlo()`.
- Monte Carlo factors are sampled once and held constant through a run.
- `SuicideBurnSimulation.run_simulation()` now calls `self.physics.resample_monte_carlo()` at start of each run.
- Added atmospheric altitude clamp (`altitude = max(0.0, altitude)`) in density query.

**Files changed:**
- `/home/runner/work/Project-Vortex/Project-Vortex/physics_engine.py`
- `/home/runner/work/Project-Vortex/Project-Vortex/simulation.py`
- `/home/runner/work/Project-Vortex/Project-Vortex/test_features.py` (regression validation)

## RocketPy comparison note

Attempted runtime comparison against the included RocketPy integration wrapper. Comparison execution is currently blocked by an adapter/runtime mismatch:

- `TypeError: GenericMotor.__init__() missing 3 required positional arguments: 'chamber_radius', 'chamber_height', and 'chamber_position'`
- Location: `scripts/build_rocketpy_ext/rocketpy_integration/core/adapters.py` during `RocketPySimulation.run_simulation()`.

This indicates current extension code targets a different RocketPy constructor signature than the installed `rocketpy` version. No change was made here to keep scope minimal and focused on core simulator physics.

## Validation performed

1. Baseline smoke test:
   - `python test_features.py` (pass)
2. Post-fix smoke test:
   - `python test_features.py` (pass, including new deterministic-drag regression check)
3. CLI single-run checks with custom non-repo configs:
   - `python cli.py --mode single --auto --config /tmp/project_vortex_case_a.json` (pass)
   - `python cli.py --mode single --auto --config /tmp/project_vortex_case_b.json --ascent` (pass)
4. CLI optimization checks with custom non-repo configs:
   - adaptive optimization (short and longer runs) completed successfully.

The resulting trajectories remain physically consistent with scenario feasibility messages (many tested cases intentionally configured as underpowered for full landing).

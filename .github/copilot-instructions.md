# Copilot / AI Agent Instructions for Project Vortex 🔧

Purpose: Provide focused, actionable context so an AI coding agent can be productive immediately in this repo.

## Quick orientation ✅
- Entry points: `main.py` (auto-GUI fallback), `gui.py` (Tkinter), `cli.py` (CLI always available).
- Core simulation implementation: `simulation.py` (high-level), `physics_engine.py` (6DOF), `solid_motor.py` (thrust/TVC), `state_estimator.py` (EKF for mass/Cd).
- Config files live at repo root: `config_ideal.json`, `config_realistic.json`, `config_challenging.json`.

## Quick commands (local dev) ▶️
- Install deps: `pip install -r requirements.txt` (Python 3.7+)
- Smoke test: `python test_features.py` (fast verification of major components)
- Single-run demo: `python cli.py --mode single`
- Full optimization: `python cli.py --mode optimize`

## Important project conventions & patterns 📏
- Units: SI (meters, seconds, kg). State vector is 14 elements: [x,y,z,vx,vy,vz,qw,qx,qy,qz,ωx,ωy,ωz,mass].
- No interactive plot windows: plotting uses `matplotlib` with backend `Agg` and saves PNGs to `results/` (timestamped folders).
- Config-driven: almost all experiment parameters come from JSON configs and are passed into `SuicideBurnSimulation`.
- Numerical settings: RK45 adaptive integrator with tight tolerances (see README/IMPLEMENTATION_SUMMARY). Keep changes small and test with `test_features.py`.

## Where to make common changes 🛠️
- Physics / dynamics: `physics_engine.py` (drag, atmosphere, quaternion integration).
- Control logic / ignition calc: `simulation.py` (functions: `calculate_ignition_altitude`, `run_simulation`, Monte Carlo logic).
- Motor/TVC: `solid_motor.py` (thrust curve, gimbal dynamics).
- State estimation experiments: `state_estimator.py` (EKF) and `ML/` (models & training data for offline correction).

## ML integration notes 🤖
- ML artifacts are in `ML/run_1` and `ML/run_2`: `correction_model.keras`, `correction_model.tflite`, and `.csv` training data.
- The repo includes `ML_Model_Test.ipynb` to inspect / load these models — scripts do not auto-load TF models in the main simulation by default.

## Tests & CI guidance ✅
- Use `python test_features.py` as a deterministic smoke test (no pytest harness currently).
- CI should: install `requirements.txt`, run `python -m matplotlib` or set `MPLBACKEND=Agg`, run `python test_features.py`, and ensure a `results/` artifact is created.

## Common pitfalls & tips ⚠️
- For headless CI, set Matplotlib backend to `Agg` or set `matplotlib.use('Agg')` before importing `pyplot` to avoid GUI errors.
- GUI depends on `tkinter`; if unavailable, `main.py` falls back to the CLI—tests expect CLI behavior.
- When changing numerical tolerances or time steps, re-run Monte Carlo jobs and `test_features.py` to check performance & stability.

## Examples to reference in PRs / patches 📌
- Changing ignition logic: update `calculate_ignition_altitude` and add unit-style checks in `test_features.py`.
- Adding a new wind model: implement in `physics_engine.py`, add a small test case in `test_features.py`, and document config options.

## Final notes & etiquette ✨
- Keep changes minimal and well-tested: this repo has many coupled physics assumptions; small numerical changes can have outsized effects.
- Prefer adding a focused test to `test_features.py` for behavior changes.

— Please review this file and tell me if any sections need more examples or you'd like me to add a short checklist for PR reviewers.

## Strict no-placeholders policy ❗

NEVER EVER IN A MILLION YEARS SHOULD YOU EVER EVER PUT ANY PLACEHOLDERS, "SAME AS BEFORE", SIMPLIFIED OR ANYTHING ELSE. NOTHING BUT THE FULL CODE! THERE SHOULD NEVER BE ANY "THIS IS WILL BE IMPLEMENTED IN FUTURE"  BULLSHIT. JUST THE FULL CODE. IT DOESN'T MATTER HOW MANY TIMES YOU REWRITE IT OR ANYTHING. ALWAYS THE FULL CODE.

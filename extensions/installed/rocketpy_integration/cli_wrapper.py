"""
CLI Wrapper for RocketPy Integration Extension
=================================================

Allows running RocketPy simulations from the command line with the same
interface as the native Vortex CLI (cli.py).

Usage:
    python -m rocketpy_integration.cli_wrapper --config config_ideal.json --mode single
    python -m rocketpy_integration.cli_wrapper --config config_realistic.json --mode optimize
    python -m rocketpy_integration.cli_wrapper --config config_realistic.json --mode grid_search \\
        --grid "rocket.dry_mass=[40,50,60]" --grid "environment.drag_coefficient=[0.3,0.5,0.7]"

Modes:
    single      — Single RocketPy flight, outputs trajectory CSV + plots
    optimize    — Monte Carlo ignition-altitude sweep (cliff plot data)
    monte_carlo — Flat Monte Carlo batch (no altitude sweep)
    grid_search — Multi-parameter grid search
    validate    — Validate config for RocketPy compatibility (no simulation)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Dict, List

import numpy as np

# Ensure the extension package and project root are importable
_ext_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_ext_dir)))
for p in (_ext_dir, _project_root, os.path.dirname(_ext_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.adapters import HAS_ROCKETPY
from core.rocketpy_sim import RocketPySimulation
from core.data_converter import (
    save_trajectory_csv,
    save_optimization_csv,
    save_config,
    make_results_folder,
    generate_trajectory_plots,
    flight_to_history,
)
from core.monte_carlo import MonteCarloRunner, save_monte_carlo_results
from core.grid_search import GridSearch
from core.validator import validate_config, format_validation_report


def load_config(path: str) -> dict:
    """Load a Vortex JSON config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_initial_state(config: dict, altitude: float = 1000.0,
                        velocity: float = -50.0) -> np.ndarray:
    """Build the 14-element Vortex initial state vector from config."""
    rc = config.get("rocket", {})
    mass = rc.get("dry_mass", 50.0) + rc.get("propellant_mass", 10.0)
    return np.array([
        0.0, 0.0, altitude,     # position
        0.0, 0.0, velocity,     # velocity
        1.0, 0.0, 0.0, 0.0,    # quaternion (identity)
        0.0, 0.0, 0.0,          # angular velocity
        mass,                    # mass
    ])


def parse_grid_spec(specs: List[str]) -> Dict[str, list]:
    """
    Parse --grid arguments like "rocket.dry_mass=[40,50,60]"
    into a dict of {param_path: [values]}.
    """
    grid = {}
    for s in specs:
        if "=" not in s:
            print(f"Invalid grid spec (expected 'param=values'): {s}")
            continue
        key, val_str = s.split("=", 1)
        try:
            values = ast.literal_eval(val_str)
            if not isinstance(values, (list, tuple)):
                values = [values]
            grid[key.strip()] = list(values)
        except Exception as exc:
            print(f"Could not parse values for '{key}': {exc}")
    return grid


def cmd_single(config: dict, args: argparse.Namespace) -> None:
    """Run a single RocketPy flight."""
    print("=" * 60)
    print("RocketPy Integration — Single Flight")
    print("=" * 60)

    folder, ts = make_results_folder("single_run")
    save_config(config, os.path.join(folder, "config.json"))

    state = build_initial_state(config, args.altitude, args.velocity)
    sim = RocketPySimulation(
        config.get("rocket", {}),
        config.get("environment", {}),
        config.get("simulation", {}),
    )

    print("Running RocketPy simulation...")
    success, final_state, history = sim.run_simulation(state, 0.0)

    csv_path = save_trajectory_csv(history, os.path.join(folder, "single_run.csv"))
    print(f"Trajectory CSV: {csv_path}")

    plot_files = generate_trajectory_plots(history, folder)
    for pf in plot_files:
        print(f"Plot: {pf}")

    print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")
    print(f"  Final altitude: {history['final_altitude']:.2f} m")
    print(f"  Final velocity: {history['final_velocity']:.2f} m/s")
    print(f"  Apogee: {history.get('apogee_altitude', 0):.1f} m")
    print(f"\nOutputs saved to: {folder}")


def cmd_optimize(config: dict, args: argparse.Namespace) -> None:
    """Run Monte Carlo ignition-altitude sweep."""
    print("=" * 60)
    print("RocketPy Integration — Ignition Altitude Optimization")
    print("=" * 60)

    folder, ts = make_results_folder("optimization")
    save_config(config, os.path.join(folder, "config.json"))

    state = build_initial_state(config, args.altitude, args.velocity)
    sim = RocketPySimulation(
        config.get("rocket", {}),
        config.get("environment", {}),
        config.get("simulation", {}),
    )

    sc = config.get("simulation", {})
    optimal, rates, best_hist = sim.optimize_ignition_altitude(
        state,
        num_monte_carlo=int(sc.get("num_monte_carlo", args.num_mc)),
        altitude_search_range=float(sc.get("altitude_search_range", 10.0)),
        altitude_step=float(sc.get("altitude_step", 0.5)),
        save_each_trial=args.save_trials,
        results_folder=folder,
    )

    opt_csv = save_optimization_csv(rates, os.path.join(folder, "optimization.csv"))
    print(f"\nOptimization CSV: {opt_csv}")

    if best_hist:
        save_trajectory_csv(best_hist, os.path.join(folder, "rocketpy_trajectory.csv"))

    print(f"\nOptimal ignition altitude: {optimal:.2f} m")
    print(f"Outputs saved to: {folder}")


def cmd_monte_carlo(config: dict, args: argparse.Namespace) -> None:
    """Flat Monte Carlo batch."""
    print("=" * 60)
    print("RocketPy Integration — Monte Carlo Analysis")
    print("=" * 60)

    folder, ts = make_results_folder("monte_carlo")
    state = build_initial_state(config, args.altitude, args.velocity)

    runner = MonteCarloRunner(config)
    summary = runner.run(
        state,
        num_trials=args.num_mc,
        save_each_trial=args.save_trials,
        results_folder=folder,
    )

    save_monte_carlo_results(summary, folder, config)

    print(f"\nMonte Carlo Results:")
    print(f"  Trials: {summary['num_trials']}")
    print(f"  Successes: {summary['successes']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Mean Landing Velocity: {summary['mean_landing_velocity']:.2f} m/s")
    print(f"  Std Landing Velocity: {summary['std_landing_velocity']:.2f} m/s")
    print(f"\nOutputs saved to: {folder}")


def cmd_grid_search(config: dict, args: argparse.Namespace) -> None:
    """Multi-parameter grid search."""
    print("=" * 60)
    print("RocketPy Integration — Grid Search")
    print("=" * 60)

    if not args.grid:
        print("ERROR: --grid argument(s) required for grid_search mode.")
        print("Example: --grid 'rocket.dry_mass=[40,50,60]'")
        sys.exit(1)

    param_grid = parse_grid_spec(args.grid)
    if not param_grid:
        print("ERROR: No valid grid specifications parsed.")
        sys.exit(1)

    print(f"Parameters: {list(param_grid.keys())}")
    total_points = 1
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
        total_points *= len(v)
    print(f"Total grid points: {total_points}")
    print(f"MC trials per point: {args.num_mc}")
    print(f"Total simulations: {total_points * args.num_mc}")

    folder, ts = make_results_folder("grid_search")
    state = build_initial_state(config, args.altitude, args.velocity)

    gs = GridSearch(config)
    results = gs.run(
        initial_state=state,
        param_grid=param_grid,
        num_monte_carlo_per_point=args.num_mc,
        results_folder=folder,
    )

    print(f"\nGrid Search Complete — {len(results)} points evaluated")
    best = max(results, key=lambda r: r.get("success_rate", 0))
    print(f"Best configuration (success rate {best['success_rate']:.1%}):")
    for k in param_grid:
        print(f"  {k} = {best.get(k)}")
    print(f"\nOutputs saved to: {folder}")


def cmd_validate(config: dict, args: argparse.Namespace) -> None:
    """Validate configuration without running a simulation."""
    print("=" * 60)
    print("RocketPy Integration — Configuration Validation")
    print("=" * 60)
    ok, msgs = validate_config(config)
    report = format_validation_report(msgs)
    print(report)
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(
        description="RocketPy Integration CLI for Project Vortex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="config_ideal.json",
                        help="Path to Vortex JSON config file")
    parser.add_argument("--mode", "-m", default="single",
                        choices=["single", "optimize", "monte_carlo", "grid_search", "validate"],
                        help="Simulation mode")
    parser.add_argument("--altitude", type=float, default=1000.0,
                        help="Initial altitude (m) for descent simulations")
    parser.add_argument("--velocity", type=float, default=-50.0,
                        help="Initial vertical velocity (m/s, negative = descending)")
    parser.add_argument("--num-mc", type=int, default=50,
                        help="Number of Monte Carlo trials per point")
    parser.add_argument("--save-trials", action="store_true",
                        help="Save individual trial CSVs")
    parser.add_argument("--grid", action="append", default=[],
                        help="Grid search parameter spec: 'param.path=[v1,v2,v3]' (repeatable)")

    args = parser.parse_args()

    if not HAS_ROCKETPY:
        print("ERROR: rocketpy is not installed.")
        print("Install with:  pip install rocketpy")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    dispatch = {
        "single": cmd_single,
        "optimize": cmd_optimize,
        "monte_carlo": cmd_monte_carlo,
        "grid_search": cmd_grid_search,
        "validate": cmd_validate,
    }

    dispatch[args.mode](config, args)


if __name__ == "__main__":
    main()

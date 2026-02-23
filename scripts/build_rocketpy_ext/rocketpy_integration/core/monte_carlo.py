"""
Monte Carlo Analysis for RocketPy Simulations
================================================

Runs batches of RocketPy flights with randomised physical parameters and
collects statistics compatible with Vortex's cliff-plot and data-store
pipelines.
"""

from __future__ import annotations

import copy
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from .rocketpy_sim import RocketPySimulation
    from .data_converter import (
        save_trajectory_csv,
        save_optimization_csv,
        save_trial_csv,
        save_config,
        save_grid_search_csv,
        make_results_folder,
        generate_trajectory_plots,
    )
except ImportError:
    from core.rocketpy_sim import RocketPySimulation
    from core.data_converter import (
        save_trajectory_csv,
        save_optimization_csv,
        save_trial_csv,
        save_config,
        save_grid_search_csv,
        make_results_folder,
        generate_trajectory_plots,
    )


class MonteCarloRunner:
    """
    Runs Monte Carlo batches with a ``RocketPySimulation`` backend.

    Usage:
        runner = MonteCarloRunner(config)
        results = runner.run(initial_state, num_trials=200,
                             progress_callback=my_callback)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.sim = RocketPySimulation(
            config.get("rocket", {}),
            config.get("environment", {}),
            config.get("simulation", {}),
        )

    def run(
        self,
        initial_state: np.ndarray,
        num_trials: int = 100,
        ignition_altitude: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        save_each_trial: bool = False,
        results_folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run *num_trials* simulations with randomised parameters.

        Returns a summary dict:
            {
                'num_trials': int,
                'successes': int,
                'success_rate': float,
                'landing_velocities': list[float],
                'mean_landing_velocity': float,
                'std_landing_velocity': float,
                'apogee_altitudes': list[float],
                'histories': list[dict]  (only successes if save_each_trial=False)
            }
        """
        successes = 0
        landing_vels: List[float] = []
        apogees: List[float] = []
        histories: List[Dict[str, Any]] = []

        trials_dir = None
        if save_each_trial and results_folder:
            trials_dir = os.path.join(results_folder, "trials")
            os.makedirs(trials_dir, exist_ok=True)

        for i in range(num_trials):
            try:
                ok, _final, history = self.sim.run_simulation_with_variations(
                    initial_state.copy(), ignition_altitude
                )
            except Exception as exc:
                print(f"[MonteCarloRunner] Trial {i+1} failed: {exc}")
                ok = False
                history = None

            if history is not None:
                landing_vels.append(history.get("final_velocity", float("inf")))
                apogees.append(history.get("apogee_altitude", 0.0))
                if ok:
                    successes += 1
                    histories.append(history)

                if save_each_trial and trials_dir:
                    idx_str = f"{i+1:04d}"
                    status = "success" if ok else "fail"
                    csv_path = os.path.join(trials_dir, f"mc_trial_{idx_str}_{status}.csv")
                    try:
                        save_trial_csv(history, ignition_altitude, csv_path)
                    except Exception:
                        pass

            if progress_callback:
                try:
                    progress_callback(i + 1, num_trials)
                except Exception:
                    pass
            else:
                print(f"\r[RocketPy MC] {i+1}/{num_trials}", end="", flush=True)

        if not progress_callback:
            print()

        landing_arr = np.array(landing_vels) if landing_vels else np.array([0.0])

        return {
            "num_trials": num_trials,
            "successes": successes,
            "success_rate": successes / max(num_trials, 1),
            "landing_velocities": landing_vels,
            "mean_landing_velocity": float(np.mean(landing_arr)),
            "std_landing_velocity": float(np.std(landing_arr)),
            "apogee_altitudes": apogees,
            "mean_apogee": float(np.mean(apogees)) if apogees else 0.0,
            "histories": histories,
        }


class MonteCarloSweep:
    """
    Runs the ignition-altitude sweep (the same pipeline as
    ``SuicideBurnSimulation.optimize_ignition_altitude``) but using RocketPy.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)

    def run(
        self,
        initial_state: np.ndarray,
        num_monte_carlo: int = 100,
        altitude_search_range: float = 10.0,
        altitude_step: float = 0.1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        save_each_trial: bool = False,
        results_folder: Optional[str] = None,
    ) -> Tuple[float, Dict[float, float], Optional[Dict[str, Any]]]:
        """
        Convenience wrapper — delegates to RocketPySimulation.optimize_ignition_altitude().
        """
        sim = RocketPySimulation(
            self.config.get("rocket", {}),
            self.config.get("environment", {}),
            self.config.get("simulation", {}),
        )
        return sim.optimize_ignition_altitude(
            initial_state,
            num_monte_carlo=num_monte_carlo,
            altitude_search_range=altitude_search_range,
            altitude_step=altitude_step,
            progress_callback=progress_callback,
            save_each_trial=save_each_trial,
            results_folder=results_folder,
        )


def save_monte_carlo_results(summary: Dict[str, Any],
                             output_dir: str,
                             config: Optional[Dict[str, Any]] = None) -> str:
    """
    Persist Monte Carlo summary to a results folder.

    Writes:
        monte_carlo_summary.csv, config.json, trajectory PNGs for best history.

    Returns:
        path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save summary CSV
    import csv
    summary_path = os.path.join(output_dir, "monte_carlo_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["num_trials", summary["num_trials"]])
        writer.writerow(["successes", summary["successes"]])
        writer.writerow(["success_rate", summary["success_rate"]])
        writer.writerow(["mean_landing_velocity", summary["mean_landing_velocity"]])
        writer.writerow(["std_landing_velocity", summary["std_landing_velocity"]])
        writer.writerow(["mean_apogee", summary.get("mean_apogee", 0.0)])

    # Save config
    if config:
        save_config(config, os.path.join(output_dir, "config.json"))

    # Save best trajectory
    if summary["histories"]:
        best = summary["histories"][0]
        save_trajectory_csv(best, os.path.join(output_dir, "rocketpy_single_run.csv"))
        try:
            generate_trajectory_plots(best, output_dir)
        except Exception:
            traceback.print_exc()

    return output_dir

"""
Multi-Parameter Grid Search for RocketPy Simulations
======================================================

Sweeps across a cartesian product of parameter values, running Monte Carlo
batches at each grid point.  Produces a CSV with columns for every swept
parameter plus success_rate, mean_landing_velocity, std_landing_velocity.
"""

from __future__ import annotations

import copy
import csv
import itertools
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from .rocketpy_sim import RocketPySimulation
    from .data_converter import (
        save_grid_search_csv,
        save_config,
        make_results_folder,
    )
except ImportError:
    from core.rocketpy_sim import RocketPySimulation
    from core.data_converter import (
        save_grid_search_csv,
        save_config,
        make_results_folder,
    )


# ---------------------------------------------------------------------------
# Parameter path helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated path.

    Example: _set_nested(cfg, "rocket.dry_mass", 55.0)
    """
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot-separated path."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

class GridSearch:
    """
    Multi-parameter grid search with Monte Carlo at each grid point.

    Usage:
        gs = GridSearch(base_config)
        results = gs.run(
            initial_state=state,
            param_grid={
                "rocket.dry_mass": [40, 50, 60],
                "environment.drag_coefficient": [0.3, 0.5, 0.7],
            },
            num_monte_carlo_per_point=50,
        )
    """

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = copy.deepcopy(base_config)

    def run(
        self,
        initial_state: np.ndarray,
        param_grid: Dict[str, List[Any]],
        num_monte_carlo_per_point: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        results_folder: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the grid search.

        Args:
            initial_state: 14-element Vortex state vector
            param_grid: Mapping of dot-path parameter names to lists of values
                        e.g. {"rocket.dry_mass": [40, 50, 60]}
            num_monte_carlo_per_point: Trials per grid point
            progress_callback: fn(current, total)
            results_folder: If set, write grid_search.csv and plots here

        Returns:
            List of dicts, one per grid point, each containing:
                - all param values (keyed by param name)
                - success_rate
                - mean_landing_velocity
                - std_landing_velocity
        """
        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        grid_points = list(itertools.product(*param_values))
        total_runs = len(grid_points) * num_monte_carlo_per_point
        global_run = 0

        results: List[Dict[str, Any]] = []

        for point in grid_points:
            # Build config for this grid point
            cfg = copy.deepcopy(self.base_config)
            point_info: Dict[str, Any] = {}
            for name, value in zip(param_names, point):
                _set_nested(cfg, name, value)
                point_info[name] = value

            # Run Monte Carlo at this grid point
            sim = RocketPySimulation(
                cfg.get("rocket", {}),
                cfg.get("environment", {}),
                cfg.get("simulation", {}),
            )

            successes = 0
            landing_vels: List[float] = []

            for j in range(num_monte_carlo_per_point):
                try:
                    ok, _, history = sim.run_simulation_with_variations(
                        initial_state.copy(), 0.0
                    )
                except Exception:
                    ok = False
                    history = None

                if ok:
                    successes += 1
                if history:
                    landing_vels.append(history.get("final_velocity", float("inf")))

                global_run += 1
                if progress_callback:
                    try:
                        progress_callback(global_run, total_runs)
                    except Exception:
                        pass

            vel_arr = np.array(landing_vels) if landing_vels else np.array([0.0])
            point_info["success_rate"] = successes / max(num_monte_carlo_per_point, 1)
            point_info["mean_landing_velocity"] = float(np.mean(vel_arr))
            point_info["std_landing_velocity"] = float(np.std(vel_arr))
            results.append(point_info)

        # Persist
        if results_folder:
            os.makedirs(results_folder, exist_ok=True)
            save_grid_search_csv(results, param_names,
                                 os.path.join(results_folder, "grid_search.csv"))
            save_config(self.base_config,
                        os.path.join(results_folder, "config.json"))

            # Generate contour / heatmap if 2D
            if len(param_names) == 2 and HAS_MATPLOTLIB:
                try:
                    self._plot_2d_heatmap(results, param_names, results_folder)
                except Exception:
                    traceback.print_exc()

        return results

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_2d_heatmap(results: List[Dict[str, Any]],
                         param_names: List[str],
                         output_dir: str) -> str:
        """Generate a 2D heatmap of success_rate for a 2-parameter grid search."""
        p1, p2 = param_names
        v1_set = sorted(set(r[p1] for r in results))
        v2_set = sorted(set(r[p2] for r in results))

        rate_map = {}
        for r in results:
            rate_map[(r[p1], r[p2])] = r["success_rate"]

        grid = np.zeros((len(v2_set), len(v1_set)))
        for i, vv2 in enumerate(v2_set):
            for j, vv1 in enumerate(v1_set):
                grid[i, j] = rate_map.get((vv1, vv2), 0.0)

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(grid, origin="lower", aspect="auto",
                        extent=[min(v1_set), max(v1_set), min(v2_set), max(v2_set)],
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xlabel(p1.split(".")[-1], fontsize=12)
        ax.set_ylabel(p2.split(".")[-1], fontsize=12)
        ax.set_title("Grid Search — Success Rate Heatmap", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Success Rate")
        fig.tight_layout()

        path = os.path.join(output_dir, "grid_search_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def plot_1d_sweep(results: List[Dict[str, Any]],
                      param_name: str,
                      output_dir: str) -> str:
        """Generate a 1D success-rate curve for a single-parameter sweep."""
        if not HAS_MATPLOTLIB:
            return ""

        vals = [r[param_name] for r in results]
        rates = [r["success_rate"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(vals, rates, "b-o", linewidth=1.5, markersize=4)
        ax.set_xlabel(param_name.split(".")[-1], fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Grid Search — {param_name.split('.')[-1]} Sweep", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(output_dir, "grid_search_sweep.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

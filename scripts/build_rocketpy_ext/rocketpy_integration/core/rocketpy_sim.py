"""
RocketPy Simulation Wrapper
=============================

Provides a ``RocketPySimulation`` class whose public API mirrors
``SuicideBurnSimulation`` so it can be used as a drop-in replacement
inside HexaKinetic (gui.py) and the CLI.

Key methods:
    run_simulation(initial_state, ignition_altitude)  ->  (success, final_state, history)
    optimize_ignition_altitude(...)  ->  (optimal_alt, success_rates, best_history)
"""

from __future__ import annotations

import copy
import json
import math
import os
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from .adapters import VortexConfigAdapter, StateConverter, HAS_ROCKETPY, _safe_float
    from .data_converter import (
        flight_to_history,
        save_trajectory_csv,
        save_optimization_csv,
        save_trial_csv,
        save_config,
        make_results_folder,
        generate_trajectory_plots,
    )
except ImportError:
    from core.adapters import VortexConfigAdapter, StateConverter, HAS_ROCKETPY, _safe_float
    from core.data_converter import (
        flight_to_history,
        save_trajectory_csv,
        save_optimization_csv,
        save_trial_csv,
        save_config,
        make_results_folder,
        generate_trajectory_plots,
    )


class RocketPySimulation:
    """
    Drop-in replacement for ``SuicideBurnSimulation`` that delegates the
    actual physics to RocketPy.

    Constructor signature intentionally mirrors SuicideBurnSimulation so it can
    be swapped in by the GUI without changes.
    """

    def __init__(self, rocket_config: Dict[str, Any],
                 environment_config: Dict[str, Any],
                 simulation_config: Dict[str, Any]):
        if not HAS_ROCKETPY:
            raise ImportError(
                "RocketPy is not installed.  Install it with:  pip install rocketpy"
            )

        self.rocket_config = copy.deepcopy(rocket_config)
        self.environment_config = copy.deepcopy(environment_config)
        self.simulation_config = copy.deepcopy(simulation_config)

        # Merge into the unified config dict that the adapter expects
        self.config: Dict[str, Any] = {
            "rocket": self.rocket_config,
            "environment": self.environment_config,
            "simulation": self.simulation_config,
            "rocketpy": self.simulation_config.get("rocketpy", {}),
        }

        # Variation magnitudes (for Monte Carlo randomisation)
        self.thrust_variation = _safe_float(rocket_config.get("thrust_variation", 0.0))
        self.mass_variation = _safe_float(rocket_config.get("mass_variation", 0.0))
        self.drag_variation = _safe_float(environment_config.get("drag_variation", 0.0))
        self.density_variation = _safe_float(environment_config.get("air_density_variation", 0.0))
        self.tvc_response_variation = _safe_float(rocket_config.get("tvc_response_variation", 0.0))

        # Dry mass + propellant (for initial state construction)
        self.dry_mass = _safe_float(rocket_config.get("dry_mass", 50.0))
        self.propellant_mass = _safe_float(rocket_config.get("propellant_mass", 10.0))

        # Mutable copy for randomisation
        self.history: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Single simulation
    # ------------------------------------------------------------------

    def run_simulation(self, initial_state: np.ndarray,
                       ignition_altitude: float = 0.0) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Run a single RocketPy flight.

        Because RocketPy handles the full flight (rail launch → apogee → descent)
        internally, the *initial_state* and *ignition_altitude* parameters from
        Vortex's suicide-burn model are mapped as follows:

        * The rocket's rail length, inclination, and heading come from the config.
        * *initial_state* is used to set altitude/velocity context when logging,
          but the actual RocketPy Flight starts from the launch rail.

        Returns:
            (success, final_state_14, history_dict)
        """
        adapter = VortexConfigAdapter(self.config)
        env = adapter.build_environment()
        motor = adapter.build_motor()
        rocket = adapter.build_rocket(motor)
        flight = adapter.build_flight(rocket, env)

        history = flight_to_history(flight)
        history["ignition_altitude"] = ignition_altitude

        # Build 14-element final state
        times, states = StateConverter.rocketpy_flight_to_vortex_states(flight)
        final_state = states[:, -1]

        success = history["success"]
        self.history = history

        return success, final_state, history

    def run_simulation_with_variations(self, initial_state: np.ndarray,
                                       ignition_altitude: float = 0.0) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Run a single simulation with randomised parameter variations (for
        Monte Carlo usage).
        """
        varied_config = self._apply_random_variations()
        adapter = VortexConfigAdapter(varied_config)

        env = adapter.build_environment()
        motor = adapter.build_motor()
        rocket = adapter.build_rocket(motor)
        flight = adapter.build_flight(rocket, env)

        history = flight_to_history(flight)
        history["ignition_altitude"] = ignition_altitude

        times, states = StateConverter.rocketpy_flight_to_vortex_states(flight)
        final_state = states[:, -1]

        success = history["success"]
        self.history = history
        return success, final_state, history

    # ------------------------------------------------------------------
    # Variation helpers
    # ------------------------------------------------------------------

    def _apply_random_variations(self) -> Dict[str, Any]:
        """Return a deep copy of self.config with randomised parameters."""
        cfg = copy.deepcopy(self.config)
        rc = cfg["rocket"]
        ec = cfg["environment"]

        # Thrust variation — scale all thrust values
        if self.thrust_variation > 0:
            factor = 1.0 + np.random.uniform(-self.thrust_variation, self.thrust_variation)
            if "thrust_curve" in rc:
                rc["thrust_curve"] = [
                    [pt[0], pt[1] * factor] for pt in rc["thrust_curve"]
                ]

        # Mass variation
        if self.mass_variation > 0:
            dm = np.random.uniform(-self.mass_variation, self.mass_variation)
            rc["dry_mass"] = max(1.0, rc.get("dry_mass", 50.0) * (1.0 + dm))
            rc["propellant_mass"] = max(0.1, rc.get("propellant_mass", 10.0) * (1.0 + dm))

        # Drag variation
        if self.drag_variation > 0:
            dd = np.random.uniform(-self.drag_variation, self.drag_variation)
            ec["drag_coefficient"] = max(0.01, ec.get("drag_coefficient", 0.5) * (1.0 + dd))

        # Air density variation
        if self.density_variation > 0:
            da = np.random.uniform(-self.density_variation, self.density_variation)
            ec["air_density"] = max(0.1, ec.get("air_density", 1.225) * (1.0 + da))

        return cfg

    # ------------------------------------------------------------------
    # Ignition-altitude optimisation (Monte Carlo sweep)
    # ------------------------------------------------------------------

    def calculate_ignition_altitude(self, velocity: float,
                                    altitude: float) -> float:
        """
        Analytical estimate of the optimal ignition altitude for a suicide burn.
        Mirrors SuicideBurnSimulation.calculate_ignition_altitude().
        """
        g = _safe_float(self.environment_config.get("gravity", 9.81))
        dry_mass = self.dry_mass
        prop_mass = self.propellant_mass
        total_mass = dry_mass + prop_mass

        # Average thrust from thrust curve
        curve = self.rocket_config.get("thrust_curve", [[0, 0], [3, 1000], [3.1, 0]])
        thrust_values = [pt[1] for pt in curve]
        avg_thrust = sum(thrust_values) / max(len(thrust_values), 1)
        burn_time = _safe_float(self.rocket_config.get("burn_time", 3.0))

        # Deceleration from thrust
        if total_mass > 0:
            decel = avg_thrust / total_mass - g
        else:
            decel = 0.0

        if decel <= 0:
            return altitude * 0.5  # fallback

        v = abs(velocity)
        # h = v^2 / (2 * decel) + safety buffer
        h_brake = v * v / (2.0 * decel) if decel > 0 else altitude * 0.5
        buffer = _safe_float(self.simulation_config.get("ignition_burnout_buffer", 5.0))
        return h_brake + buffer

    def optimize_ignition_altitude(
        self,
        initial_state: np.ndarray,
        num_monte_carlo: int = 100,
        altitude_search_range: float = 10.0,
        altitude_step: float = 0.1,
        progress_callback: Optional[Callable] = None,
        save_each_trial: bool = False,
        results_folder: Optional[str] = None,
        save_plots_per_trial: bool = False,
    ) -> Tuple[float, Dict[float, float], Optional[Dict[str, Any]]]:
        """
        Sweep ignition altitudes with Monte Carlo trials at each point.

        Mirrors SuicideBurnSimulation.optimize_ignition_altitude() but uses
        RocketPy as the simulation backend.

        NOTE: Because RocketPy runs a full flight from rail launch, the
        *ignition_altitude* parameter is recorded for bookkeeping /
        compatibility with Vortex's cliff-plot pipeline, but the actual
        motor ignition timing is handled by RocketPy internally.  The
        variation across trials comes from randomised physical parameters.

        Returns:
            (optimal_altitude, {altitude: success_rate}, best_history)
        """
        v_for_est = abs(float(initial_state[5])) if len(initial_state) > 5 else 50.0
        h_for_est = float(initial_state[2]) if len(initial_state) > 2 else 1000.0
        estimate = self.calculate_ignition_altitude(v_for_est, h_for_est)

        altitudes = np.arange(
            max(0, estimate - altitude_search_range),
            estimate + altitude_search_range + altitude_step,
            altitude_step,
        )

        success_rates: Dict[float, float] = {}
        best_altitude = estimate
        best_success_rate = 0.0
        best_history: Optional[Dict[str, Any]] = None
        total_runs = len(altitudes) * int(num_monte_carlo)
        runs_completed = 0

        trials_dir = None
        if save_each_trial and results_folder:
            trials_dir = os.path.join(results_folder, "trials")
            os.makedirs(trials_dir, exist_ok=True)

        for altitude in altitudes:
            successes = 0
            last_success_history = None

            for i in range(int(num_monte_carlo)):
                try:
                    success, _final, history = self.run_simulation_with_variations(
                        initial_state.copy(), float(altitude)
                    )
                except Exception as exc:
                    print(f"[RocketPy MC] Trial failed: {exc}")
                    success = False
                    history = None

                if success:
                    successes += 1
                    last_success_history = history

                # Save per-trial CSV
                if save_each_trial and trials_dir and history is not None:
                    alt_str = f"{altitude:.2f}".replace(".", "p")
                    idx_str = f"{i+1:04d}"
                    status = "success" if success else "fail"
                    csv_path = os.path.join(
                        trials_dir,
                        f"trial_alt{alt_str}_idx{idx_str}_{status}.csv",
                    )
                    try:
                        save_trial_csv(history, altitude, csv_path)
                    except Exception:
                        pass

                runs_completed += 1
                if progress_callback:
                    try:
                        progress_callback(runs_completed, total_runs)
                    except Exception:
                        pass
                else:
                    print(f"\r[RocketPy] Optimization: {runs_completed}/{total_runs}", end="", flush=True)

            rate = successes / max(float(num_monte_carlo), 1.0)
            success_rates[float(altitude)] = rate

            if rate > best_success_rate:
                best_success_rate = rate
                best_altitude = float(altitude)
                if last_success_history is not None:
                    best_history = last_success_history

        if progress_callback is None:
            print()  # newline after progress

        return best_altitude, success_rates, best_history

    # ------------------------------------------------------------------
    # Adaptive optimisation
    # ------------------------------------------------------------------

    def optimize_ignition_altitude_adaptive(
        self,
        initial_state: np.ndarray,
        num_monte_carlo: int = 10,
        max_iterations: int = 5,
        samples_per_step: int = 20,
        target_step: float = 0.01,
        altitude_search_range: float = 10.0,
        progress_callback: Optional[Callable] = None,
        save_each_trial: bool = False,
        results_folder: Optional[str] = None,
        save_plots_per_trial: bool = False,
    ) -> Tuple[float, List[Dict], Optional[Dict[str, Any]]]:
        """
        Iterative coarse-to-fine optimisation.

        Returns:
            (optimal_altitude, convergence_history, best_history)
        """
        v_for_est = abs(float(initial_state[5])) if len(initial_state) > 5 else 50.0
        h_for_est = float(initial_state[2]) if len(initial_state) > 2 else 1000.0
        center = self.calculate_ignition_altitude(v_for_est, h_for_est)
        half_range = altitude_search_range

        convergence: List[Dict] = []
        best_altitude = center
        best_history: Optional[Dict[str, Any]] = None
        total_estimated = max_iterations * samples_per_step * num_monte_carlo
        global_run = 0

        trials_dir = None
        if save_each_trial and results_folder:
            trials_dir = os.path.join(results_folder, "trials")
            os.makedirs(trials_dir, exist_ok=True)

        for iteration in range(max_iterations):
            step = 2.0 * half_range / max(samples_per_step - 1, 1)
            if step < target_step and iteration > 0:
                break

            altitudes = np.linspace(center - half_range, center + half_range, samples_per_step)
            altitudes = altitudes[altitudes >= 0]

            local_rates: Dict[float, float] = {}
            for alt in altitudes:
                successes = 0
                last_hist = None
                for j in range(int(num_monte_carlo)):
                    try:
                        ok, _, hist = self.run_simulation_with_variations(
                            initial_state.copy(), float(alt)
                        )
                    except Exception:
                        ok = False
                        hist = None

                    if ok:
                        successes += 1
                        last_hist = hist

                    if save_each_trial and trials_dir and hist is not None:
                        alt_str = f"{alt:.2f}".replace(".", "p")
                        idx_str = f"{global_run+1:04d}"
                        status = "success" if ok else "fail"
                        csv_path = os.path.join(trials_dir, f"trial_alt{alt_str}_idx{idx_str}_{status}.csv")
                        try:
                            save_trial_csv(hist, float(alt), csv_path)
                        except Exception:
                            pass

                    global_run += 1
                    if progress_callback:
                        try:
                            progress_callback(global_run, total_estimated)
                        except Exception:
                            pass

                rate = successes / max(float(num_monte_carlo), 1.0)
                local_rates[float(alt)] = rate

                if rate >= local_rates.get(best_altitude, 0.0):
                    best_altitude = float(alt)
                    if last_hist is not None:
                        best_history = last_hist

            convergence.append({
                "iteration": iteration,
                "center": center,
                "half_range": half_range,
                "step": step,
                "rates": dict(local_rates),
                "best_altitude": best_altitude,
            })

            # Re-center on the best altitude found
            center = best_altitude
            half_range *= 0.5

        return best_altitude, convergence, best_history

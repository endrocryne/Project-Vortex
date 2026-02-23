"""
RocketPy Flight -> Vortex CSV Data Converter
==============================================

Converts RocketPy Flight objects into the Vortex CSV format used throughout
the system:

    Time, X, Y, Z, VX, VY, VZ, QW, QX, QY, QZ, Mass

Single-run CSVs go to results/<single_run_TIMESTAMP>/single_run.csv
Optimization CSVs go to results/<optimization_TIMESTAMP>/optimization.csv
Trial CSVs follow the pattern: trial_alt{alt}_idx{idx}_{status}.csv
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Use a relative import when loaded as part of the extension package
try:
    from .adapters import StateConverter, HAS_ROCKETPY
except ImportError:
    from core.adapters import StateConverter, HAS_ROCKETPY


# ---------------------------------------------------------------------------
# CSV column spec (must match simulation.py output exactly)
# ---------------------------------------------------------------------------

TRAJECTORY_COLUMNS = [
    "Time", "X", "Y", "Z",
    "VX", "VY", "VZ",
    "QW", "QX", "QY", "QZ",
    "Mass",
]

OPTIMIZATION_COLUMNS = ["Ignition Altitude (m)", "Success Rate"]

TRIAL_COLUMNS = TRAJECTORY_COLUMNS + ["AltitudeTest"]

GRID_SEARCH_EXTRA = ["success_rate", "mean_landing_velocity", "std_landing_velocity"]


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def flight_to_trajectory_rows(flight: Any) -> Tuple[List[str], List[List[float]]]:
    """
    Convert a RocketPy Flight object to rows matching Vortex single-run CSV.

    Returns:
        (header, rows) where each row is a list of floats
    """
    times, states = StateConverter.rocketpy_flight_to_vortex_states(flight)
    rows = []
    for i in range(len(times)):
        rows.append([
            float(times[i]),
            float(states[0, i]),   # X
            float(states[1, i]),   # Y
            float(states[2, i]),   # Z
            float(states[3, i]),   # VX
            float(states[4, i]),   # VY
            float(states[5, i]),   # VZ
            float(states[6, i]),   # QW
            float(states[7, i]),   # QX
            float(states[8, i]),   # QY
            float(states[9, i]),   # QZ
            float(states[13, i]),  # Mass
        ])
    return TRAJECTORY_COLUMNS, rows


def states_to_trajectory_rows(times: np.ndarray,
                               states: np.ndarray) -> Tuple[List[str], List[List[float]]]:
    """
    Convert pre-extracted Vortex state arrays (14, N) to CSV rows.
    """
    rows = []
    for i in range(len(times)):
        rows.append([
            float(times[i]),
            float(states[0, i]),
            float(states[1, i]),
            float(states[2, i]),
            float(states[3, i]),
            float(states[4, i]),
            float(states[5, i]),
            float(states[6, i]),
            float(states[7, i]),
            float(states[8, i]),
            float(states[9, i]),
            float(states[13, i]),
        ])
    return TRAJECTORY_COLUMNS, rows


def flight_to_dataframe(flight: Any) -> "pd.DataFrame":
    """Convert a RocketPy Flight to a pandas DataFrame with Vortex columns."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for DataFrame conversion")
    header, rows = flight_to_trajectory_rows(flight)
    return pd.DataFrame(rows, columns=header)


def states_to_dataframe(times: np.ndarray, states: np.ndarray) -> "pd.DataFrame":
    """Convert pre-extracted state arrays to a pandas DataFrame."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for DataFrame conversion")
    header, rows = states_to_trajectory_rows(times, states)
    return pd.DataFrame(rows, columns=header)


# ---------------------------------------------------------------------------
# History dict (matches simulation.py history format)
# ---------------------------------------------------------------------------

def flight_to_history(flight: Any) -> Dict[str, Any]:
    """
    Convert a RocketPy Flight to Vortex's internal history dict format — the
    same dict structure returned by SuicideBurnSimulation.run_simulation().
    """
    times, states = StateConverter.rocketpy_flight_to_vortex_states(flight)

    # Landing analysis
    final_z = float(states[2, -1])
    final_speed = float(np.linalg.norm(states[3:6, -1]))
    altitude_ok = abs(final_z) < 1.0
    velocity_ok = abs(states[5, -1]) < 2.0
    total_velocity_ok = final_speed < 3.0
    success = altitude_ok and velocity_ok and total_velocity_ok

    # Apogee
    apogee_idx = int(np.argmax(states[2, :]))
    apogee_alt = float(states[2, apogee_idx])
    apogee_time = float(times[apogee_idx])

    history = {
        "t": times,
        "x": states[0, :],
        "y": states[1, :],
        "z": states[2, :],
        "vx": states[3, :],
        "vy": states[4, :],
        "vz": states[5, :],
        "qw": states[6, :],
        "qx": states[7, :],
        "qy": states[8, :],
        "qz": states[9, :],
        "omega_x": states[10, :],
        "omega_y": states[11, :],
        "omega_z": states[12, :],
        "mass": states[13, :],
        "success": success,
        "final_altitude": final_z,
        "final_velocity": final_speed,
        "apogee_altitude": apogee_alt,
        "apogee_time": apogee_time,
        "ignition_altitude": 0.0,  # RocketPy handles ignition internally
        "backend": "rocketpy",
    }
    return history


# ---------------------------------------------------------------------------
# File I/O — matches Vortex results/ directory patterns
# ---------------------------------------------------------------------------

def make_results_folder(mode: str = "single_run",
                        base_dir: str = "results") -> Tuple[str, str]:
    """
    Create a timestamped results folder.

    Args:
        mode: 'single_run' or 'optimization' or 'grid_search'
        base_dir: Parent results directory

    Returns:
        (folder_path, timestamp_string)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, f"rocketpy_{mode}_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder, ts


def save_trajectory_csv(flight_or_history: Any, filepath: str) -> str:
    """
    Save trajectory data to a Vortex-compatible CSV.

    Args:
        flight_or_history: Either a RocketPy Flight object or a Vortex history dict
        filepath: Output CSV path

    Returns:
        Absolute path to the saved file
    """
    if isinstance(flight_or_history, dict):
        # Already a history dict
        h = flight_or_history
        header = TRAJECTORY_COLUMNS
        n = len(h["t"])
        rows = []
        for i in range(n):
            rows.append([
                float(h["t"][i]),
                float(h["x"][i]),
                float(h["y"][i]),
                float(h["z"][i]),
                float(h["vx"][i]),
                float(h["vy"][i]),
                float(h["vz"][i]),
                float(h["qw"][i]),
                float(h["qx"][i]),
                float(h["qy"][i]),
                float(h["qz"][i]),
                float(h["mass"][i]),
            ])
    else:
        # RocketPy Flight object
        header, rows = flight_to_trajectory_rows(flight_or_history)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return os.path.abspath(filepath)


def save_optimization_csv(success_rates: Dict[float, float], filepath: str) -> str:
    """
    Save ignition-altitude sweep results in Vortex optimization CSV format.

    Args:
        success_rates: {altitude: success_rate} mapping
        filepath: Output CSV path
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OPTIMIZATION_COLUMNS)
        for alt in sorted(success_rates.keys()):
            writer.writerow([alt, success_rates[alt]])
    return os.path.abspath(filepath)


def save_trial_csv(history: Dict[str, Any], altitude_test: float,
                   filepath: str) -> str:
    """Save a single Monte Carlo trial CSV (with AltitudeTest column)."""
    h = history
    n = len(h["t"])
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRIAL_COLUMNS)
        for i in range(n):
            writer.writerow([
                float(h["t"][i]),
                float(h["x"][i]),
                float(h["y"][i]),
                float(h["z"][i]),
                float(h["vx"][i]),
                float(h["vy"][i]),
                float(h["vz"][i]),
                float(h["qw"][i]),
                float(h["qx"][i]),
                float(h["qy"][i]),
                float(h["qz"][i]),
                float(h["mass"][i]),
                altitude_test,
            ])
    return os.path.abspath(filepath)


def save_grid_search_csv(results: List[Dict[str, Any]], param_names: List[str],
                         filepath: str) -> str:
    """
    Save grid-search results.

    Each entry in *results* is a dict with the swept parameter values plus
    'success_rate', 'mean_landing_velocity', 'std_landing_velocity'.
    """
    header = list(param_names) + GRID_SEARCH_EXTRA
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in results:
            row = [entry.get(p, 0.0) for p in param_names]
            row.append(entry.get("success_rate", 0.0))
            row.append(entry.get("mean_landing_velocity", 0.0))
            row.append(entry.get("std_landing_velocity", 0.0))
            writer.writerow(row)
    return os.path.abspath(filepath)


def save_config(config: Dict[str, Any], filepath: str) -> str:
    """Persist the configuration dict as JSON."""
    import json
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Plot generation (Agg backend, PNG output)
# ---------------------------------------------------------------------------

def generate_trajectory_plots(history: Dict[str, Any],
                              output_dir: str) -> List[str]:
    """
    Generate standard Vortex trajectory PNG plots from a history dict.

    Plots:
        - trajectory_2d.png: Altitude vs Time, Velocity vs Time
        - trajectory_3d.png: 3D trajectory path

    Returns:
        List of generated file paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    files: List[str] = []
    t = history["t"]
    z = history["z"]
    vz = history["vz"]
    x = history["x"]
    y = history["y"]
    speed = np.sqrt(np.array(history["vx"]) ** 2 +
                    np.array(history["vy"]) ** 2 +
                    np.array(history["vz"]) ** 2)

    # --- 2D trajectory ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, z, "b-", linewidth=1.2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 0].set_title("Altitude vs Time")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, speed, "r-", linewidth=1.2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Speed vs Time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, history["mass"], "g-", linewidth=1.2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Mass (kg)")
    axes[1, 0].set_title("Mass vs Time")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, y, "m-", linewidth=1.2)
    axes[1, 1].set_xlabel("X Position (m)")
    axes[1, 1].set_ylabel("Y Position (m)")
    axes[1, 1].set_title("Ground Track")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect("equal")

    fig.suptitle("RocketPy Flight — Trajectory Profile", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path_2d = os.path.join(output_dir, "trajectory_2d.png")
    fig.savefig(path_2d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(path_2d)

    # --- 3D trajectory ---
    fig3d = plt.figure(figsize=(10, 8))
    ax3 = fig3d.add_subplot(111, projection="3d")
    ax3.plot(x, y, z, "b-", linewidth=1.2)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z / Altitude (m)")
    ax3.set_title("RocketPy Flight — 3D Trajectory")
    path_3d = os.path.join(output_dir, "trajectory_3d.png")
    fig3d.savefig(path_3d, dpi=150, bbox_inches="tight")
    plt.close(fig3d)
    files.append(path_3d)

    return files

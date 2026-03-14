"""Generate PNG validation plots and summary metrics for synthetic demo runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "demo_runs"
PLOTS_DIR = RUNS_DIR / "validation_plots"
SUMMARY_PATH = PLOTS_DIR / "validation_summary.json"


def read_rows(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def tilt_angle_deg(row: Dict[str, float]) -> float:
    qx = row["QX"]
    qy = row["QY"]
    cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
    cos_tilt = max(-1.0, min(1.0, cos_tilt))
    return math.degrees(math.acos(cos_tilt))


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    altitudes = [row["Z"] for row in rows]
    vz_values = [row["VZ"] for row in rows]
    tilts = [tilt_angle_deg(row) for row in rows]
    freefall_vz = [abs(row["VZ"]) for row in rows if row["Phase"] == 2]
    ascent_tilts = [tilt for row, tilt in zip(rows, tilts) if row["Phase"] == 0]
    freefall_tilts = [tilt for row, tilt in zip(rows, tilts) if row["Phase"] in (1, 2)]
    descent_tilts = [tilt for row, tilt in zip(rows, tilts) if row["Phase"] == 3]
    burn_faults = [row for row in rows if row["Phase"] in (3, 4) and (row["FaultType"] != 0 or row["FaultMag"] != 0)]
    ignition_row = next(row for row in rows if row["Phase"] == 3)
    landing_row = rows[-1]
    return {
        "apogee": max(altitudes),
        "peak_abs_vz": max(abs(value) for value in vz_values),
        "terminal_speed_before_ignition": max(freefall_vz) if freefall_vz else 0.0,
        "max_ascent_tilt": max(ascent_tilts) if ascent_tilts else 0.0,
        "max_freefall_tilt": max(freefall_tilts) if freefall_tilts else 0.0,
        "max_descent_tilt": max(descent_tilts) if descent_tilts else 0.0,
        "ignition_altitude": ignition_row["Z"],
        "ignition_vz": ignition_row["VZ"],
        "landing_speed": abs(landing_row["VZ"]),
        "landing_x": landing_row["X"],
        "landing_y": landing_row["Y"],
        "burn_fault_rows": len(burn_faults),
    }


def plot_run(run_id: str, rows: List[Dict[str, float]], metrics: Dict[str, float]) -> None:
    times = [row["Time"] for row in rows]
    x_values = [row["X"] for row in rows]
    y_values = [row["Y"] for row in rows]
    z_values = [row["Z"] for row in rows]
    vz_values = [row["VZ"] for row in rows]
    tilts = [tilt_angle_deg(row) for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(x_values, z_values, color="#006699", linewidth=2.0)
    axes[0, 0].set_title("XZ Trajectory")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(y_values, z_values, color="#009966", linewidth=2.0)
    axes[0, 1].set_title("YZ Trajectory")
    axes[0, 1].set_xlabel("Y (m)")
    axes[0, 1].set_ylabel("Altitude (m)")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(times, z_values, label="Altitude", color="#222222", linewidth=2.0)
    axes[1, 0].plot(times, vz_values, label="Vertical Velocity", color="#cc5500", linewidth=2.0)
    axes[1, 0].set_title("Altitude and Vertical Velocity")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(x_values, y_values, color="#663399", linewidth=2.0, label="Ground Track")
    axes[1, 1].plot([0.0], [0.0], marker="o", color="#000000", label="Pad")
    axes[1, 1].plot([x_values[-1]], [y_values[-1]], marker="x", color="#cc0000", label="Landing")
    axes[1, 1].set_title("XY Ground Track")
    axes[1, 1].set_xlabel("X (m)")
    axes[1, 1].set_ylabel("Y (m)")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle(
        f"{run_id} | apogee={metrics['apogee']:.1f} m | ignition={metrics['ignition_altitude']:.1f} m | landing_v={metrics['landing_speed']:.2f} m/s",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / f"{run_id}_trajectory_velocity.png", dpi=180)
    plt.close(figure)

    tilt_figure, tilt_axis = plt.subplots(figsize=(10, 4.5))
    tilt_axis.plot(times, tilts, color="#bb2200", linewidth=2.0)
    tilt_axis.axhline(15.0, color="#666666", linestyle="--", linewidth=1.0)
    tilt_axis.axhline(30.0, color="#000000", linestyle=":", linewidth=1.0)
    tilt_axis.set_title(f"{run_id} Tilt Angle")
    tilt_axis.set_xlabel("Time (s)")
    tilt_axis.set_ylabel("Tilt (deg)")
    tilt_axis.grid(True, alpha=0.25)
    tilt_figure.tight_layout()
    tilt_figure.savefig(PLOTS_DIR / f"{run_id}_tilt.png", dpi=180)
    plt.close(tilt_figure)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUNS_DIR / "demo_manifest.json").read_text())
    summary = {}

    for run in manifest["runs"]:
        run_id = run["id"]
        rows = read_rows(RUNS_DIR / run_id / "trajectory.csv")
        metrics = summarize(rows)
        summary[run_id] = metrics
        plot_run(run_id, rows, metrics)
        print(
            f"[{run_id}] ignition={metrics['ignition_altitude']:.2f} m terminal={metrics['terminal_speed_before_ignition']:.2f} m/s "
            f"landing={metrics['landing_speed']:.2f} m/s tilt(ascent/freefall/descent)=({metrics['max_ascent_tilt']:.1f}/{metrics['max_freefall_tilt']:.1f}/{metrics['max_descent_tilt']:.1f})"
        )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote validation plots to {PLOTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
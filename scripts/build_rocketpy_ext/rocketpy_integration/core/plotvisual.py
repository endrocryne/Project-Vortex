"""
PlotVisual Graph Hooks for RocketPy Integration
==================================================

Registers custom graphs in PlotVisual for RocketPy-specific data and
for side-by-side comparison of Vortex-native vs RocketPy simulations.

These graphs read data from VortexDataStore using the keys:
    'rocketpy_trajectory'    — single-run RocketPy trajectory DataFrame
    'rocketpy_single_run'    — alias used by data_store loader
    'rocketpy_grid_search'   — grid search results DataFrame
    'trajectory'             — native Vortex trajectory (for comparison)
    'optimization'           — native Vortex optimization (for comparison)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

try:
    from plugins.base import GraphDefinition
except ImportError:
    # Minimal stub so the module can be imported stand-alone
    class GraphDefinition:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Graph definitions
# ---------------------------------------------------------------------------

ROCKETPY_GRAPHS: List[GraphDefinition] = [
    GraphDefinition(
        key="rocketpy_trajectory_profile",
        label="RocketPy Trajectory Profile",
        icon="🚀",
        category="RocketPy",
        description="4-panel trajectory breakdown from a RocketPy flight: altitude, speed, mass, ground track.",
        required_data_types=["rocketpy_trajectory"],
    ),
    GraphDefinition(
        key="rocketpy_stability_analysis",
        label="RocketPy Stability Analysis",
        icon="📐",
        category="RocketPy",
        description="Quaternion / angular velocity timeline showing flight stability.",
        required_data_types=["rocketpy_trajectory"],
    ),
    GraphDefinition(
        key="rocketpy_grid_search_heatmap",
        label="Grid Search Heatmap",
        icon="🗺️",
        category="RocketPy",
        description="2D heatmap of success rate across two swept parameters.",
        required_data_types=["rocketpy_grid_search"],
    ),
    GraphDefinition(
        key="comparison_vortex_vs_rocketpy",
        label="Vortex vs RocketPy Comparison",
        icon="⚖️",
        category="RocketPy",
        description="Overlay native Vortex and RocketPy trajectories for the same configuration.",
        required_data_types=["trajectory", "rocketpy_trajectory"],
    ),
    GraphDefinition(
        key="rocketpy_landing_dispersion",
        label="RocketPy Landing Dispersion",
        icon="🎯",
        category="RocketPy",
        description="Scatter plot of landing positions from Monte Carlo trials.",
        required_data_types=["monte_carlo_trials"],
    ),
    GraphDefinition(
        key="rocketpy_velocity_profile",
        label="RocketPy Velocity Components",
        icon="📈",
        category="RocketPy",
        description="Individual velocity components (VX, VY, VZ) over time.",
        required_data_types=["rocketpy_trajectory"],
    ),
]


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------

def render_rocketpy_graph(graph_key: str, data_store: Any, fig: Any,
                          **kwargs) -> bool:
    """
    Render one of the registered RocketPy graphs onto *fig*.

    Args:
        graph_key: One of the keys defined in ROCKETPY_GRAPHS
        data_store: VortexDataStore instance
        fig: matplotlib Figure

    Returns:
        True if rendering succeeded, False otherwise
    """
    dispatch = {
        "rocketpy_trajectory_profile": _render_trajectory_profile,
        "rocketpy_stability_analysis": _render_stability_analysis,
        "rocketpy_grid_search_heatmap": _render_grid_search_heatmap,
        "comparison_vortex_vs_rocketpy": _render_comparison,
        "rocketpy_landing_dispersion": _render_landing_dispersion,
        "rocketpy_velocity_profile": _render_velocity_profile,
    }
    fn = dispatch.get(graph_key)
    if fn is None:
        return False
    try:
        fn(data_store, fig, **kwargs)
        return True
    except Exception as exc:
        # Show error on the figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Render error:\n{exc}",
                ha="center", va="center", fontsize=12, color="red",
                transform=ax.transAxes)
        return False


# ---------------------------------------------------------------------------
# Individual renderers
# ---------------------------------------------------------------------------

def _get_rocketpy_df(data_store: Any):
    """Try multiple keys to find a RocketPy trajectory DataFrame."""
    for key in ("rocketpy_trajectory", "rocketpy_single_run"):
        df = data_store.get(key)
        if df is not None and len(df) > 0:
            return df
    return None


def _render_trajectory_profile(data_store: Any, fig: Any, **kw):
    df = _get_rocketpy_df(data_store)
    if df is None:
        raise ValueError("No RocketPy trajectory data loaded.")

    axes = fig.subplots(2, 2)

    t = df["Time"].values
    z = df["Z"].values
    speed = np.sqrt(df["VX"].values ** 2 + df["VY"].values ** 2 + df["VZ"].values ** 2)

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

    axes[1, 0].plot(t, df["Mass"].values, "g-", linewidth=1.2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Mass (kg)")
    axes[1, 0].set_title("Mass vs Time")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["X"].values, df["Y"].values, "m-", linewidth=1.2)
    axes[1, 1].set_xlabel("X Position (m)")
    axes[1, 1].set_ylabel("Y Position (m)")
    axes[1, 1].set_title("Ground Track")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect("equal")

    fig.suptitle("RocketPy Flight — Trajectory Profile", fontsize=14, fontweight="bold")
    fig.tight_layout()


def _render_stability_analysis(data_store: Any, fig: Any, **kw):
    df = _get_rocketpy_df(data_store)
    if df is None:
        raise ValueError("No RocketPy trajectory data loaded.")

    axes = fig.subplots(2, 2)
    t = df["Time"].values

    # Quaternion components
    for i, (col, label, color) in enumerate([
        ("QW", "qw (scalar)", "#1f77b4"),
        ("QX", "qx", "#ff7f0e"),
        ("QY", "qy", "#2ca02c"),
        ("QZ", "qz", "#d62728"),
    ]):
        ax = axes[i // 2, i % 2]
        if col in df.columns:
            ax.plot(t, df[col].values, color=color, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(f"Quaternion — {label}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RocketPy — Orientation Stability", fontsize=14, fontweight="bold")
    fig.tight_layout()


def _render_grid_search_heatmap(data_store: Any, fig: Any, **kw):
    df = data_store.get("rocketpy_grid_search")
    if df is None or len(df) == 0:
        raise ValueError("No grid search data loaded.")

    # Identify the swept parameters (all columns except the standard outputs)
    output_cols = {"success_rate", "mean_landing_velocity", "std_landing_velocity"}
    param_cols = [c for c in df.columns if c not in output_cols]

    if len(param_cols) >= 2:
        p1, p2 = param_cols[0], param_cols[1]
        v1 = sorted(df[p1].unique())
        v2 = sorted(df[p2].unique())

        grid = np.full((len(v2), len(v1)), np.nan)
        for _, row in df.iterrows():
            i = v2.index(row[p2])
            j = v1.index(row[p1])
            grid[i, j] = row["success_rate"]

        ax = fig.add_subplot(111)
        im = ax.imshow(grid, origin="lower", aspect="auto",
                        extent=[min(v1), max(v1), min(v2), max(v2)],
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xlabel(p1, fontsize=12)
        ax.set_ylabel(p2, fontsize=12)
        ax.set_title("Grid Search — Success Rate Heatmap", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Success Rate")

    elif len(param_cols) == 1:
        p = param_cols[0]
        ax = fig.add_subplot(111)
        ax.plot(df[p].values, df["success_rate"].values, "b-o", linewidth=1.5, markersize=4)
        ax.set_xlabel(p, fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Grid Search — {p} Sweep", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
    else:
        raise ValueError("Grid search data has no identifiable parameter columns.")

    fig.tight_layout()


def _render_comparison(data_store: Any, fig: Any, **kw):
    """Overlay native Vortex and RocketPy trajectories."""
    vortex_df = data_store.get("trajectory")
    rp_df = _get_rocketpy_df(data_store)

    if vortex_df is None and rp_df is None:
        raise ValueError("Need at least one trajectory to display.")

    axes = fig.subplots(1, 3, figsize=(15, 5))

    # Altitude
    ax = axes[0]
    if vortex_df is not None and "Z" in vortex_df.columns:
        ax.plot(vortex_df["Time"], vortex_df["Z"], "b-", linewidth=1.2,
                label="Vortex Native", alpha=0.8)
    if rp_df is not None and "Z" in rp_df.columns:
        ax.plot(rp_df["Time"], rp_df["Z"], "r--", linewidth=1.2,
                label="RocketPy", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speed
    ax = axes[1]
    if vortex_df is not None and "VZ" in vortex_df.columns:
        speed_v = np.sqrt(vortex_df["VX"] ** 2 + vortex_df["VY"] ** 2 + vortex_df["VZ"] ** 2)
        ax.plot(vortex_df["Time"], speed_v, "b-", linewidth=1.2,
                label="Vortex Native", alpha=0.8)
    if rp_df is not None and "VZ" in rp_df.columns:
        speed_r = np.sqrt(rp_df["VX"] ** 2 + rp_df["VY"] ** 2 + rp_df["VZ"] ** 2)
        ax.plot(rp_df["Time"], speed_r, "r--", linewidth=1.2,
                label="RocketPy", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ground track
    ax = axes[2]
    if vortex_df is not None and "X" in vortex_df.columns:
        ax.plot(vortex_df["X"], vortex_df["Y"], "b-", linewidth=1.2,
                label="Vortex Native", alpha=0.8)
    if rp_df is not None and "X" in rp_df.columns:
        ax.plot(rp_df["X"], rp_df["Y"], "r--", linewidth=1.2,
                label="RocketPy", alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ground Track Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle("Vortex Native vs RocketPy — Trajectory Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()


def _render_landing_dispersion(data_store: Any, fig: Any, **kw):
    """Scatter plot of final (X, Y) positions from Monte Carlo trials."""
    df = data_store.get("monte_carlo_trials")
    if df is None or len(df) == 0:
        raise ValueError("No Monte Carlo trial data loaded.")

    ax = fig.add_subplot(111)

    # Get final row of each trial
    if "_trial_file" in df.columns:
        groups = df.groupby("_trial_file")
        xs, ys, colors = [], [], []
        for name, grp in groups:
            last = grp.iloc[-1]
            xs.append(last["X"])
            ys.append(last["Y"])
            colors.append("green" if last.get("_trial_success", True) else "red")
        ax.scatter(xs, ys, c=colors, s=15, alpha=0.6, edgecolors="none")
    else:
        # Fall back: just plot all X, Y
        ax.scatter(df["X"], df["Y"], s=5, alpha=0.3, edgecolors="none")

    ax.set_xlabel("X — Downrange (m)")
    ax.set_ylabel("Y — Crossrange (m)")
    ax.set_title("RocketPy Monte Carlo — Landing Dispersion")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def _render_velocity_profile(data_store: Any, fig: Any, **kw):
    df = _get_rocketpy_df(data_store)
    if df is None:
        raise ValueError("No RocketPy trajectory data loaded.")

    axes = fig.subplots(3, 1, sharex=True)
    t = df["Time"].values

    for ax, col, label, color in [
        (axes[0], "VX", "VX (m/s)", "#1f77b4"),
        (axes[1], "VY", "VY (m/s)", "#ff7f0e"),
        (axes[2], "VZ", "VZ (m/s)", "#2ca02c"),
    ]:
        ax.plot(t, df[col].values, color=color, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    fig.suptitle("RocketPy — Velocity Components", fontsize=14, fontweight="bold")
    fig.tight_layout()

"""
HexaVisual 3D Overlay Hooks for RocketPy Integration
======================================================

Provides overlay definitions and rendering hooks for presenting
RocketPy-specific flight data in HexaVisual's 3D PyVista viewer.

Overlays:
    - Apogee marker (sphere at maximum altitude)
    - Parachute deployment marker(s)
    - Rail departure point marker
    - CP / CG stability visualisation bar
    - RocketPy trajectory overlay (red dashed) for comparison
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from extensions.hooks.hexavisual import (
        HexaVisualExtension,
        OverlayDefinition,
        CameraModeDefinition,
        HUDWidgetDefinition,
    )
    HAS_HEXAVISUAL_HOOKS = True
except ImportError:
    HAS_HEXAVISUAL_HOOKS = False

    # Stubs so the module can be imported stand-alone
    class OverlayDefinition:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class CameraModeDefinition:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class HUDWidgetDefinition:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Overlay definitions
# ---------------------------------------------------------------------------

ROCKETPY_OVERLAYS: List[OverlayDefinition] = [
    OverlayDefinition(
        key="rocketpy_apogee_marker",
        label="Apogee Marker",
        description="Sphere at the maximum altitude reached during RocketPy flight.",
        icon="🔴",
        default_visible=True,
    ),
    OverlayDefinition(
        key="rocketpy_trajectory_comparison",
        label="RocketPy Trajectory Overlay",
        description="Red dashed line showing the RocketPy trajectory alongside the native Vortex trajectory.",
        icon="🔻",
        default_visible=True,
    ),
    OverlayDefinition(
        key="rocketpy_rail_departure",
        label="Rail Departure Point",
        description="Mark where the rocket leaves the launch rail.",
        icon="📍",
        default_visible=False,
    ),
    OverlayDefinition(
        key="rocketpy_landing_zone",
        label="Landing Zone Circle",
        description="Circle on the ground showing Monte Carlo landing dispersion.",
        icon="🎯",
        default_visible=False,
    ),
]

ROCKETPY_HUD_WIDGETS: List[HUDWidgetDefinition] = [
    HUDWidgetDefinition(
        key="rocketpy_flight_stats",
        label="RocketPy Stats",
        description="Shows key flight metrics: apogee, max speed, landing velocity.",
        position="top-right",
        width=220,
        height=120,
    ),
]


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def render_overlay(overlay_key: str, plotter: Any,
                   trajectory_data: Any, time_index: int,
                   params: Dict[str, Any]) -> None:
    """
    Render a RocketPy overlay into a PyVista plotter.

    Args:
        overlay_key: Key from ROCKETPY_OVERLAYS
        plotter: PyVista QtInteractor
        trajectory_data: pandas DataFrame with Vortex trajectory columns
        time_index: Current playback time index
        params: Additional parameters from the extension settings
    """
    dispatch = {
        "rocketpy_apogee_marker": _render_apogee_marker,
        "rocketpy_trajectory_comparison": _render_trajectory_comparison,
        "rocketpy_rail_departure": _render_rail_departure,
        "rocketpy_landing_zone": _render_landing_zone,
    }
    fn = dispatch.get(overlay_key)
    if fn:
        fn(plotter, trajectory_data, time_index, params)


def remove_overlay(overlay_key: str, plotter: Any) -> None:
    """Remove a previously rendered overlay from the plotter."""
    actor_name = f"rocketpy_{overlay_key}"
    try:
        plotter.remove_actor(actor_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Individual overlay renderers
# ---------------------------------------------------------------------------

def _render_apogee_marker(plotter: Any, df: Any, time_index: int,
                          params: Dict[str, Any]) -> None:
    """Render a red sphere at the apogee point."""
    try:
        import pyvista as pv
    except ImportError:
        return

    if df is None or len(df) == 0:
        return

    apogee_idx = int(df["Z"].values.argmax())
    x = float(df["X"].iloc[apogee_idx])
    y = float(df["Y"].iloc[apogee_idx])
    z = float(df["Z"].iloc[apogee_idx])

    sphere = pv.Sphere(radius=max(z * 0.01, 2.0), center=(x, y, z))
    try:
        plotter.add_mesh(sphere, color="red", opacity=0.7,
                         name="rocketpy_rocketpy_apogee_marker")
    except Exception:
        pass


def _render_trajectory_comparison(plotter: Any, df: Any, time_index: int,
                                   params: Dict[str, Any]) -> None:
    """Render the RocketPy trajectory as a red line."""
    try:
        import pyvista as pv
    except ImportError:
        return

    # The comparison trajectory should come from params or a secondary dataset
    rp_df = params.get("rocketpy_trajectory_df")
    if rp_df is None or len(rp_df) == 0:
        return

    points = np.column_stack([
        rp_df["X"].values,
        rp_df["Y"].values,
        rp_df["Z"].values,
    ])

    if len(points) < 2:
        return

    line = pv.Spline(points, n_points=min(len(points), 500))
    try:
        plotter.add_mesh(line, color="red", line_width=2, opacity=0.6,
                         name="rocketpy_rocketpy_trajectory_comparison")
    except Exception:
        pass


def _render_rail_departure(plotter: Any, df: Any, time_index: int,
                            params: Dict[str, Any]) -> None:
    """Small cone at the rail departure position (early in flight)."""
    try:
        import pyvista as pv
    except ImportError:
        return

    if df is None or len(df) < 10:
        return

    # Rail departure ~ first few data points
    idx = min(5, len(df) - 1)
    x = float(df["X"].iloc[idx])
    y = float(df["Y"].iloc[idx])
    z = float(df["Z"].iloc[idx])

    cone = pv.Cone(center=(x, y, z), direction=(0, 0, 1),
                   height=3.0, radius=1.0)
    try:
        plotter.add_mesh(cone, color="yellow", opacity=0.5,
                         name="rocketpy_rocketpy_rail_departure")
    except Exception:
        pass


def _render_landing_zone(plotter: Any, df: Any, time_index: int,
                          params: Dict[str, Any]) -> None:
    """Ground-level circle showing landing dispersion radius."""
    try:
        import pyvista as pv
    except ImportError:
        return

    radius = float(params.get("landing_dispersion_radius", 50.0))
    center_x = float(params.get("landing_center_x", 0.0))
    center_y = float(params.get("landing_center_y", 0.0))

    disc = pv.Disc(center=(center_x, center_y, 0.1),
                   normal=(0, 0, 1), inner=radius * 0.95, outer=radius)
    try:
        plotter.add_mesh(disc, color="orange", opacity=0.3,
                         name="rocketpy_rocketpy_landing_zone")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HUD widget rendering
# ---------------------------------------------------------------------------

def render_hud_widget(widget_key: str, painter: Any, rect: Any,
                      state: Dict[str, Any]) -> None:
    """
    Render a HUD widget via QPainter.

    Args:
        widget_key: Key from ROCKETPY_HUD_WIDGETS
        painter: QPainter instance
        rect: QRect defining the widget area
        state: Current flight state dict
    """
    if widget_key == "rocketpy_flight_stats":
        _render_flight_stats_hud(painter, rect, state)


def _render_flight_stats_hud(painter: Any, rect: Any,
                              state: Dict[str, Any]) -> None:
    """Draw flight statistics text overlay."""
    try:
        from PyQt5.QtCore import Qt, QRect as QR
        from PyQt5.QtGui import QFont, QColor, QPen
    except ImportError:
        return

    apogee = state.get("apogee_altitude", 0.0)
    max_speed = state.get("max_speed", 0.0)
    landing_v = state.get("final_velocity", 0.0)
    backend = state.get("backend", "rocketpy")

    painter.save()
    painter.setPen(QPen(QColor(255, 255, 255)))
    painter.setFont(QFont("Consolas", 9))

    lines = [
        f"Backend: {backend}",
        f"Apogee: {apogee:.1f} m",
        f"Max Speed: {max_speed:.1f} m/s",
        f"Landing V: {landing_v:.2f} m/s",
    ]

    y = rect.y() + 5
    for line in lines:
        painter.drawText(rect.x() + 5, y + 12, line)
        y += 16

    painter.restore()

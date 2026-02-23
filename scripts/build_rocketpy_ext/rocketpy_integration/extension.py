"""
RocketPy Integration — Main Extension Class
=============================================

This is the entry point loaded by the Vortex extension system.
It implements the VortexExtension ABC and provides hooks for
HexaKinetic (simulation backend), PlotVisual (graphs), and
HexaVisual (3D overlays).

Because the extension_type is 'universal', it will be activated
by every Vortex app that creates an ExtensionRegistry.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Vortex extension framework imports ---
try:
    from extensions.base import VortexExtension, ExtensionType
except ImportError:
    # Fallback stubs for stand-alone testing
    class ExtensionType:
        UNIVERSAL = "universal"

    class VortexExtension:
        def activate(self, app_context=None):
            pass
        def deactivate(self):
            pass
        def get_settings_schema(self):
            return []

# --- HexaKinetic hook definitions ---
try:
    from extensions.hooks.hexakinetic import (
        FaultModelDefinition,
        MotorProfileDefinition,
        ControllerDefinition,
    )
    HAS_HEXAKINETIC_HOOKS = True
except ImportError:
    HAS_HEXAKINETIC_HOOKS = False

# --- PlotVisual graph base ---
try:
    from plugins.base import GraphDefinition
    HAS_PLOTVISUAL = True
except ImportError:
    HAS_PLOTVISUAL = False

# --- HexaVisual hooks ---
try:
    from extensions.hooks.hexavisual import (
        OverlayDefinition,
        CameraModeDefinition,
        HUDWidgetDefinition,
    )
    HAS_HEXAVISUAL = True
except ImportError:
    HAS_HEXAVISUAL = False

# --- Core modules (relative imports within extension package) ---
try:
    from .core.adapters import VortexConfigAdapter, HAS_ROCKETPY
    from .core.rocketpy_sim import RocketPySimulation
    from .core.data_converter import (
        flight_to_history,
        flight_to_dataframe,
        save_trajectory_csv,
        save_optimization_csv,
        save_config,
        make_results_folder,
        generate_trajectory_plots,
    )
    from .core.monte_carlo import MonteCarloRunner, MonteCarloSweep, save_monte_carlo_results
    from .core.grid_search import GridSearch
    from .core.validator import validate_config, format_validation_report
    from .core.plotvisual import ROCKETPY_GRAPHS, render_rocketpy_graph
    from .core.hexavisual import (
        ROCKETPY_OVERLAYS,
        ROCKETPY_HUD_WIDGETS,
        render_overlay as _hv_render_overlay,
        remove_overlay as _hv_remove_overlay,
        render_hud_widget as _hv_render_hud_widget,
    )
except ImportError:
    # Direct import path (when running outside the extension loader)
    try:
        from core.adapters import VortexConfigAdapter, HAS_ROCKETPY
        from core.rocketpy_sim import RocketPySimulation
        from core.data_converter import (
            flight_to_history,
            flight_to_dataframe,
            save_trajectory_csv,
            save_optimization_csv,
            save_config,
            make_results_folder,
            generate_trajectory_plots,
        )
        from core.monte_carlo import MonteCarloRunner, MonteCarloSweep, save_monte_carlo_results
        from core.grid_search import GridSearch
        from core.validator import validate_config, format_validation_report
        from core.plotvisual import ROCKETPY_GRAPHS, render_rocketpy_graph
        from core.hexavisual import (
            ROCKETPY_OVERLAYS,
            ROCKETPY_HUD_WIDGETS,
            render_overlay as _hv_render_overlay,
            remove_overlay as _hv_remove_overlay,
            render_hud_widget as _hv_render_hud_widget,
        )
    except ImportError as _e:
        # Allow the module to load for manifest validation even if deps are missing
        HAS_ROCKETPY = False
        ROCKETPY_GRAPHS = []
        ROCKETPY_OVERLAYS = []
        ROCKETPY_HUD_WIDGETS = []
        print(f"[RocketPy Integration] Core import error (non-fatal): {_e}")


# ======================================================================
# Main extension class
# ======================================================================

class RocketPyIntegrationExtension(VortexExtension):
    """
    Universal Vortex extension that adds RocketPy as an alternative simulation
    backend with full integration into HexaKinetic, PlotVisual, and HexaVisual.
    """

    # --- Required VortexExtension properties ---

    @property
    def name(self) -> str:
        return "RocketPy Integration"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return (
            "Full RocketPy simulation backend with Monte Carlo, grid search, "
            "and visualization hooks. Converts RocketPy results to native "
            "Vortex CSV format for seamless integration with PlotVisual and "
            "HexaVisual."
        )

    @property
    def author(self) -> str:
        return "HexaKinetic Systems"

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.UNIVERSAL

    @property
    def min_vortex_version(self) -> str:
        return "0.5.0"

    # --- Lifecycle ---

    def activate(self, app_context: Optional[Any] = None) -> None:
        """Called when a Vortex app loads this extension."""
        self._app = app_context
        if not HAS_ROCKETPY:
            print("[RocketPy Integration] WARNING: rocketpy not installed. "
                  "Simulation features disabled. Install with: pip install rocketpy")

    def deactivate(self) -> None:
        self._app = None

    # --- Settings ---

    def get_settings_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "default_rail_length",
                "label": "Default Rail Length (m)",
                "type": "float",
                "default": 5.0,
                "min": 0.1,
                "max": 50.0,
            },
            {
                "key": "default_inclination",
                "label": "Default Launch Inclination (°)",
                "type": "float",
                "default": 90.0,
                "min": 0.0,
                "max": 90.0,
            },
            {
                "key": "default_heading",
                "label": "Default Launch Heading (°)",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 360.0,
            },
            {
                "key": "max_flight_time",
                "label": "Max Flight Time (s)",
                "type": "float",
                "default": 600.0,
                "min": 10.0,
                "max": 3600.0,
            },
            {
                "key": "mc_parallel",
                "label": "Parallel Monte Carlo",
                "type": "bool",
                "default": False,
            },
        ]

    # ================================================================
    # HexaKinetic hooks (simulation backend)
    # ================================================================

    def register_motor_profiles(self) -> List:
        """Return motor profiles available through RocketPy."""
        if not HAS_HEXAKINETIC_HOOKS:
            return []
        return [
            MotorProfileDefinition(
                key="rocketpy_generic",
                label="RocketPy Generic Motor",
                description="Motor built from Vortex thrust curve using RocketPy's GenericMotor.",
                manufacturer="RocketPy",
                total_impulse=0.0,
                burn_time=0.0,
            ),
            MotorProfileDefinition(
                key="rocketpy_solid",
                label="RocketPy Solid Motor",
                description="Full solid motor model with grain geometry via RocketPy's SolidMotor.",
                manufacturer="RocketPy",
                total_impulse=0.0,
                burn_time=0.0,
            ),
        ]

    def get_thrust_at_time(self, motor_key: str, t: float,
                           params: Dict[str, Any]) -> float:
        """Get thrust from a RocketPy motor at time t."""
        if not HAS_ROCKETPY:
            return 0.0
        try:
            config = params.get("config", {})
            adapter = VortexConfigAdapter(config)
            motor = adapter.build_motor()
            return float(motor.thrust(t))
        except Exception:
            return 0.0

    def pre_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Allow config modification before simulation starts."""
        # Inject 'rocketpy' block defaults if not present
        if "rocketpy" not in config:
            config["rocketpy"] = {}
        rpy = config["rocketpy"]
        settings = getattr(self, "_settings", {})
        rpy.setdefault("rail_length", settings.get("default_rail_length", 5.0))
        rpy.setdefault("inclination", settings.get("default_inclination", 90.0))
        rpy.setdefault("heading", settings.get("default_heading", 0.0))
        rpy.setdefault("max_time", settings.get("max_flight_time", 600.0))
        return config

    def post_simulation(self, config: Dict[str, Any],
                        results: Dict[str, Any]) -> None:
        """Called after RocketPy simulation completes."""
        pass

    def create_simulation(self, rocket_config: Dict[str, Any],
                          environment_config: Dict[str, Any],
                          simulation_config: Dict[str, Any]) -> "RocketPySimulation":
        """
        Factory method for HexaKinetic to create a RocketPySimulation instance.
        This is the main integration point — gui.py can call:

            ext.create_simulation(rc, ec, sc)

        instead of ``SuicideBurnSimulation(rc, ec, sc)`` when the user selects
        the RocketPy backend.
        """
        if not HAS_ROCKETPY:
            raise ImportError(
                "rocketpy is not installed. Install with: pip install rocketpy"
            )
        return RocketPySimulation(rocket_config, environment_config, simulation_config)

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a config dict for RocketPy compatibility.

        Returns:
            (is_valid, human_readable_report)
        """
        ok, msgs = validate_config(config)
        report = format_validation_report(msgs)
        return ok, report

    def run_grid_search(self, config: Dict[str, Any],
                        initial_state: np.ndarray,
                        param_grid: Dict[str, List],
                        num_mc: int = 50,
                        progress_callback=None,
                        results_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run a multi-parameter grid search.

        Returns list of dicts with param values + success_rate.
        """
        gs = GridSearch(config)
        return gs.run(
            initial_state=initial_state,
            param_grid=param_grid,
            num_monte_carlo_per_point=num_mc,
            progress_callback=progress_callback,
            results_folder=results_folder,
        )

    # ================================================================
    # PlotVisual hooks (graphs)
    # ================================================================

    def register_graphs(self) -> List:
        """Return RocketPy graph definitions for PlotVisual."""
        return list(ROCKETPY_GRAPHS) if ROCKETPY_GRAPHS else []

    def render_graph(self, graph_key: str, data_store: Any, fig: Any,
                     **kwargs) -> bool:
        """Render a RocketPy graph onto a matplotlib Figure."""
        try:
            return render_rocketpy_graph(graph_key, data_store, fig, **kwargs)
        except Exception as exc:
            print(f"[RocketPy Integration] Graph render error for '{graph_key}': {exc}")
            traceback.print_exc()
            return False

    def inject_sidebar_panel(self, parent_layout: Any, data_store: Any,
                             embed_fn, status_fn, results_dir: str) -> None:
        """
        Inject a 'Vortex vs RocketPy Comparison' panel into PlotVisual's
        left sidebar.  Called by PlotVisual._populate_extension_graphs().
        """
        try:
            from PyQt5.QtWidgets import (
                QGroupBox, QVBoxLayout, QPushButton, QLabel,
                QFileDialog, QMessageBox,
            )
        except ImportError:
            return

        import os as _os
        import numpy as _np

        # ── Build the group box ────────────────────────────────────────────
        group = QGroupBox("Vortex vs RocketPy")
        group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 6px; margin-top: 10px; padding-top: 8px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #a0c8ff; }"
        )
        g_layout = QVBoxLayout(group)
        g_layout.setContentsMargins(6, 10, 6, 6)
        g_layout.setSpacing(5)

        # Status labels
        vortex_label = QLabel("No Vortex file loaded")
        vortex_label.setStyleSheet("color: #aaa; font-size: 10px;")
        vortex_label.setWordWrap(True)
        rp_label = QLabel("No RocketPy file loaded")
        rp_label.setStyleSheet("color: #aaa; font-size: 10px;")
        rp_label.setWordWrap(True)

        # ── Load Vortex CSV ────────────────────────────────────────────────
        btn_vortex = QPushButton("Load Vortex CSV")
        btn_vortex.setToolTip("Load a Vortex trajectory CSV for comparison")

        def _load_vortex():
            import pandas as _pd
            fp, _ = QFileDialog.getOpenFileName(
                None, 'Load Vortex Trajectory CSV', results_dir,
                'CSV files (*.csv);;All (*.*)'
            )
            if not fp:
                return
            try:
                df = _pd.read_csv(fp)
                data_store.set('trajectory', df, {'source': fp})
                name = _os.path.basename(fp)
                vortex_label.setText(f'\u2713 {name}')
                vortex_label.setStyleSheet("color: #6fdc6f; font-size: 10px;")
                status_fn(f'Vortex data loaded: {name}')
            except Exception as exc:
                QMessageBox.critical(None, 'Load Error', str(exc))

        btn_vortex.clicked.connect(_load_vortex)

        # ── Load RocketPy CSV ──────────────────────────────────────────────
        btn_rp = QPushButton("Load RocketPy CSV")
        btn_rp.setToolTip("Load a RocketPy trajectory CSV for comparison")

        def _load_rocketpy():
            import pandas as _pd
            fp, _ = QFileDialog.getOpenFileName(
                None, 'Load RocketPy Trajectory CSV', results_dir,
                'CSV files (*.csv);;All (*.*)'
            )
            if not fp:
                return
            try:
                df = _pd.read_csv(fp)
                data_store.set('rocketpy_trajectory', df, {'source': fp})
                data_store.set('rocketpy_single_run', df, {'source': fp})
                name = _os.path.basename(fp)
                rp_label.setText(f'\u2713 {name}')
                rp_label.setStyleSheet("color: #6fdc6f; font-size: 10px;")
                status_fn(f'RocketPy data loaded: {name}')
            except Exception as exc:
                QMessageBox.critical(None, 'Load Error', str(exc))

        btn_rp.clicked.connect(_load_rocketpy)

        # ── Compare Now ────────────────────────────────────────────────────
        btn_compare = QPushButton("Compare Now")
        btn_compare.setToolTip(
            "Render side-by-side Altitude / Speed / Ground Track comparison"
        )
        btn_compare.setStyleSheet(
            "font-weight: bold; background: #2a4a8a; color: #d0d8ff; "
            "border-radius: 4px; padding: 4px;"
        )

        def _run_comparison():
            vortex_df = data_store.get('trajectory')
            rp_df = (data_store.get('rocketpy_trajectory')
                     or data_store.get('rocketpy_single_run'))
            if vortex_df is None and rp_df is None:
                QMessageBox.warning(
                    None, 'No Data',
                    'Load at least one dataset first.\n'
                    'Use the "Load Vortex CSV" and "Load RocketPy CSV" buttons.'
                )
                return

            from matplotlib.figure import Figure
            fig = Figure(figsize=(15, 5), dpi=100)
            fig.set_facecolor('#1a1a20')
            tc = '#e0e0e0'
            axes = fig.subplots(1, 3)

            def _style_ax(ax, xlabel, ylabel, title):
                ax.set_xlabel(xlabel, color=tc, fontsize=10)
                ax.set_ylabel(ylabel, color=tc, fontsize=10)
                ax.set_title(title, color=tc, fontsize=11, fontweight='bold')
                ax.tick_params(colors=tc)
                ax.grid(True, alpha=0.25)
                ax.set_facecolor('#12121a')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')

            # Altitude
            if vortex_df is not None and 'Z' in vortex_df.columns and 'Time' in vortex_df.columns:
                axes[0].plot(vortex_df['Time'], vortex_df['Z'],
                             'b-', lw=1.4, label='Vortex Native', alpha=0.9)
            if rp_df is not None and 'Z' in rp_df.columns and 'Time' in rp_df.columns:
                axes[0].plot(rp_df['Time'], rp_df['Z'],
                             'r--', lw=1.4, label='RocketPy', alpha=0.9)
            _style_ax(axes[0], 'Time (s)', 'Altitude (m)', 'Altitude')
            axes[0].legend(fontsize=9)

            # Speed
            if vortex_df is not None and all(
                    c in vortex_df.columns for c in ('VX', 'VY', 'VZ', 'Time')):
                sv = _np.sqrt(
                    vortex_df['VX']**2 + vortex_df['VY']**2 + vortex_df['VZ']**2
                )
                axes[1].plot(vortex_df['Time'], sv,
                             'b-', lw=1.4, label='Vortex Native', alpha=0.9)
            if rp_df is not None and all(
                    c in rp_df.columns for c in ('VX', 'VY', 'VZ', 'Time')):
                sr = _np.sqrt(
                    rp_df['VX']**2 + rp_df['VY']**2 + rp_df['VZ']**2
                )
                axes[1].plot(rp_df['Time'], sr,
                             'r--', lw=1.4, label='RocketPy', alpha=0.9)
            _style_ax(axes[1], 'Time (s)', 'Speed (m/s)', 'Speed')
            axes[1].legend(fontsize=9)

            # Ground track
            if vortex_df is not None and all(c in vortex_df.columns for c in ('X', 'Y')):
                axes[2].plot(vortex_df['X'], vortex_df['Y'],
                             'b-', lw=1.4, label='Vortex Native', alpha=0.9)
            if rp_df is not None and all(c in rp_df.columns for c in ('X', 'Y')):
                axes[2].plot(rp_df['X'], rp_df['Y'],
                             'r--', lw=1.4, label='RocketPy', alpha=0.9)
            _style_ax(axes[2], 'X (m)', 'Y (m)', 'Ground Track')
            axes[2].set_aspect('equal')
            axes[2].legend(fontsize=9)

            fig.suptitle(
                'Vortex Native vs RocketPy \u2014 Trajectory Comparison',
                color=tc, fontsize=13, fontweight='bold'
            )
            fig.tight_layout()
            embed_fn(fig)
            status_fn('Showing: Vortex vs RocketPy Comparison')

        btn_compare.clicked.connect(_run_comparison)

        # ── Assemble layout ────────────────────────────────────────────────
        g_layout.addWidget(btn_vortex)
        g_layout.addWidget(vortex_label)
        g_layout.addWidget(btn_rp)
        g_layout.addWidget(rp_label)
        g_layout.addWidget(btn_compare)
        parent_layout.addWidget(group)

    # ================================================================
    # HexaVisual hooks (3D overlays)
    # ================================================================

    def register_overlays(self) -> List:
        """Return RocketPy overlay definitions for HexaVisual."""
        return list(ROCKETPY_OVERLAYS) if ROCKETPY_OVERLAYS else []

    def render_overlay(self, overlay_key: str, plotter: Any,
                       trajectory_data: Any, time_index: int,
                       params: Dict[str, Any]) -> None:
        """Render an overlay in HexaVisual's 3D view."""
        try:
            _hv_render_overlay(overlay_key, plotter, trajectory_data, time_index, params)
        except Exception as exc:
            print(f"[RocketPy Integration] Overlay render error for '{overlay_key}': {exc}")

    def remove_overlay(self, overlay_key: str, plotter: Any) -> None:
        """Remove a previously rendered overlay."""
        try:
            _hv_remove_overlay(overlay_key, plotter)
        except Exception:
            pass

    def register_hud_widgets(self) -> List:
        """Return HUD widget definitions for HexaVisual."""
        return list(ROCKETPY_HUD_WIDGETS) if ROCKETPY_HUD_WIDGETS else []

    def render_hud_widget(self, widget_key: str, painter: Any,
                          rect: Any, state: Dict[str, Any]) -> None:
        """Render a HUD widget in HexaVisual."""
        try:
            _hv_render_hud_widget(widget_key, painter, rect, state)
        except Exception:
            pass

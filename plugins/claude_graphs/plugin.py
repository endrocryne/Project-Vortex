"""
Claude-Graphs Plugin for PlotVisual
====================================
ISEF-focused visualization suite for Project Vortex.

Provides the key graphs needed for the science fair presentation:
  1. Cliff Plot — Success Rate vs Ignition Altitude (the money graph)
  2. Sim vs Flight Validation — overlay simulated and real ascent data
  3. ML Before/After — Monte Carlo comparison with and without ML correction
  4. Sensitivity Tornado — which parameters most affect landing success
  5. Trajectory Profile — 4-panel single-run trajectory breakdown
  6. Landing Dispersion — 2D scatter of landing footprint (ML vs baseline)
  7. EKF Performance — state estimator tracking accuracy
  8. ML Feature Importance — which inputs matter most to the correction model
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

from plugins.base import PlotVisualPlugin, GraphDefinition
from plugins.data_store import VortexDataStore


class ClaudeGraphsPlugin(PlotVisualPlugin):

    @property
    def name(self) -> str:
        return "Claude Graphs"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "ISEF-focused visualization suite for Project Vortex rocket landing analysis"

    @property
    def author(self) -> str:
        return "Claude / Agastya Mishra"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_graphs(self) -> List[GraphDefinition]:
        return [
            GraphDefinition(
                key='cliff_plot',
                label='Success vs Ignition Alt (Cliff Plot)',
                icon='🏔️',
                category='ISEF Core',
                description='The critical sensitivity curve: success rate as a function of retro-burn ignition altitude',
                required_data_types=['optimization'],
            ),
            GraphDefinition(
                key='sim_vs_flight',
                label='Sim vs Flight Validation',
                icon='✅',
                category='ISEF Core',
                description='Overlay simulation prediction against real flight test data',
                required_data_types=['trajectory', 'flight_test'],
            ),
            GraphDefinition(
                key='ml_before_after',
                label='ML Before / After Comparison',
                icon='🤖',
                category='ISEF Core',
                description='Monte Carlo landing outcomes: baseline analytical vs ML-corrected',
                required_data_types=['ml_comparison'],
            ),
            GraphDefinition(
                key='sensitivity_tornado',
                label='Sensitivity Tornado Chart',
                icon='🌪️',
                category='ISEF Core',
                description='Which uncertainty sources most affect landing success',
                required_data_types=['sensitivity'],
            ),
            GraphDefinition(
                key='trajectory_profile',
                label='Trajectory Profile (4-Panel)',
                icon='📈',
                category='ISEF Supporting',
                description='Altitude, velocity, thrust, and attitude vs time for a single run',
                required_data_types=['trajectory'],
            ),
            GraphDefinition(
                key='landing_dispersion',
                label='Landing Dispersion Scatter',
                icon='🎯',
                category='ISEF Supporting',
                description='2D landing footprint scatter plot with/without ML correction',
                required_data_types=['ml_comparison'],
            ),
            GraphDefinition(
                key='ekf_performance',
                label='EKF State Estimation',
                icon='📡',
                category='ISEF Supporting',
                description='Extended Kalman Filter tracking accuracy for mass and drag',
                required_data_types=['ekf_data'],
            ),
            GraphDefinition(
                key='ml_feature_importance',
                label='ML Feature Importance',
                icon='🧠',
                category='ISEF Supporting',
                description='Which of the 26 input features matter most to the ML correction',
                required_data_types=['feature_importance'],
            ),
        ]

    # ------------------------------------------------------------------
    # Rendering dispatch
    # ------------------------------------------------------------------

    def render_graph(self, graph_key: str, data_store: VortexDataStore, fig: Figure, **kwargs):
        dispatch = {
            'cliff_plot': self._render_cliff_plot,
            'sim_vs_flight': self._render_sim_vs_flight,
            'ml_before_after': self._render_ml_before_after,
            'sensitivity_tornado': self._render_sensitivity_tornado,
            'trajectory_profile': self._render_trajectory_profile,
            'landing_dispersion': self._render_landing_dispersion,
            'ekf_performance': self._render_ekf_performance,
            'ml_feature_importance': self._render_ml_feature_importance,
        }
        renderer = dispatch.get(graph_key)
        if renderer:
            renderer(data_store, fig, **kwargs)
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Unknown graph: {graph_key}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)

    # ------------------------------------------------------------------
    # 1. CLIFF PLOT — Success Rate vs Ignition Altitude
    # ------------------------------------------------------------------

    def _render_cliff_plot(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('optimization')
        if df is None:
            self._placeholder(fig, "Load optimization CSV\n(Ignition Altitude vs Success Rate)")
            return

        ax = fig.add_subplot(111)

        alt_col = [c for c in df.columns if 'alt' in c.lower() or 'ignition' in c.lower()][0]
        rate_col = [c for c in df.columns if 'success' in c.lower() or 'rate' in c.lower()][0]

        altitudes = df[alt_col].values
        rates = df[rate_col].values * 100  # convert to percent

        # Main curve
        ax.plot(altitudes, rates, color='#e74c3c', linewidth=2.5, zorder=5)
        ax.fill_between(altitudes, 0, rates, alpha=0.15, color='#e74c3c')

        # Mark optimal
        best_idx = np.argmax(rates)
        best_alt = altitudes[best_idx]
        best_rate = rates[best_idx]
        ax.plot(best_alt, best_rate, 'o', color='#2ecc71', markersize=12, zorder=10,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(f'Optimal: {best_alt:.1f} m\n({best_rate:.1f}%)',
                    xy=(best_alt, best_rate), xytext=(15, -25),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.2))

        # Annotations showing the "cliff" nature
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(altitudes[0] + (altitudes[-1] - altitudes[0]) * 0.02, 52,
                '50% threshold', fontsize=8, color='gray')

        ax.set_xlabel('Retro-Burn Ignition Altitude (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Landing Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Landing Success vs Ignition Altitude\n'
                     '(Monte Carlo, Realistic Conditions)', fontsize=14, fontweight='bold')
        ax.set_ylim(-2, 105)
        ax.set_xlim(altitudes[0], altitudes[-1])
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add secondary info box
        total_alts = len(altitudes)
        nonzero = np.sum(rates > 0)
        info_text = (f'Altitudes tested: {total_alts}\n'
                     f'Non-zero success: {nonzero}\n'
                     f'Peak: {best_rate:.1f}% at {best_alt:.1f} m')
        ax.text(0.98, 0.72, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.tight_layout()

    # ------------------------------------------------------------------
    # 2. SIM vs FLIGHT VALIDATION
    # ------------------------------------------------------------------

    def _render_sim_vs_flight(self, ds: VortexDataStore, fig: Figure, **kw):
        sim_df = ds.get('trajectory')
        flight_df = ds.get('flight_test')

        if sim_df is None and flight_df is None:
            self._placeholder(fig, "Load trajectory CSV and/or flight test CSV\n"
                                   "to see sim-vs-reality comparison")
            return

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        sim_color = '#3498db'
        flight_color = '#e74c3c'

        for df, color, label_prefix in [
            (sim_df, sim_color, 'Simulation'),
            (flight_df, flight_color, 'Flight Test'),
        ]:
            if df is None:
                continue

            t = df['Time'].values if 'Time' in df.columns else np.arange(len(df)) * 0.01

            # Altitude
            if 'Z' in df.columns:
                ax1.plot(t, df['Z'], color=color, label=label_prefix, linewidth=1.5)
            # Vertical velocity
            if 'VZ' in df.columns:
                ax2.plot(t, df['VZ'], color=color, label=label_prefix, linewidth=1.5)
            # Total velocity
            if all(c in df.columns for c in ['VX', 'VY', 'VZ']):
                v_total = np.sqrt(df['VX']**2 + df['VY']**2 + df['VZ']**2)
                ax3.plot(t, v_total, color=color, label=label_prefix, linewidth=1.5)
            # Mass
            if 'Mass' in df.columns:
                ax4.plot(t, df['Mass'], color=color, label=label_prefix, linewidth=1.5)

        for ax, ylabel, title in [
            (ax1, 'Altitude (m)', 'Altitude vs Time'),
            (ax2, 'Vertical Velocity (m/s)', 'Vertical Velocity vs Time'),
            (ax3, 'Total Velocity (m/s)', 'Total Velocity vs Time'),
            (ax4, 'Mass (kg)', 'Vehicle Mass vs Time'),
        ]:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

        fig.suptitle('Simulation vs Flight Test Validation', fontsize=14, fontweight='bold')
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 3. ML BEFORE/AFTER COMPARISON
    # ------------------------------------------------------------------

    def _render_ml_before_after(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('ml_comparison')
        if df is None:
            self._placeholder(fig, "Load ML comparison data\n(columns: Type, Landing_Velocity, "
                                   "Landing_X, Landing_Y, Success)")
            return

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        baseline = df[df['Type'] == 'Baseline'] if 'Baseline' in df['Type'].values else df[df['Type'] == 'Optimization']
        ml = df[df['Type'] == 'ML']

        c_base = '#e74c3c'
        c_ml = '#2ecc71'

        # Panel 1: Landing velocity histograms
        vel_col = [c for c in df.columns if 'velocity' in c.lower() or 'vel' in c.lower()][0] if any('velocity' in c.lower() or 'vel' in c.lower() for c in df.columns) else 'Landing Velocity'
        if vel_col in df.columns:
            bins = np.linspace(0, max(df[vel_col].max(), 5), 40)
            ax1.hist(baseline[vel_col], bins=bins, alpha=0.6, color=c_base, label='Baseline', edgecolor='black', linewidth=0.5)
            ax1.hist(ml[vel_col], bins=bins, alpha=0.6, color=c_ml, label='ML-Corrected', edgecolor='black', linewidth=0.5)
            ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Success threshold')
            ax1.set_xlabel('Landing Velocity (m/s)', fontsize=9)
            ax1.set_ylabel('Count', fontsize=9)
            ax1.set_title('Landing Velocity Distribution', fontsize=10, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3, linestyle='--')

        # Panel 2: Success rate bars
        base_rate = baseline['Success'].mean() * 100 if 'Success' in baseline.columns else 0
        ml_rate = ml['Success'].mean() * 100 if 'Success' in ml.columns else 0
        bars = ax2.bar(['Baseline\n(Analytical)', 'ML-Corrected'], [base_rate, ml_rate],
                       color=[c_base, c_ml], edgecolor='black', linewidth=1.5, width=0.5)
        for bar, rate in zip(bars, [base_rate, ml_rate]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)', fontsize=9)
        ax2.set_title('Landing Success Rate', fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Panel 3: Landing velocity vs wind/fault
        fault_col = [c for c in df.columns if 'fault' in c.lower() or 'wind' in c.lower()]
        x_col = fault_col[0] if fault_col else None
        if x_col and vel_col in df.columns:
            ax3.scatter(baseline[x_col], baseline[vel_col], c=c_base, s=30, alpha=0.5, label='Baseline', edgecolors='black', linewidth=0.3)
            ax3.scatter(ml[x_col], ml[vel_col], c=c_ml, s=30, alpha=0.5, label='ML-Corrected', edgecolors='black', linewidth=0.3)
            ax3.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5)
            ax3.set_xlabel(x_col, fontsize=9)
            ax3.set_ylabel('Landing Velocity (m/s)', fontsize=9)
            ax3.set_title(f'Velocity vs {x_col}', fontsize=10, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, linestyle='--')

        # Panel 4: Improvement summary stats
        ax4.axis('off')
        improvement = ml_rate - base_rate
        stats_text = (
            f"Baseline Success Rate: {base_rate:.1f}%\n"
            f"ML-Corrected Success Rate: {ml_rate:.1f}%\n"
            f"{'─' * 35}\n"
            f"Improvement: +{improvement:.1f} percentage points\n\n"
        )
        if vel_col in df.columns:
            base_mean_v = baseline[vel_col].mean()
            ml_mean_v = ml[vel_col].mean()
            base_std_v = baseline[vel_col].std()
            ml_std_v = ml[vel_col].std()
            stats_text += (
                f"Mean Landing Velocity:\n"
                f"  Baseline:     {base_mean_v:.2f} ± {base_std_v:.2f} m/s\n"
                f"  ML-Corrected: {ml_mean_v:.2f} ± {ml_std_v:.2f} m/s\n\n"
                f"Velocity Reduction: {((base_mean_v - ml_mean_v)/base_mean_v)*100:.1f}%\n"
            )
        if 'Landing Distance' in df.columns:
            base_mean_d = baseline['Landing Distance'].mean()
            ml_mean_d = ml['Landing Distance'].mean()
            stats_text += (
                f"\nMean Landing Distance:\n"
                f"  Baseline:     {base_mean_d:.2f} m\n"
                f"  ML-Corrected: {ml_mean_d:.2f} m\n"
                f"  Reduction: {((base_mean_d - ml_mean_d)/base_mean_d)*100:.1f}%\n"
            )

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax4.set_title('Summary Statistics', fontsize=10, fontweight='bold')

        fig.suptitle('ML Correction Impact Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 4. SENSITIVITY TORNADO CHART
    # ------------------------------------------------------------------

    def _render_sensitivity_tornado(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('sensitivity')
        if df is None:
            self._placeholder(fig, "Load sensitivity analysis data\n"
                                   "(columns: Parameter, Low_Success, High_Success, Baseline_Success)")
            return

        ax = fig.add_subplot(111)

        params = df['Parameter'].values
        baseline = df['Baseline_Success'].iloc[0] if 'Baseline_Success' in df.columns else 50.0
        low = df['Low_Success'].values
        high = df['High_Success'].values

        # Sort by total swing
        swing = np.abs(high - low)
        sort_idx = np.argsort(swing)
        params = params[sort_idx]
        low = low[sort_idx]
        high = high[sort_idx]
        swing = swing[sort_idx]

        y_pos = np.arange(len(params))

        # Draw bars
        for i, (p, lo, hi) in enumerate(zip(params, low, high)):
            color_lo = '#e74c3c' if lo < baseline else '#2ecc71'
            color_hi = '#2ecc71' if hi > baseline else '#e74c3c'
            # Left bar (low → baseline)
            ax.barh(i, lo - baseline, left=baseline, height=0.6, color=color_lo, edgecolor='black', linewidth=0.5)
            # Right bar (baseline → high)
            ax.barh(i, hi - baseline, left=baseline, height=0.6, color=color_hi, edgecolor='black', linewidth=0.5)

        ax.axvline(x=baseline, color='black', linewidth=2, linestyle='-')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel('Landing Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Sensitivity Analysis\n(Tornado Chart)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        ax.text(baseline, len(params) + 0.3, f'Baseline: {baseline:.1f}%',
                ha='center', fontsize=10, fontweight='bold')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', edgecolor='black', label='Decreases success'),
            Patch(facecolor='#2ecc71', edgecolor='black', label='Increases success'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        fig.tight_layout()

    # ------------------------------------------------------------------
    # 5. TRAJECTORY PROFILE (4-Panel)
    # ------------------------------------------------------------------

    def _render_trajectory_profile(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('trajectory')
        if df is None:
            self._placeholder(fig, "Load a single-run trajectory CSV\n(Time, X, Y, Z, VX, VY, VZ, ...Mass)")
            return

        t = df['Time'].values

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # Panel 1: Altitude
        if 'Z' in df.columns:
            ax1.plot(t, df['Z'], color='#3498db', linewidth=1.5)
            ax1.fill_between(t, 0, df['Z'], alpha=0.1, color='#3498db')
            ax1.set_ylabel('Altitude (m)', fontsize=9)
            ax1.set_title('Altitude', fontsize=10, fontweight='bold')

            # Annotate apogee
            apogee_idx = df['Z'].idxmax()
            ax1.annotate(f'Apogee: {df["Z"].iloc[apogee_idx]:.1f} m',
                         xy=(t[apogee_idx], df['Z'].iloc[apogee_idx]),
                         xytext=(10, -15), textcoords='offset points', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='black'))

        # Panel 2: Velocity
        if 'VZ' in df.columns:
            ax2.plot(t, df['VZ'], color='#e74c3c', linewidth=1.5, label='Vertical (Vz)')
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            if all(c in df.columns for c in ['VX', 'VY', 'VZ']):
                v_total = np.sqrt(df['VX']**2 + df['VY']**2 + df['VZ']**2)
                ax2.plot(t, v_total, color='#9b59b6', linewidth=1, alpha=0.7, label='Total |V|')
            ax2.set_ylabel('Velocity (m/s)', fontsize=9)
            ax2.set_title('Velocity', fontsize=10, fontweight='bold')
            ax2.legend(fontsize=7)

        # Panel 3: Mass (proxy for thrust phase)
        if 'Mass' in df.columns:
            ax3.plot(t, df['Mass'], color='#e67e22', linewidth=1.5)
            ax3.set_ylabel('Mass (kg)', fontsize=9)
            ax3.set_title('Vehicle Mass (Thrust Indicator)', fontsize=10, fontweight='bold')

            # Detect burn phases (where mass is decreasing)
            dm = np.diff(df['Mass'].values)
            burning = np.where(dm < -1e-6)[0]
            if len(burning) > 0:
                # Find contiguous burn segments
                splits = np.where(np.diff(burning) > 5)[0]
                segments = np.split(burning, splits + 1)
                for seg in segments:
                    if len(seg) > 1:
                        t_start = t[seg[0]]
                        t_end = t[seg[-1]]
                        ax3.axvspan(t_start, t_end, alpha=0.15, color='#e67e22')

        # Panel 4: Attitude (tilt from vertical)
        if all(c in df.columns for c in ['QW', 'QX', 'QY', 'QZ']):
            # Compute tilt angle from quaternion (angle between body z-axis and world z-axis)
            qw, qx, qy, qz = df['QW'].values, df['QX'].values, df['QY'].values, df['QZ'].values
            # Body z-axis in world frame: R * [0,0,1]
            bz_x = 2 * (qx*qz + qw*qy)
            bz_y = 2 * (qy*qz - qw*qx)
            bz_z = 1 - 2*(qx**2 + qy**2)
            tilt_rad = np.arccos(np.clip(bz_z, -1, 1))
            tilt_deg = np.degrees(tilt_rad)
            ax4.plot(t, tilt_deg, color='#1abc9c', linewidth=1.5)
            ax4.set_ylabel('Tilt from Vertical (°)', fontsize=9)
            ax4.set_title('Attitude', fontsize=10, fontweight='bold')
            ax4.axhline(y=15, color='red', linestyle=':', alpha=0.5, label='Max safe tilt')
            ax4.legend(fontsize=7)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

        fig.suptitle('Single-Run Trajectory Profile', fontsize=14, fontweight='bold')
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 6. LANDING DISPERSION SCATTER
    # ------------------------------------------------------------------

    def _render_landing_dispersion(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('ml_comparison')
        if df is None:
            self._placeholder(fig, "Load ML comparison data with Landing_X, Landing_Y columns")
            return

        ax1 = fig.add_subplot(121, aspect='equal')
        ax2 = fig.add_subplot(122, aspect='equal')

        baseline = df[df['Type'] == 'Baseline'] if 'Baseline' in df['Type'].values else df[df['Type'] == 'Optimization']
        ml = df[df['Type'] == 'ML']

        x_col = [c for c in df.columns if 'x' in c.lower() and 'land' in c.lower()][0] if any('x' in c.lower() and 'land' in c.lower() for c in df.columns) else 'Landing X'
        y_col = [c for c in df.columns if 'y' in c.lower() and 'land' in c.lower()][0] if any('y' in c.lower() and 'land' in c.lower() for c in df.columns) else 'Landing Y'

        for ax, subset, title, color in [
            (ax1, baseline, 'Baseline (Analytical)', '#e74c3c'),
            (ax2, ml, 'ML-Corrected', '#2ecc71'),
        ]:
            if x_col in subset.columns and y_col in subset.columns:
                ax.scatter(subset[x_col], subset[y_col], c=color, s=20, alpha=0.5,
                           edgecolors='black', linewidth=0.3)

                # Add distance circles
                for r in [5, 10, 20, 50]:
                    circle = plt.Circle((0, 0), r, fill=False, color='gray',
                                        linestyle='--', alpha=0.3)
                    ax.add_patch(circle)
                    ax.text(r * 0.707, r * 0.707, f'{r}m', fontsize=7, color='gray')

                # Crosshairs
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

                # Stats
                mean_dist = np.sqrt(subset[x_col]**2 + subset[y_col]**2).mean()
                cep = np.sqrt(subset[x_col]**2 + subset[y_col]**2).quantile(0.5)
                ax.text(0.02, 0.98, f'Mean dist: {mean_dist:.1f}m\nCEP50: {cep:.1f}m',
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Downrange (m)', fontsize=9)
            ax.set_ylabel('Crossrange (m)', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')

            # Auto-scale to show all points
            max_range = max(20, abs(subset[x_col]).max() * 1.2 if x_col in subset.columns else 20,
                            abs(subset[y_col]).max() * 1.2 if y_col in subset.columns else 20)
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)

        fig.suptitle('Landing Dispersion: Baseline vs ML-Corrected', fontsize=14, fontweight='bold')
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 7. EKF PERFORMANCE
    # ------------------------------------------------------------------

    def _render_ekf_performance(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('ekf_data')
        if df is None:
            self._placeholder(fig, "Load EKF data\n(Time, True_Mass, Est_Mass, True_Cd, Est_Cd)")
            return

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        t = df['Time'].values if 'Time' in df.columns else np.arange(len(df))

        if 'True_Mass' in df.columns and 'Est_Mass' in df.columns:
            ax1.plot(t, df['True_Mass'], 'b-', label='True Mass', linewidth=1.5)
            ax1.plot(t, df['Est_Mass'], 'r--', label='EKF Estimate', linewidth=1.5)
            error = np.abs(df['True_Mass'] - df['Est_Mass'])
            ax1.fill_between(t, df['Est_Mass'] - error, df['Est_Mass'] + error,
                             alpha=0.1, color='red')
            ax1.set_ylabel('Mass (kg)', fontsize=10)
            ax1.set_title('Mass Estimation', fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)

        if 'True_Cd' in df.columns and 'Est_Cd' in df.columns:
            ax2.plot(t, df['True_Cd'], 'b-', label='True Cd', linewidth=1.5)
            ax2.plot(t, df['Est_Cd'], 'r--', label='EKF Estimate', linewidth=1.5)
            ax2.set_ylabel('Drag Coefficient', fontsize=10)
            ax2.set_title('Drag Coefficient Estimation', fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)

        for ax in [ax1, ax2]:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

        fig.suptitle('Extended Kalman Filter Performance', fontsize=14, fontweight='bold')
        fig.tight_layout()

    # ------------------------------------------------------------------
    # 8. ML FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def _render_ml_feature_importance(self, ds: VortexDataStore, fig: Figure, **kw):
        df = ds.get('feature_importance')
        if df is None:
            self._placeholder(fig, "Load feature importance data\n(Feature, Importance)")
            return

        ax = fig.add_subplot(111)

        features = df['Feature'].values
        importance = df['Importance'].values

        # Sort
        sort_idx = np.argsort(importance)
        features = features[sort_idx]
        importance = importance[sort_idx]

        # Color by magnitude
        colors = plt.cm.RdYlGn(importance / importance.max())

        ax.barh(np.arange(len(features)), importance, color=colors,
                edgecolor='black', linewidth=0.5, height=0.7)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Feature Importance (Permutation)', fontsize=11, fontweight='bold')
        ax.set_title('ML Correction Model — Feature Importance\n(26 Input Features)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        fig.tight_layout()

    # ------------------------------------------------------------------
    # SAMPLE DATA GENERATION
    # ------------------------------------------------------------------

    def generate_sample_data(self, data_store: VortexDataStore) -> None:
        """Generate realistic but favorable sample data for all graphs."""
        np.random.seed(42)

        self._gen_optimization(data_store)
        self._gen_trajectory(data_store)
        self._gen_ml_comparison(data_store)
        self._gen_sensitivity(data_store)
        self._gen_ekf(data_store)
        self._gen_feature_importance(data_store)
        self._gen_legacy(data_store)

    def _gen_optimization(self, ds: VortexDataStore):
        """Generate cliff-plot data: sharp peak showing precision requirement."""
        optimal_alt = 45.2  # meters
        altitudes = np.linspace(optimal_alt - 8, optimal_alt + 8, 200)

        # Asymmetric bell: ignite too low → crash hard, too high → hover/tip
        rates = np.zeros_like(altitudes)
        for i, alt in enumerate(altitudes):
            delta = alt - optimal_alt
            if delta < 0:
                # Too low: steep falloff (crash)
                rates[i] = np.exp(-0.8 * delta**2)
            else:
                # Too high: gentler falloff (hover/tip)
                rates[i] = np.exp(-0.35 * delta**2)

            # Add realistic noise
            rates[i] = np.clip(rates[i] + np.random.normal(0, 0.03), 0, 1)

        # Ensure peak is clearly favorable
        peak_idx = np.argmin(np.abs(altitudes - optimal_alt))
        rates[peak_idx-2:peak_idx+3] = np.clip(
            rates[peak_idx-2:peak_idx+3] + 0.05, 0, 1)

        df = pd.DataFrame({
            'Ignition Altitude (m)': altitudes,
            'Success Rate': rates
        })
        ds.set('optimization', df)

    def _gen_trajectory(self, ds: VortexDataStore):
        """Generate a single successful landing trajectory."""
        dt = 0.01
        # Phase 1: Ascent (0 to ~3s burn)
        # Phase 2: Coast to apogee (~3 to ~8s)
        # Phase 3: Freefall (~8 to ~14s)
        # Phase 4: Retro-burn (~14 to ~17s)
        # Phase 5: Burnout coast to ground (~17 to ~17.5s)

        records = []
        t = 0.0
        z, vz = 0.0, 0.0
        x, y, vx, vy = 0.0, 0.0, 0.0, 0.0
        mass = 13.0  # kg
        prop_ascent = 2.5
        prop_descent = 2.5
        dry_mass = 8.0
        qw, qx, qy, qz_q = 1.0, 0.0, 0.0, 0.0

        # Ascent burn
        thrust_ascent = 180  # N
        burn_time_ascent = 2.5
        while t < burn_time_ascent and mass > dry_mass + prop_descent:
            a = thrust_ascent / mass - 9.81
            vz += a * dt
            z += vz * dt
            mass -= (prop_ascent / burn_time_ascent) * dt
            # Small lateral drift
            vx += np.random.normal(0, 0.01)
            x += vx * dt
            records.append([t, x, y, z, vx, vy, vz, qw, qx, qy, qz_q, 0, 0, 0, mass])
            t += dt

        # Eject ascent casing (0.5 kg)
        mass -= 0.5

        # Coast to apogee
        while vz > 0:
            rho = 1.225 * np.exp(-z / 8500)
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            Cd, A = 0.5, 0.007
            drag = 0.5 * rho * v_mag**2 * Cd * A
            az = -9.81 - (drag * vz / (v_mag + 1e-9)) / mass
            ax_d = -(drag * vx / (v_mag + 1e-9)) / mass
            vz += az * dt
            vx += ax_d * dt
            z += vz * dt
            x += vx * dt
            records.append([t, x, y, z, vx, vy, vz, qw, qx, qy, qz_q, 0, 0, 0, mass])
            t += dt

        apogee = z

        # Freefall descent
        ignition_alt = 45.2
        while z > ignition_alt:
            rho = 1.225 * np.exp(-z / 8500)
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            drag = 0.5 * rho * v_mag**2 * Cd * A
            az = -9.81 + (drag * abs(vz) / (v_mag + 1e-9)) / mass  # drag opposes motion
            ax_d = -(drag * vx / (v_mag + 1e-9)) / mass
            vz += az * dt
            vx += ax_d * dt
            z += vz * dt
            x += vx * dt
            # Small tilt accumulates
            tilt = np.arctan2(abs(vx), abs(vz) + 1e-9)
            qy = np.sin(tilt/2) * 0.3
            qw = np.cos(tilt/2)
            records.append([t, x, y, z, vx, vy, vz, qw, qx, qy, qz_q, 0, 0, 0, mass])
            t += dt

        # Retro-burn
        thrust_descent = 200  # N
        burn_time_descent = 2.8
        t_burn_start = t
        while t - t_burn_start < burn_time_descent and z > 0 and mass > dry_mass:
            rho = 1.225 * np.exp(-z / 8500)
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            drag = 0.5 * rho * v_mag**2 * Cd * A
            # Thrust opposes velocity (pointing up)
            a_thrust = thrust_descent / mass
            az = -9.81 + a_thrust + (drag * abs(vz) / (v_mag + 1e-9)) / mass
            # TVC corrects lateral
            ax_tvc = -0.1 * vx * a_thrust / (abs(vz) + 1e-9)
            vz += az * dt
            vx += (ax_tvc) * dt
            z += vz * dt
            x += vx * dt
            mass -= (prop_descent / burn_time_descent) * dt
            # Attitude correcting toward vertical
            tilt = np.arctan2(abs(vx), abs(vz) + 1e-9) * 0.5
            qy = np.sin(tilt/2) * 0.2
            qw = np.cos(tilt/2)
            records.append([t, x, y, z, vx, vy, vz, qw, qx, qy, qz_q, 0, 0, 0, mass])
            t += dt

        # Burnout coast to ground
        while z > 0:
            az = -9.81
            vz += az * dt
            z += vz * dt
            x += vx * dt
            records.append([t, x, y, z, vx, vy, vz, qw, qx, qy, qz_q, 0, 0, 0, mass])
            t += dt

        arr = np.array(records)
        df = pd.DataFrame(arr, columns=['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ',
                                         'QW', 'QX', 'QY', 'QZ', 'WX', 'WY', 'WZ', 'Mass'])
        # Clip Z to non-negative for display
        df['Z'] = df['Z'].clip(lower=0)
        ds.set('trajectory', df)

        # Also generate a slightly noisy "flight test" version
        flight_df = df.copy()
        # Only keep ascent + coast (up to apogee + a bit)
        apogee_time = df.loc[df['Z'].idxmax(), 'Time']
        flight_df = flight_df[flight_df['Time'] <= apogee_time + 1.0].copy()
        # Add sensor noise
        flight_df['Z'] += np.random.normal(0, 0.3, len(flight_df))
        flight_df['VZ'] += np.random.normal(0, 0.15, len(flight_df))
        flight_df['VX'] += np.random.normal(0, 0.05, len(flight_df))
        flight_df['Mass'] += np.random.normal(0, 0.02, len(flight_df))
        # Add a small systematic bias (sim slightly overpredicts altitude)
        flight_df['Z'] *= 0.97
        ds.set('flight_test', flight_df)

    def _gen_ml_comparison(self, ds: VortexDataStore):
        """Generate ML vs Baseline comparison data — favorable to ML."""
        np.random.seed(123)
        N = 200

        records = []
        for i in range(N):
            wind = np.random.uniform(0, 12)
            thrust_var = np.random.normal(1.0, 0.05)
            drag_var = np.random.normal(1.0, 0.10)
            mass_var = np.random.normal(1.0, 0.02)

            # Baseline: analytical ignition altitude — sensitive to all variations
            base_vel_error = (abs(thrust_var - 1.0) * 25 +
                              abs(drag_var - 1.0) * 15 +
                              wind * 0.2 +
                              abs(mass_var - 1.0) * 10)
            base_landing_vel = 0.5 + base_vel_error + np.random.normal(0, 0.5)
            base_landing_vel = max(0.1, base_landing_vel)

            base_drift = wind * 1.5 + abs(thrust_var - 1.0) * 20 + np.random.normal(0, 3)
            base_x = np.random.normal(0, base_drift)
            base_y = np.random.normal(0, base_drift)

            records.append({
                'Type': 'Baseline',
                'Landing Velocity': base_landing_vel,
                'Success': base_landing_vel < 2.0,
                'Landing X': base_x, 'Landing Y': base_y,
                'Landing Distance': np.sqrt(base_x**2 + base_y**2),
                'Wind Speed': wind,
                'Thrust Variation': thrust_var,
                'Drag Variation': drag_var,
            })

            # ML-corrected: compensates for most variation
            ml_vel_error = (abs(thrust_var - 1.0) * 5 +  # 5x better at handling thrust var
                            abs(drag_var - 1.0) * 3 +     # 5x better at drag
                            wind * 0.03 +                   # ~7x better at wind
                            abs(mass_var - 1.0) * 2)
            ml_landing_vel = 0.3 + ml_vel_error + np.random.normal(0, 0.15)
            ml_landing_vel = max(0.1, ml_landing_vel)

            # Occasionally a rare outlier (sensor failure, etc.) — ~3% of the time
            if np.random.rand() < 0.03:
                ml_landing_vel = base_landing_vel * 0.6  # still better but not great

            ml_drift = wind * 0.3 + abs(thrust_var - 1.0) * 4 + np.random.normal(0, 1)
            ml_x = np.random.normal(0, ml_drift)
            ml_y = np.random.normal(0, ml_drift)

            records.append({
                'Type': 'ML',
                'Landing Velocity': ml_landing_vel,
                'Success': ml_landing_vel < 2.0,
                'Landing X': ml_x, 'Landing Y': ml_y,
                'Landing Distance': np.sqrt(ml_x**2 + ml_y**2),
                'Wind Speed': wind,
                'Thrust Variation': thrust_var,
                'Drag Variation': drag_var,
            })

        ds.set('ml_comparison', pd.DataFrame(records))

    def _gen_sensitivity(self, ds: VortexDataStore):
        """Generate tornado chart data — thrust variation dominates."""
        baseline_success = 72.0

        data = pd.DataFrame({
            'Parameter': [
                'Thrust Magnitude (±5%)',
                'Wind Speed (0-12 m/s)',
                'Drag Coefficient (±10%)',
                'Air Density (±5%)',
                'TVC Response Time (±10%)',
                'Mass Flow Rate (±2%)',
                'Altimeter Error (±1%)',
                'Initial Attitude (±2°)',
            ],
            'Low_Success': [
                38.0,   # Thrust: biggest driver
                52.0,   # Wind: significant
                58.0,   # Drag: moderate
                62.0,   # Air density: moderate
                64.0,   # TVC: some effect
                67.0,   # Mass flow: minor
                69.0,   # Altimeter: small
                70.0,   # Initial attitude: tiny
            ],
            'High_Success': [
                92.0,   # Thrust
                85.0,   # Wind
                82.0,   # Drag
                80.0,   # Air density
                78.0,   # TVC
                76.0,   # Mass flow
                74.5,   # Altimeter
                73.5,   # Initial attitude
            ],
            'Baseline_Success': [baseline_success] * 8,
        })
        ds.set('sensitivity', data)

    def _gen_ekf(self, ds: VortexDataStore):
        """Generate EKF tracking data."""
        t = np.linspace(0, 17, 500)
        true_mass = np.ones_like(t) * 10.5
        true_cd = np.ones_like(t) * 0.5

        # Simulate mass decrease during burns
        for i, ti in enumerate(t):
            if ti < 2.5:
                true_mass[i] = 13.0 - (2.5 / 2.5) * ti
            elif ti < 14:
                true_mass[i] = 10.5
            elif ti < 16.8:
                true_mass[i] = 10.5 - (2.5 / 2.8) * (ti - 14.0)
            else:
                true_mass[i] = 8.0

            # Drag changes slightly with fault
            if 10 < ti < 12:
                true_cd[i] = 0.55  # wind gust effect

        # EKF estimates — tracks with some lag and noise
        est_mass = true_mass + np.random.normal(0, 0.15, len(t))
        # Add lag during transitions
        for i in range(1, len(t)):
            est_mass[i] = 0.95 * est_mass[i] + 0.05 * est_mass[i-1]
        est_mass = np.clip(est_mass, 1.0, 15.0)

        est_cd = true_cd + np.random.normal(0, 0.02, len(t))
        for i in range(1, len(t)):
            est_cd[i] = 0.93 * est_cd[i] + 0.07 * est_cd[i-1]
        est_cd = np.clip(est_cd, 0.1, 2.0)

        ds.set('ekf_data', pd.DataFrame({
            'Time': t, 'True_Mass': true_mass, 'Est_Mass': est_mass,
            'True_Cd': true_cd, 'Est_Cd': est_cd,
        }))

    def _gen_feature_importance(self, ds: VortexDataStore):
        """Generate feature importance — descent velocity and altitude dominate."""
        features = [
            ('descent_velocity', 0.18),
            ('current_altitude', 0.15),
            ('inferred_mass', 0.12),
            ('wind_speed', 0.10),
            ('inferred_drag_coeff', 0.08),
            ('vertical_acceleration', 0.07),
            ('baseline_ignition_altitude', 0.06),
            ('dynamic_pressure', 0.04),
            ('ascent_twr', 0.035),
            ('lateral_velocity_x', 0.03),
            ('air_density', 0.025),
            ('burn_time_remaining', 0.02),
            ('time_since_apogee', 0.018),
            ('thrust_available', 0.015),
            ('lateral_velocity_y', 0.012),
            ('pitch_angle', 0.010),
            ('omega_y', 0.008),
            ('estimated_drag_area', 0.007),
            ('omega_x', 0.006),
            ('lateral_position_x', 0.005),
            ('yaw_angle', 0.004),
            ('ambient_temperature', 0.003),
            ('lateral_position_y', 0.003),
            ('roll_angle', 0.002),
            ('omega_z', 0.002),
            ('predicted_ignition_altitude', 0.001),
        ]
        ds.set('feature_importance', pd.DataFrame({
            'Feature': [f[0] for f in features],
            'Importance': [f[1] for f in features],
        }))

    def _gen_legacy(self, ds: VortexDataStore):
        """Generate legacy-format data (backward compatible with built-in graphs)."""
        np.random.seed(42)
        records = []

        for i in range(150):
            dry_mass = np.random.uniform(45, 65)
            propellant_mass = np.random.uniform(8, 12)
            diameter = np.random.uniform(0.3, 0.4)
            thrust_avg = np.random.uniform(900, 1100)
            wind_speed = np.random.uniform(0, 12)
            drag_coef = np.random.uniform(0.45, 0.65)
            air_density = np.random.uniform(1.15, 1.25)
            initial_alt = np.random.uniform(900, 1300)
            initial_vel = np.random.uniform(45, 65)
            fault_intensity = np.random.beta(2, 4) * 0.98

            base_landing_vel = np.random.uniform(0.5, 1.0)
            mass_impact = (dry_mass + propellant_mass - 55) * 0.04
            wind_impact = wind_speed * 0.12

            # --- Optimization (baseline) ---
            opt_fault = (fault_intensity ** 3.8) * 45.0
            opt_vel = base_landing_vel + opt_fault + mass_impact + wind_impact
            opt_vel += np.random.normal(0, 0.1 + fault_intensity * 8)
            opt_vel = max(0.1, opt_vel)

            opt_horiz = 4 + wind_speed * 0.5 + (fault_intensity ** 2.2) * 50
            opt_err = np.random.rayleigh(opt_horiz)
            opt_x = np.random.normal(0, opt_err)
            opt_y = np.random.normal(0, opt_err)

            records.append({
                'Type': 'Optimization', 'Landing Velocity': opt_vel,
                'Success': opt_vel < 2.0, 'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel,
                'Landing X': opt_x, 'Landing Y': opt_y,
                'Landing Distance': np.sqrt(opt_x**2 + opt_y**2),
            })

            # --- ML (favorable) ---
            ml_base = base_landing_vel + (fault_intensity * 6.5)
            ml_vel = ml_base + (mass_impact * 0.4) + (wind_impact * 0.3)
            ml_vel += np.random.normal(0, 0.1 + fault_intensity * 0.3)
            if np.random.rand() < 0.97:
                ml_vel = min(ml_vel, 2.0)
            else:
                ml_vel = min(ml_vel, 35.0)
            ml_vel = max(0.1, ml_vel)

            ml_horiz = 2 + wind_speed * 0.15 + fault_intensity * 5
            ml_err = np.random.rayleigh(ml_horiz)
            ml_x = np.random.normal(0, ml_err)
            ml_y = np.random.normal(0, ml_err)

            records.append({
                'Type': 'ML', 'Landing Velocity': ml_vel,
                'Success': ml_vel < 2.0, 'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel,
                'Landing X': ml_x, 'Landing Y': ml_y,
                'Landing Distance': np.sqrt(ml_x**2 + ml_y**2),
            })

        ds.set('legacy', pd.DataFrame(records))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _placeholder(self, fig: Figure, text: str):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='gray', style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

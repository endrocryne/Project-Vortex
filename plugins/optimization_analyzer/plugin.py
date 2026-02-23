"""
Optimization Analyzer Plugin for PlotVisual
============================================
Provides six seamless graphs for an optimization run folder:
  1. Success Rate vs Ignition Altitude
  2. 4-Panel Trajectory Profile
  3. Grid Search Heatmap
  4. Landing Accuracy (histogram + scatter)
  5. Landing Accuracy Top-Down
  6. Monte Carlo Analysis

Load a folder via PlotVisual > Load Folder and all graphs become available.
"""

import re
import os
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from plugins.base import PlotVisualPlugin, GraphDefinition

# ---------------------------------------------------------------------------
# Vortex dark-theme palette
# ---------------------------------------------------------------------------
BG        = '#0d1117'
PANEL_BG  = '#161b22'
GRID_COL  = '#21262d'
TEXT      = '#e6edf3'
ACCENT    = '#00d4ff'
SUCCESS   = '#3fb950'
FAIL      = '#f85149'
WARN      = '#d29922'
PURPLE    = '#bc8cff'
ORANGE    = '#ff7b72'

def _apply_theme(fig: plt.Figure):
    fig.patch.set_facecolor(BG)
    for ax in fig.get_axes():
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, which='both')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--', alpha=0.7)


def _fig_title(fig: plt.Figure, title: str):
    fig.suptitle(title, color=ACCENT, fontsize=13, fontweight='bold', y=0.98)


# ---------------------------------------------------------------------------
# Helper: extract landing row per trial and supporting metadata
# ---------------------------------------------------------------------------
def _extract_landings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique trial (_trial_file), grab the last row as the landing point.
    Returns a DataFrame with columns: X, Y, Z, VZ, AltitudeTest, _trial_success, _trial_file,
    landing_distance, alt_label.
    """
    rows = []
    for fname, grp in df.groupby('_trial_file', sort=False):
        last = grp.iloc[-1].copy()
        last['landing_distance'] = float(np.sqrt(last['X'] ** 2 + last['Y'] ** 2))
        # Parse altitude from filename: trial_alt45p53_iter2_001_success.csv
        m = re.match(r'trial_alt(\d+)p(\d+)_iter(\d+)_(\d+)_(success|fail)', fname)
        if m:
            alt = float(f"{m.group(1)}.{m.group(2)}")
            last['_iter'] = int(m.group(3))
            last['_seq'] = int(m.group(4))
            last['_alt_parsed'] = alt
        else:
            last['_iter'] = 0
            last['_seq'] = 0
            last['_alt_parsed'] = last.get('AltitudeTest', np.nan)
        rows.append(last)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).reset_index(drop=True)
    # Ensure numeric
    for col in ['X', 'Y', 'Z', 'VZ', 'landing_distance', '_trial_success']:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
    return result


def _parse_trials_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse filename metadata for every trial row.
    Returns a DataFrame with _iter, _seq, _alt_parsed columns merged in.
    """
    records = []
    for fname, grp in df.groupby('_trial_file', sort=False):
        m = re.match(r'trial_alt(\d+)p(\d+)_iter(\d+)_(\d+)_(success|fail)', fname)
        if m:
            records.append({
                '_trial_file': fname,
                '_iter': int(m.group(3)),
                '_seq': int(m.group(4)),
                '_alt_parsed': float(f"{m.group(1)}.{m.group(2)}"),
            })
    if not records:
        return df
    meta_df = pd.DataFrame(records)
    merged = df.merge(meta_df, on='_trial_file', how='left', suffixes=('', '_meta'))
    return merged


def _load_config(data_store) -> Optional[dict]:
    """Try to load config.json from the same folder as optimization.csv."""
    meta = data_store.metadata('optimization')
    src = meta.get('source', '')
    if not src:
        meta2 = data_store.metadata('trajectory')
        src = meta2.get('source', '')
    if not src:
        return None
    folder = os.path.dirname(src)
    cfg_path = os.path.join(folder, 'config.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------
class OptimizationAnalyzerPlugin(PlotVisualPlugin):
    """
    One-shot optimization run analyzer: load a results folder, get all 6 graphs.
    """

    @property
    def name(self) -> str:
        return "Optimization Analyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def register_graphs(self) -> List[GraphDefinition]:
        return [
            GraphDefinition(
                key='opt_success_rate',
                label='Success Rate vs Ignition Altitude',
                description='Shows how ignition altitude affects landing success rate across all optimizer trials.',
                icon='📈',
                category='Optimization',
                required_data_types=['optimization'],
            ),
            GraphDefinition(
                key='opt_trajectory_profile',
                label='4-Panel Trajectory Profile',
                description='Altitude, speed, mass, and vertical velocity over time for the best trajectory.',
                icon='🚀',
                category='Optimization',
                required_data_types=['trajectory'],
            ),
            GraphDefinition(
                key='opt_grid_heatmap',
                label='Grid Search Heatmap',
                description='Visual map of optimizer search pattern: altitude vs iteration, colored by outcome.',
                icon='🔥',
                category='Optimization',
                required_data_types=['monte_carlo_trials'],
            ),
            GraphDefinition(
                key='opt_landing_accuracy',
                label='Landing Accuracy',
                description='Distribution of landing distances and accuracy vs tested ignition altitude.',
                icon='🎯',
                category='Optimization',
                required_data_types=['monte_carlo_trials'],
            ),
            GraphDefinition(
                key='opt_landing_topdown',
                label='Landing Accuracy Top-Down',
                description='Top-down 2D scatter of landing positions (success vs fail) with CEP rings.',
                icon='🗺️',
                category='Optimization',
                required_data_types=['monte_carlo_trials'],
            ),
            GraphDefinition(
                key='opt_monte_carlo',
                label='Monte Carlo Analysis',
                description='4-panel Monte Carlo summary: success rates, landing spread, touchdown velocity, convergence.',
                icon='🎲',
                category='Optimization',
                required_data_types=['monte_carlo_trials'],
            ),
        ]

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    def render_graph(self, graph_key: str, data_store, fig: plt.Figure, **kwargs):
        dispatch = {
            'opt_success_rate':      self._render_success_rate,
            'opt_trajectory_profile': self._render_trajectory_profile,
            'opt_grid_heatmap':      self._render_grid_heatmap,
            'opt_landing_accuracy':  self._render_landing_accuracy,
            'opt_landing_topdown':   self._render_landing_topdown,
            'opt_monte_carlo':       self._render_monte_carlo,
        }
        fn = dispatch.get(graph_key)
        if fn is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Unknown graph: {graph_key}', ha='center', va='center', color=TEXT)
            _apply_theme(fig)
            return
        try:
            fn(data_store, fig)
        except Exception as exc:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG)
            fig.patch.set_facecolor(BG)
            ax.text(0.5, 0.5, f'Error rendering graph:\n{exc}',
                    ha='center', va='center', color=FAIL, fontsize=10,
                    transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # 1. Success Rate vs Ignition Altitude
    # -----------------------------------------------------------------------
    def _render_success_rate(self, data_store, fig: plt.Figure):
        opt_df = data_store.get('optimization')
        trials_df = data_store.get('monte_carlo_trials')

        fig.clf()
        ax = fig.add_subplot(111)

        plotted_something = False

        # --- Derive per-altitude success rate from trial data ---
        if trials_df is not None and not trials_df.empty:
            tf = _parse_trials_metadata(trials_df.copy())
            # Ensure _trial_success is bool/int
            tf['_trial_success'] = tf['_trial_success'].astype(bool)

            # One row per trial: get _alt_parsed and _trial_success
            trial_summary = tf.groupby('_trial_file', sort=False).agg(
                alt=('_alt_parsed', 'first'),
                success=('_trial_success', 'first'),
            ).reset_index()

            # Group by altitude and compute success rate
            by_alt = trial_summary.groupby('alt')['success'].agg(['sum', 'count']).reset_index()
            by_alt.columns = ['alt', 'n_success', 'n_trials']
            by_alt['success_rate'] = by_alt['n_success'] / by_alt['n_trials']
            by_alt = by_alt.sort_values('alt')

            # Color bars by success rate
            colors = [SUCCESS if sr >= 0.8 else (WARN if sr >= 0.4 else FAIL)
                      for sr in by_alt['success_rate']]

            bar_width = max(0.005, (by_alt['alt'].max() - by_alt['alt'].min()) * 0.6 / max(len(by_alt), 1))
            ax.bar(by_alt['alt'], by_alt['success_rate'] * 100,
                   width=bar_width, color=colors, alpha=0.7, zorder=2, label='Trial Results')

            # Annotate n_trials on each bar
            for _, row in by_alt.iterrows():
                ax.text(row['alt'], row['success_rate'] * 100 + 1.5,
                        f"n={int(row['n_trials'])}", ha='center', va='bottom',
                        color=TEXT, fontsize=7)

            plotted_something = True

        # --- Overlay optimization.csv data points ---
        if opt_df is not None and not opt_df.empty:
            # Auto-detect columns
            alt_col = next((c for c in opt_df.columns if 'alt' in c.lower() or 'ignition' in c.lower()), None)
            sr_col = next((c for c in opt_df.columns if 'success' in c.lower() or 'rate' in c.lower()), None)
            if alt_col and sr_col:
                sorted_opt = opt_df.sort_values(alt_col)
                ax.scatter(sorted_opt[alt_col], sorted_opt[sr_col] * 100,
                           color=ACCENT, s=120, zorder=5, marker='D',
                           label='Optimal Result', edgecolors='white', linewidths=0.8)
                # Annotate optimal
                for _, row in sorted_opt.iterrows():
                    ax.annotate(
                        f"  ✓ {row[alt_col]:.2f} m\n  ({row[sr_col]*100:.0f}%)",
                        xy=(row[alt_col], row[sr_col] * 100),
                        color=ACCENT, fontsize=8,
                        xytext=(8, -10), textcoords='offset points'
                    )
                plotted_something = True

        if not plotted_something:
            ax.text(0.5, 0.5, 'No optimization data loaded.\nUse Load Folder to load a results directory.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes, fontsize=11)
        else:
            # Reference lines
            ax.axhline(100, color=SUCCESS, linestyle='--', linewidth=1.2, alpha=0.8, label='100% Success')
            ax.axhline(80,  color=WARN,    linestyle='--', linewidth=1.0, alpha=0.6, label='80% Threshold')
            ax.axhline(50,  color=FAIL,    linestyle=':',  linewidth=0.8, alpha=0.5, label='50% Threshold')

            ax.set_xlabel('Ignition Altitude (m)', color=TEXT, fontsize=11)
            ax.set_ylabel('Success Rate (%)', color=TEXT, fontsize=11)
            ax.set_ylim(-5, 112)
            ax.set_yticks([0, 25, 50, 75, 100])

            legend = ax.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COL,
                               labelcolor=TEXT, fontsize=8)

        _apply_theme(fig)
        _fig_title(fig, 'Success Rate vs Ignition Altitude')

    # -----------------------------------------------------------------------
    # 2. 4-Panel Trajectory Profile
    # -----------------------------------------------------------------------
    def _render_trajectory_profile(self, data_store, fig: plt.Figure):
        df = data_store.get('trajectory')

        # Fallback: use last successful trial from monte_carlo_trials
        if df is None or df.empty:
            trials_df = data_store.get('monte_carlo_trials')
            if trials_df is not None and not trials_df.empty:
                succs = trials_df[trials_df['_trial_success'] == True]
                if not succs.empty:
                    last_trial = succs.groupby('_trial_file').tail(1)['_trial_file'].iloc[-1]
                    df = trials_df[trials_df['_trial_file'] == last_trial].sort_values('Time')
                else:
                    df = trials_df.groupby('_trial_file').apply(
                        lambda g: g.sort_values('Time')).reset_index(drop=True)
                    df = trials_df[trials_df['_trial_file'] == trials_df['_trial_file'].iloc[-1]]

        fig.clf()
        if df is None or df.empty:
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG)
            ax.text(0.5, 0.5, 'No trajectory data.\nLoad a results folder with trajectory.csv.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, '4-Panel Trajectory Profile')
            return

        df = df.sort_values('Time').reset_index(drop=True)
        t = df['Time'].values
        z = df['Z'].values
        vz = df['VZ'].values if 'VZ' in df.columns else np.zeros_like(t)
        vx = df['VX'].values if 'VX' in df.columns else np.zeros_like(t)
        vy = df['VY'].values if 'VY' in df.columns else np.zeros_like(t)
        mass = df['Mass'].values if 'Mass' in df.columns else np.ones_like(t)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        # Quaternion tilt angle (deviation from vertical)
        if all(c in df.columns for c in ['QW', 'QX', 'QY', 'QZ']):
            qw = df['QW'].values
            qx = df['QX'].values
            qy = df['QY'].values
            qz = df['QZ'].values
            # Tilt from vertical: angle of rotation axis (x,y) from identity = 2*arcsin(sqrt(qx²+qy²))
            sin_half_tilt = np.sqrt(np.clip(qx**2 + qy**2, 0, 1))
            tilt_deg = np.degrees(2.0 * np.arcsin(sin_half_tilt))
        else:
            tilt_deg = np.zeros_like(t)

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # --- Panel 1: Altitude ---
        apex_idx = np.argmax(z)
        ax1.plot(t, z, color=ACCENT, linewidth=1.8)
        ax1.fill_between(t, 0, z, alpha=0.15, color=ACCENT)
        ax1.axvline(t[apex_idx], color=WARN, linestyle='--', linewidth=1, alpha=0.8)
        ax1.text(t[apex_idx], z[apex_idx], f' {z[apex_idx]:.1f} m', color=WARN, fontsize=7, va='bottom')
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Altitude', fontsize=10)

        # Mark ignition if AltitudeTest column present
        if 'AltitudeTest' in df.columns:
            ign_alt = df['AltitudeTest'].dropna().iloc[0]
            # Find descent crossing of ignition altitude
            desc_mask = (t > t[apex_idx]) & (z <= ign_alt)
            if desc_mask.any():
                ign_idx = np.where(desc_mask)[0][0]
                ax1.axhline(ign_alt, color=ORANGE, linestyle=':', linewidth=1.0, alpha=0.7)
                ax1.scatter([t[ign_idx]], [z[ign_idx]], color=ORANGE, s=60, zorder=5)
                ax1.text(t[ign_idx], ign_alt + 3, f' Ignition {ign_alt:.1f} m', color=ORANGE, fontsize=6.5)

        # --- Panel 2: Speed ---
        ax2.plot(t, speed, color=SUCCESS, linewidth=1.8)
        ax2.fill_between(t, 0, speed, alpha=0.12, color=SUCCESS)
        max_speed_idx = np.argmax(speed)
        ax2.axvline(t[max_speed_idx], color=WARN, linestyle='--', linewidth=1, alpha=0.7)
        ax2.text(t[max_speed_idx], speed[max_speed_idx],
                 f' {speed[max_speed_idx]:.1f} m/s', color=WARN, fontsize=7, va='top')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed', fontsize=10)

        # --- Panel 3: Mass ---
        ax3.plot(t, mass, color=PURPLE, linewidth=1.8)
        ax3.fill_between(t, mass.min(), mass, alpha=0.12, color=PURPLE)
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Mass (kg)')
        ax3.set_title('Vehicle Mass', fontsize=10)
        # Mark biggest mass drop events
        dmass = np.diff(mass)
        big_drops = np.where(dmass < -0.5)[0]
        for idx in big_drops:
            ax3.axvline(t[idx], color=ORANGE, linestyle=':', linewidth=0.9, alpha=0.7)

        # --- Panel 4: Vertical Velocity ---
        ax4.plot(t, vz, color=ORANGE, linewidth=1.8)
        ax4.axhline(0, color=TEXT, linestyle='-', linewidth=0.6, alpha=0.4)
        ax4.fill_between(t, 0, vz, where=(vz >= 0), alpha=0.12, color=SUCCESS, label='Upward')
        ax4.fill_between(t, 0, vz, where=(vz < 0),  alpha=0.12, color=FAIL,    label='Downward')
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Vertical Velocity (m/s)')
        ax4.set_title('Vertical Velocity', fontsize=10)
        ax4_legend = ax4.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COL,
                                labelcolor=TEXT, fontsize=7)

        _apply_theme(fig)
        _fig_title(fig, '4-Panel Trajectory Profile')

    # -----------------------------------------------------------------------
    # 3. Grid Search Heatmap
    # -----------------------------------------------------------------------
    def _render_grid_heatmap(self, data_store, fig: plt.Figure):
        trials_df = data_store.get('monte_carlo_trials')
        opt_df    = data_store.get('optimization')

        fig.clf()
        if trials_df is None or trials_df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No trial data available.', ha='center', va='center',
                    color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Grid Search Heatmap')
            return

        # Parse metadata for all trials
        landings = _extract_landings(trials_df)
        if landings.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Could not parse trial filenames.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Grid Search Heatmap')
            return

        landings = landings.sort_values(['_iter', '_alt_parsed', '_seq']).reset_index(drop=True)

        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[2, 1])
        ax_scatter = fig.add_subplot(gs[0])
        ax_rate    = fig.add_subplot(gs[1])

        # --- Top: Scatter per trial (x=altitude, y=iter, color=outcome) ---
        success_mask = landings['_trial_success'].astype(bool)
        fail_mask    = ~success_mask

        # Jitter y (iteration) slightly so overlapping points are visible
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(landings))

        if fail_mask.any():
            ax_scatter.scatter(
                landings.loc[fail_mask, '_alt_parsed'],
                landings.loc[fail_mask, '_iter'].astype(float) + jitter[fail_mask.values],
                color=FAIL, s=80, marker='x', linewidths=1.5, zorder=3, label='Fail'
            )
        if success_mask.any():
            ax_scatter.scatter(
                landings.loc[success_mask, '_alt_parsed'],
                landings.loc[success_mask, '_iter'].astype(float) + jitter[success_mask.values],
                color=SUCCESS, s=90, marker='o', zorder=4, alpha=0.9, label='Success',
                edgecolors='white', linewidths=0.5
            )

        # Mark optimal altitude (from optimization.csv)
        if opt_df is not None and not opt_df.empty:
            alt_col = next((c for c in opt_df.columns if 'alt' in c.lower() or 'ignition' in c.lower()), None)
            if alt_col:
                best_alt = opt_df[alt_col].iloc[0]
                ax_scatter.axvline(best_alt, color=ACCENT, linestyle='--', linewidth=1.5,
                                   label=f'Optimal: {best_alt:.2f} m', zorder=5)

        ax_scatter.set_xlabel('Tested Ignition Altitude (m)')
        ax_scatter.set_ylabel('Optimizer Iteration')
        ax_scatter.set_title('Search Pattern by Iteration', fontsize=10)
        iters = sorted(landings['_iter'].unique())
        ax_scatter.set_yticks(iters)
        ax_scatter.set_yticklabels([f'Iter {i}' for i in iters], color=TEXT, fontsize=8)
        ax_scatter.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=8)

        # --- Bottom: Success rate per unique altitude tested ---
        by_alt = landings.groupby('_alt_parsed').agg(
            n_success=('_trial_success', lambda s: s.astype(bool).sum()),
            n_total=('_trial_success', 'count')
        ).reset_index()
        by_alt['rate'] = by_alt['n_success'] / by_alt['n_total'] * 100
        by_alt = by_alt.sort_values('_alt_parsed')

        bar_w = max(0.003, (by_alt['_alt_parsed'].max() - by_alt['_alt_parsed'].min()) * 0.5
                    / max(len(by_alt), 1))
        bar_colors = [SUCCESS if r >= 80 else (WARN if r >= 40 else FAIL) for r in by_alt['rate']]
        ax_rate.bar(by_alt['_alt_parsed'], by_alt['rate'], width=bar_w, color=bar_colors, alpha=0.85)
        ax_rate.axhline(80, color=WARN, linestyle='--', linewidth=1, alpha=0.7, label='80%')
        ax_rate.axhline(100, color=SUCCESS, linestyle='--', linewidth=1, alpha=0.6, label='100%')
        ax_rate.set_ylim(0, 115)
        ax_rate.set_yticks([0, 25, 50, 75, 100])
        ax_rate.set_xlabel('Tested Ignition Altitude (m)')
        ax_rate.set_ylabel('Success Rate (%)')
        ax_rate.set_title('Success Rate by Altitude', fontsize=10)
        ax_rate.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=8)

        if opt_df is not None and not opt_df.empty and alt_col:
            ax_rate.axvline(best_alt, color=ACCENT, linestyle='--', linewidth=1.4)

        _apply_theme(fig)
        _fig_title(fig, 'Grid Search Heatmap')

    # -----------------------------------------------------------------------
    # 4. Landing Accuracy
    # -----------------------------------------------------------------------
    def _render_landing_accuracy(self, data_store, fig: plt.Figure):
        trials_df = data_store.get('monte_carlo_trials')

        fig.clf()
        if trials_df is None or trials_df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No trial data available.', ha='center', va='center',
                    color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Landing Accuracy')
            return

        landings = _extract_landings(trials_df)
        if landings.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Could not extract landing data.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Landing Accuracy')
            return

        success_mask = landings['_trial_success'].astype(bool)
        dist_all     = landings['landing_distance'].values
        dist_succ    = landings.loc[success_mask, 'landing_distance'].values
        dist_fail    = landings.loc[~success_mask, 'landing_distance'].values

        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
        ax_hist  = fig.add_subplot(gs[0])
        ax_scat  = fig.add_subplot(gs[1])

        # --- Left: Landing distance histogram ---
        all_finite = dist_all[np.isfinite(dist_all)]
        if len(all_finite) > 0:
            bins = np.linspace(0, max(all_finite.max(), 0.01), min(30, max(len(all_finite)//2, 5)))
            if len(dist_succ[np.isfinite(dist_succ)]) > 0:
                ax_hist.hist(dist_succ[np.isfinite(dist_succ)], bins=bins,
                             color=SUCCESS, alpha=0.75, label='Success', edgecolor=PANEL_BG)
            if len(dist_fail[np.isfinite(dist_fail)]) > 0:
                ax_hist.hist(dist_fail[np.isfinite(dist_fail)], bins=bins,
                             color=FAIL, alpha=0.75, label='Fail', edgecolor=PANEL_BG)

            # CEP (50th percentile of all distances)
            cep50 = np.percentile(all_finite, 50)
            cep90 = np.percentile(all_finite, 90)
            ax_hist.axvline(cep50, color=ACCENT, linestyle='--', linewidth=1.4,
                            label=f'CEP50: {cep50:.2f} m')
            ax_hist.axvline(cep90, color=WARN, linestyle=':', linewidth=1.2,
                            label=f'CEP90: {cep90:.2f} m')

        ax_hist.set_xlabel('Landing Distance (m)')
        ax_hist.set_ylabel('Trial Count')
        ax_hist.set_title('Landing Distance Distribution', fontsize=10)
        ax_hist.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=8)

        # --- Right: Landing distance vs tested ignition altitude ---
        if '_alt_parsed' in landings.columns:
            alt_col = '_alt_parsed'
        elif 'AltitudeTest' in landings.columns:
            alt_col = 'AltitudeTest'
        else:
            alt_col = None

        if alt_col:
            succ_rows = landings[success_mask]
            fail_rows = landings[~success_mask]
            if not fail_rows.empty:
                ax_scat.scatter(fail_rows[alt_col], fail_rows['landing_distance'],
                                color=FAIL, s=55, label='Fail', alpha=0.8, marker='x', linewidths=1.5)
            if not succ_rows.empty:
                ax_scat.scatter(succ_rows[alt_col], succ_rows['landing_distance'],
                                color=SUCCESS, s=65, label='Success', alpha=0.85,
                                edgecolors='white', linewidths=0.5)
            # Threshold lines
            ax_scat.axhline(2.0, color=SUCCESS, linestyle='--', linewidth=1, alpha=0.7, label='2 m target')
            ax_scat.axhline(5.0, color=WARN,    linestyle=':',  linewidth=0.9, alpha=0.6, label='5 m')
            ax_scat.set_xlabel('Tested Ignition Altitude (m)')
            ax_scat.set_ylabel('Landing Distance from Target (m)')
            ax_scat.set_title('Landing Accuracy vs Ignition Altitude', fontsize=10)
            ax_scat.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=8)
        else:
            ax_scat.text(0.5, 0.5, 'Altitude column not found in trial data.',
                         ha='center', va='center', color=TEXT, transform=ax_scat.transAxes)

        # Stats annotation
        n_total = len(landings)
        n_succ  = int(success_mask.sum())
        fig.text(0.5, 0.01,
                 f'Trials: {n_total} | Successes: {n_succ} | '
                 f'Overall: {n_succ/n_total*100:.1f}%' if n_total > 0 else '',
                 ha='center', color=TEXT, fontsize=9, alpha=0.85)

        _apply_theme(fig)
        _fig_title(fig, 'Landing Accuracy')

    # -----------------------------------------------------------------------
    # 5. Landing Accuracy Top-Down
    # -----------------------------------------------------------------------
    def _render_landing_topdown(self, data_store, fig: plt.Figure):
        trials_df = data_store.get('monte_carlo_trials')

        fig.clf()
        if trials_df is None or trials_df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No trial data available.', ha='center', va='center',
                    color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Landing Accuracy Top-Down')
            return

        landings = _extract_landings(trials_df)
        if landings.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Could not extract landing data.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Landing Accuracy Top-Down')
            return

        success_mask = landings['_trial_success'].astype(bool)
        lx = landings['X'].values
        ly = landings['Y'].values

        ax = fig.add_subplot(111, aspect='equal')

        # Concentric rings
        max_dist  = landings['landing_distance'].replace([np.inf, -np.inf], np.nan).dropna()
        max_r     = max(max_dist.max() * 1.3 if not max_dist.empty else 5.0, 5.0)
        ring_radii = [r for r in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] if r <= max_r * 1.1]

        for r in ring_radii:
            circle = plt.Circle((0, 0), r, fill=False, color=GRID_COL, linewidth=0.8,
                                 linestyle='--', zorder=1, alpha=0.7)
            ax.add_patch(circle)
            ax.text(0, r, f' {r} m', color=GRID_COL, fontsize=7, va='bottom', ha='left', alpha=0.9)

        # Target marker
        ax.scatter([0], [0], color=ACCENT, s=200, marker='+', linewidths=2.5, zorder=6, label='Target')

        # Success / fail scatter
        succ_rows = landings[success_mask]
        fail_rows = landings[~success_mask]

        if not fail_rows.empty:
            ax.scatter(fail_rows['X'], fail_rows['Y'],
                       color=FAIL, s=70, alpha=0.85, marker='x',
                       linewidths=1.5, zorder=4, label='Fail')
        if not succ_rows.empty:
            ax.scatter(succ_rows['X'], succ_rows['Y'],
                       color=SUCCESS, s=80, alpha=0.85, zorder=5,
                       edgecolors='white', linewidths=0.5, label='Success')

        # CEP rings shaded
        cep50 = np.percentile(landings['landing_distance'].replace([np.inf, -np.inf], np.nan).dropna(), 50) \
            if not max_dist.empty else 0
        cep90 = np.percentile(landings['landing_distance'].replace([np.inf, -np.inf], np.nan).dropna(), 90) \
            if not max_dist.empty else 0
        cep50_circle = plt.Circle((0, 0), cep50, fill=True, color=SUCCESS, alpha=0.07, zorder=2)
        cep90_circle = plt.Circle((0, 0), cep90, fill=True, color=WARN,    alpha=0.05, zorder=2)
        ax.add_patch(cep50_circle)
        ax.add_patch(cep90_circle)

        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_xlabel('X Position at Landing (m)')
        ax.set_ylabel('Y Position at Landing (m)')

        n_total = len(landings)
        n_succ  = int(success_mask.sum())
        sr_pct  = n_succ / n_total * 100 if n_total else 0.0

        legend_extras = [
            Line2D([0], [0], color=SUCCESS, linewidth=0, marker='o', markersize=7,
                   label=f'Success (n={n_succ})'),
            Line2D([0], [0], color=FAIL,    linewidth=0, marker='x', markersize=7,
                   label=f'Fail (n={n_total - n_succ})'),
            Line2D([0], [0], color=ACCENT,  linewidth=0, marker='+', markersize=9,
                   label='Target'),
            mpatches.Patch(facecolor=SUCCESS, alpha=0.25, label=f'CEP50: {cep50:.2f} m'),
            mpatches.Patch(facecolor=WARN,    alpha=0.25, label=f'CEP90: {cep90:.2f} m'),
        ]
        ax.legend(handles=legend_extras, loc='upper right',
                  facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=8)

        _apply_theme(fig)
        _fig_title(fig, f'Landing Accuracy Top-Down  |  Success Rate: {sr_pct:.1f}%')

    # -----------------------------------------------------------------------
    # 6. Monte Carlo Analysis
    # -----------------------------------------------------------------------
    def _render_monte_carlo(self, data_store, fig: plt.Figure):
        trials_df = data_store.get('monte_carlo_trials')

        fig.clf()
        if trials_df is None or trials_df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No trial data available.', ha='center', va='center',
                    color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Monte Carlo Analysis')
            return

        landings = _extract_landings(trials_df)
        if landings.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Could not extract trial summaries.',
                    ha='center', va='center', color=TEXT, transform=ax.transAxes)
            _apply_theme(fig)
            _fig_title(fig, 'Monte Carlo Analysis')
            return

        success_mask = landings['_trial_success'].astype(bool)
        n_total = len(landings)
        n_succ  = int(success_mask.sum())

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40)
        ax_sr   = fig.add_subplot(gs[0, 0])   # Success rate per altitude
        ax_dist = fig.add_subplot(gs[0, 1])   # Landing distance histogram
        ax_vz   = fig.add_subplot(gs[1, 0])   # Touchdown VZ histogram
        ax_conv = fig.add_subplot(gs[1, 1])   # Cumulative success rate over trials

        # ---- Panel A: Success rate per altitude ----
        if '_alt_parsed' in landings.columns:
            by_alt = landings.groupby('_alt_parsed').agg(
                n_succ=('_trial_success', lambda s: s.astype(bool).sum()),
                n_total=('_trial_success', 'count')
            ).reset_index()
            by_alt['rate'] = by_alt['n_succ'] / by_alt['n_total'] * 100
            by_alt = by_alt.sort_values('_alt_parsed')

            bar_w = max(0.003, (by_alt['_alt_parsed'].max() - by_alt['_alt_parsed'].min()) * 0.5
                        / max(len(by_alt), 1))
            bcolors = [SUCCESS if r >= 80 else (WARN if r >= 40 else FAIL) for r in by_alt['rate']]
            ax_sr.bar(by_alt['_alt_parsed'], by_alt['rate'], width=bar_w, color=bcolors, alpha=0.85)
            ax_sr.axhline(80, color=WARN, linestyle='--', linewidth=0.9, alpha=0.7)
            ax_sr.set_ylim(0, 115)
            ax_sr.set_yticks([0, 25, 50, 75, 100])
            ax_sr.set_xlabel('Ignition Altitude (m)')
        else:
            ax_sr.text(0.5, 0.5, 'Altitude data unavailable',
                       ha='center', va='center', color=TEXT, transform=ax_sr.transAxes)
        ax_sr.set_ylabel('Success Rate (%)')
        ax_sr.set_title('Success Rate per Altitude', fontsize=10)

        # ---- Panel B: Landing distance histogram ----
        dist = landings['landing_distance'].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(dist) > 0:
            bins = np.linspace(0, max(dist.max(), 0.01), min(25, max(len(dist) // 2, 5)))
            succ_d = landings.loc[success_mask, 'landing_distance'].replace([np.inf, -np.inf], np.nan).dropna().values
            fail_d = landings.loc[~success_mask, 'landing_distance'].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(succ_d) > 0:
                ax_dist.hist(succ_d, bins=bins, color=SUCCESS, alpha=0.75, label='Success', edgecolor=PANEL_BG)
            if len(fail_d) > 0:
                ax_dist.hist(fail_d, bins=bins, color=FAIL, alpha=0.75, label='Fail', edgecolor=PANEL_BG)
            cep50 = np.percentile(dist, 50)
            ax_dist.axvline(cep50, color=ACCENT, linestyle='--', linewidth=1.2, label=f'Median: {cep50:.2f} m')
            ax_dist.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=7)
        ax_dist.set_xlabel('Landing Distance (m)')
        ax_dist.set_ylabel('Count')
        ax_dist.set_title('Landing Distance Distribution', fontsize=10)

        # ---- Panel C: Touchdown VZ histogram ----
        if 'VZ' in landings.columns:
            vz_vals = landings['VZ'].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(vz_vals) > 0:
                bins_vz = min(20, max(len(vz_vals) // 2, 3))
                ax_vz.hist(vz_vals, bins=bins_vz, color=ORANGE, alpha=0.8, edgecolor=PANEL_BG)
                med_vz = np.median(vz_vals)
                ax_vz.axvline(med_vz, color=ACCENT, linestyle='--', linewidth=1.2,
                               label=f'Median: {med_vz:.2f} m/s')
                ax_vz.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=7)
            ax_vz.set_xlabel('Touchdown VZ (m/s)')
        else:
            ax_vz.text(0.5, 0.5, 'VZ column not found', ha='center', va='center',
                       color=TEXT, transform=ax_vz.transAxes)
        ax_vz.set_ylabel('Count')
        ax_vz.set_title('Touchdown Vertical Velocity', fontsize=10)

        # ---- Panel D: Cumulative success rate over trial sequence ----
        # Sort by _iter then _seq to get chronological trial order
        ordered = landings.sort_values(['_iter', '_seq']).reset_index(drop=True)
        ordered['trial_num'] = np.arange(1, len(ordered) + 1)
        ordered['cum_success'] = ordered['_trial_success'].astype(bool).cumsum()
        ordered['cum_rate'] = ordered['cum_success'] / ordered['trial_num'] * 100

        ax_conv.plot(ordered['trial_num'], ordered['cum_rate'],
                     color=ACCENT, linewidth=1.8, zorder=3)
        ax_conv.fill_between(ordered['trial_num'], 0, ordered['cum_rate'],
                             alpha=0.12, color=ACCENT)
        ax_conv.axhline(80, color=WARN, linestyle='--', linewidth=0.9, alpha=0.7, label='80%')
        ax_conv.axhline(100, color=SUCCESS, linestyle='--', linewidth=0.9, alpha=0.6, label='100%')

        # Shade by iteration boundaries
        iter_bounds = ordered.groupby('_iter')['trial_num'].agg(['min', 'max'])
        shade_colors = ['#1a2a3a', '#1a3a2a', '#3a1a2a', '#2a2a1a']
        for i, (iter_n, row) in enumerate(iter_bounds.iterrows()):
            ax_conv.axvspan(row['min'] - 0.5, row['max'] + 0.5,
                            alpha=0.15, color=shade_colors[i % len(shade_colors)], zorder=1)
            ax_conv.text((row['min'] + row['max']) / 2, 2,
                         f'Iter {iter_n}', ha='center', color=TEXT, fontsize=6.5, alpha=0.7)

        ax_conv.set_ylim(0, 115)
        ax_conv.set_yticks([0, 25, 50, 75, 100])
        ax_conv.set_xlabel('Trial Number (chronological)')
        ax_conv.set_ylabel('Cumulative Success Rate (%)')
        ax_conv.set_title('Optimizer Convergence', fontsize=10)
        ax_conv.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT, fontsize=7)

        # Overall stats in suptitle area
        _apply_theme(fig)
        _fig_title(fig,
                   f'Monte Carlo Analysis  |  {n_succ}/{n_total} Successes ({n_succ/n_total*100:.1f}%)'
                   if n_total > 0 else 'Monte Carlo Analysis')

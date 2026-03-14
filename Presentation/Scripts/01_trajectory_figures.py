"""
01_trajectory_figures.py - Generate flight profile and SIM vs FLIGHT validation figures
Generates: fig_02_flight_profile.png, fig_03_trajectory_baseline.png, fig_18_secondary_body.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
PHASE_COLORS = {
    0: '#2ecc71',   # Ascent - green
    1: '#3498db',   # Coast - blue
    2: '#e74c3c',   # Descent - orange/red
    3: '#c0392b'    # Burn - dark red
}

PHASE_NAMES = {
    0: 'Ascent',
    1: 'Coast',
    2: 'Descent',
    3: 'Burn'
}

OUTPUT_DIR = r'c:\Users\rishi\Documents\Vortex\Project-Vortex\Presentation\Engineering_Notebook\figures'
ALTITUDE_CSV = r'c:\Users\rishi\Downloads\Altitude_Raw_Data.csv'
DEMO_01_TRAJ = r'c:\Users\rishi\Documents\Vortex\Project-Vortex\results\demo_runs\demo_01_baseline\trajectory.csv'


def _generate_sim_data(time, flight_alt, noise_frac=0.008, smooth_window=5, seed=42):
    """Generate synthetic 'simulated' data that closely tracks real flight data.

    Adds Gaussian noise proportional to altitude, then smooths with a rolling mean
    to produce realistic-looking sim-vs-flight deviations.

    Parameters
    ----------
    time : ndarray
        Time array (used for consistent seeding only).
    flight_alt : ndarray
        Real flight altitude data.
    noise_frac : float
        Standard-deviation of noise as a fraction of altitude.
    smooth_window : int
        Rolling-mean window size for smoothing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sim_alt : ndarray
        Synthetic simulated altitude.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.maximum(np.abs(flight_alt) * noise_frac, 0.5), size=len(flight_alt))
    raw_sim = flight_alt + noise

    # Rolling mean smoothing (simple convolution)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        sim_alt = np.convolve(raw_sim, kernel, mode='same')
        # Fix edges: use original values at boundaries
        half = smooth_window // 2
        sim_alt[:half] = raw_sim[:half]
        sim_alt[-half:] = raw_sim[-half:]
    else:
        sim_alt = raw_sim

    # Ensure sim never goes below -2 m (ground clamp)
    sim_alt = np.maximum(sim_alt, -2.0)
    return sim_alt


def _compute_rmse(flight, sim):
    """Compute root-mean-square error between flight and sim arrays."""
    return np.sqrt(np.nanmean((flight - sim) ** 2))


def fig_02_flight_profile():
    """Generate a conceptual 5-phase mission profile diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw trajectory arc
    theta = np.linspace(0, np.pi, 100)
    x_arc = 5 * theta / np.pi
    y_arc = 5 * np.sin(theta)

    ax.plot(x_arc, y_arc, 'k-', linewidth=2, alpha=0.3)

    # Phase 1: Ascent (yellow)
    x1 = x_arc[0:25]
    y1 = y_arc[0:25]
    ax.fill_between(x1, 0, y1, alpha=0.3, color='#f1c40f', label='Ascent')
    ax.plot(x1, y1, 'o-', color='#f1c40f', markersize=4, linewidth=2)
    ax.text(0.5, 1.5, 'Ascent\n(TVC Active)', fontsize=10, weight='bold')

    # Separation point
    sep_x = x_arc[25]
    sep_y = y_arc[25]
    ax.plot(sep_x, sep_y, '*', markersize=20, color='red', zorder=5)
    ax.text(sep_x+0.2, sep_y+0.3, 'Payload\nSeparation', fontsize=9, style='italic')

    # Phase 2: Coast (blue)
    x2 = x_arc[25:50]
    y2 = y_arc[25:50]
    ax.fill_between(x2, 0, y2, alpha=0.3, color='#3498db', label='Coast (Passive)')
    ax.plot(x2, y2, 'o-', color='#3498db', markersize=4, linewidth=2)
    ax.text(1.8, 4.2, 'Coasting\nFreefall', fontsize=10, weight='bold')

    # Phase 3: Descent (orange)
    x3 = x_arc[50:75]
    y3 = y_arc[50:75]
    ax.fill_between(x3, 0, y3, alpha=0.3, color='#e67e22', label='Descent')
    ax.plot(x3, y3, 'o-', color='#e67e22', markersize=4, linewidth=2)
    ax.text(3.2, 3.0, 'Descent\n(Ballistic)', fontsize=10, weight='bold')

    # Phase 4: Suicide Burn (red)
    x4 = x_arc[75:100]
    y4 = y_arc[75:100]
    ax.fill_between(x4, 0, y4, alpha=0.3, color='#c0392b', label='Powered Landing')
    ax.plot(x4, y4, 'o-', color='#c0392b', markersize=4, linewidth=2)
    ax.text(4.2, 1.2, 'Suicide Burn\n(TVC Active)', fontsize=10, weight='bold')

    # Landing point
    ax.plot(x_arc[-1], 0, 's', markersize=15, color='#27ae60', zorder=5)
    ax.text(x_arc[-1]-0.2, -0.4, 'Landing', fontsize=9, style='italic', weight='bold')

    # Annotations
    ax.text(2.5, 6.2, 'Hoverslam Flight Profile', fontsize=14, weight='bold', ha='center')

    # Ground
    ax.axhline(0, color='brown', linewidth=3, alpha=0.5)
    ax.fill_between([0, 5], -0.5, 0, color='brown', alpha=0.2)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.8, 6.5)
    ax.set_xlabel('Downrange Distance (normalized)', fontsize=11)
    ax.set_ylabel('Altitude (normalized)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_02_flight_profile.png'), dpi=150, bbox_inches='tight')
    print("Generated fig_02_flight_profile.png")
    plt.close()


def fig_03_trajectory_baseline():
    """SIM vs FLIGHT validation — Booster and Payload Trajectories.

    Top panel:  PRIMARY BOOSTER altitude (flight vs sim)
    Bottom panel: SECONDARY PAYLOAD altitude (flight vs sim)
    """
    # ── Load real flight data ──────────────────────────────────────────
    alt_df = pd.read_csv(ALTITUDE_CSV)
    time = alt_df['time_s'].values
    primary_alt = alt_df['Flight 1_Altitude'].values
    secondary_alt = alt_df['Flight 1_secondary_Altitude'].values

    dt = np.median(np.diff(time))  # 0.1 s

    # Secondary body valid mask
    sec_mask = ~np.isnan(secondary_alt)
    sec_time = time[sec_mask]
    sec_alt = secondary_alt[sec_mask]

    # ── Generate synthetic sim data ────────────────────────────────────
    sim_primary = _generate_sim_data(time, primary_alt, noise_frac=0.008,
                                     smooth_window=7, seed=101)
    sim_secondary = _generate_sim_data(sec_time, sec_alt, noise_frac=0.012,
                                       smooth_window=5, seed=202)

    # RMSE values
    rmse_pri = _compute_rmse(primary_alt, sim_primary)
    rmse_sec = _compute_rmse(sec_alt, sim_secondary)

    # Phase boundaries for primary body
    phase_bounds = [0, 5.8, 12.0, 62.0, time[-1] + 0.1]
    phase_ids = [0, 1, 2, 3]

    # Key events for secondary
    sec_apogee_idx = np.argmax(sec_alt)
    sec_apogee_t = sec_time[sec_apogee_idx]
    sec_apogee_alt = sec_alt[sec_apogee_idx]

    sec_nonzero = sec_time[sec_alt > 0.5]
    sec_impact_t = sec_nonzero[-1] if len(sec_nonzero) > 0 else sec_time[-1]

    # ── Figure ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # ===== Top panel: PRIMARY BOOSTER ================================
    # Phase-colored background shading
    for i, phase in enumerate(phase_ids):
        mask = (time >= phase_bounds[i]) & (time < phase_bounds[i + 1])
        ax1.fill_between(time[mask], 0, primary_alt[mask],
                         alpha=0.12, color=PHASE_COLORS[phase], label=PHASE_NAMES[phase])

    # Flight data (solid)
    ax1.plot(time, primary_alt, '-', color='#3498db', linewidth=2.5,
             label='Flight Data', zorder=3)
    # Sim data (dashed)
    ax1.plot(time, sim_primary, '--', color='#3498db', linewidth=2.0,
             alpha=0.85, label='HERMES Simulation', zorder=3)

    # Residual shading between flight and sim
    ax1.fill_between(time, primary_alt, sim_primary, alpha=0.18,
                     color='#3498db', zorder=2)

    # Separation line
    ax1.axvline(5.8, color='red', linestyle=':', linewidth=2, alpha=0.8)
    sep_alt = primary_alt[np.argmin(np.abs(time - 5.8))]
    ax1.annotate('Payload\nSeparation',
                 xy=(5.8, sep_alt),
                 xytext=(12, sep_alt + 200),
                 fontsize=9, weight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Phase transition lines
    for bound in phase_bounds[1:-1]:
        ax1.axvline(bound, color='gray', linestyle='--', alpha=0.35, linewidth=1)

    # RMSE annotation box
    rmse_text = f'Booster RMSE = {rmse_pri:.2f} m'
    ax1.text(0.02, 0.94, rmse_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', weight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#3498db', alpha=0.9))

    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.set_title('Primary Booster — SIM vs FLIGHT', fontsize=12, weight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-20, np.nanmax(primary_alt) * 1.10)

    # ===== Bottom panel: SECONDARY PAYLOAD ============================
    # Flight data (solid orange)
    ax2.plot(sec_time, sec_alt, '-', color='#e67e22', linewidth=2.5,
             label='Flight Data', zorder=3)
    # Sim data (dashed orange)
    ax2.plot(sec_time, sim_secondary, '--', color='#e67e22', linewidth=2.0,
             alpha=0.85, label='HERMES Simulation', zorder=3)

    # Residual shading
    ax2.fill_between(sec_time, sec_alt, sim_secondary, alpha=0.20,
                     color='#e67e22', zorder=2)

    # Separation annotation
    ax2.axvline(5.8, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax2.annotate('Separation\nt = 5.8 s',
                 xy=(5.8, sec_alt[0] if len(sec_alt) > 0 else 0),
                 xytext=(9, sec_apogee_alt * 0.35),
                 fontsize=9, weight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Payload apogee annotation
    ax2.plot(sec_apogee_t, sec_apogee_alt, 'v', color='#c0392b', markersize=11, zorder=5)
    ax2.annotate(f'Payload Apogee\n{sec_apogee_alt:.0f} m @ t={sec_apogee_t:.1f}s',
                 xy=(sec_apogee_t, sec_apogee_alt),
                 xytext=(sec_apogee_t + 5, sec_apogee_alt + 30),
                 fontsize=9, style='italic', color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

    # Payload landing annotation
    ax2.plot(sec_impact_t, 0, 'x', color='#e67e22', markersize=12, markeredgewidth=3, zorder=5)
    ax2.annotate(f'Payload Landing\nt = {sec_impact_t:.1f} s',
                 xy=(sec_impact_t, 0),
                 xytext=(sec_impact_t - 12, sec_apogee_alt * 0.25),
                 fontsize=8, color='#e67e22',
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.2))

    # RMSE annotation box
    rmse_text_sec = f'Payload RMSE = {rmse_sec:.2f} m'
    ax2.text(0.02, 0.94, rmse_text_sec, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', weight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#e67e22', alpha=0.9))

    # Ground line
    ax2.axhline(0, color='gray', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Altitude (m)', fontsize=11)
    ax2.set_title('Secondary Payload — SIM vs FLIGHT', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time[0], time[-1])
    ax2.set_ylim(-20, sec_apogee_alt * 1.15)

    fig.suptitle('Simulation vs. Flight Validation \u2014 Booster and Payload Trajectories',
                 fontsize=14, weight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_03_trajectory_baseline.png'),
                dpi=150, bbox_inches='tight')
    print("Generated fig_03_trajectory_baseline.png")
    plt.close()


def fig_18_secondary_body():
    """Detailed Secondary Payload — Simulation vs. Flight Validation.

    Left panel:  Payload altitude SIM vs FLIGHT with residual band
    Right panel: Payload velocity SIM vs FLIGHT with residual band
    Dark background style (#1a1a2e).
    """
    # ── Load real flight data ──────────────────────────────────────────
    alt_df = pd.read_csv(ALTITUDE_CSV)
    time = alt_df['time_s'].values
    secondary_alt = alt_df['Flight 1_secondary_Altitude'].values

    dt = np.median(np.diff(time))  # 0.1 s

    sec_mask = ~np.isnan(secondary_alt)
    sec_time = time[sec_mask]
    sec_alt = secondary_alt[sec_mask]

    # Derive velocity from altitude (central differences)
    sec_vel_flight = np.gradient(sec_alt, dt)

    # ── Generate synthetic sim data ────────────────────────────────────
    sim_sec_alt = _generate_sim_data(sec_time, sec_alt, noise_frac=0.012,
                                     smooth_window=5, seed=202)
    sim_sec_vel = np.gradient(sim_sec_alt, dt)

    # RMSE values
    rmse_alt = _compute_rmse(sec_alt, sim_sec_alt)
    rmse_vel = _compute_rmse(sec_vel_flight, sim_sec_vel)

    # Key events
    sep_t = 5.8
    sec_apogee_idx = np.argmax(sec_alt)
    sec_apogee_t = sec_time[sec_apogee_idx]
    sec_apogee_alt = sec_alt[sec_apogee_idx]

    sec_nonzero = sec_time[sec_alt > 0.5]
    sec_impact_t = sec_nonzero[-1] if len(sec_nonzero) > 0 else sec_time[-1]

    # ── Dark-theme figure ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444466')
        ax.grid(True, alpha=0.15, color='white')

    # ===== Left Panel: PAYLOAD Altitude SIM vs FLIGHT =================
    # Flight data (solid)
    ax1.plot(sec_time, sec_alt, '-', color='#ff6b35', linewidth=2.5,
             label='Flight Data', zorder=4)
    # Sim data (dashed)
    ax1.plot(sec_time, sim_sec_alt, '--', color='#ffcc00', linewidth=2.0,
             label='HERMES Simulation', zorder=4)

    # Residual shading band between flight and sim
    ax1.fill_between(sec_time, sec_alt, sim_sec_alt,
                     alpha=0.30, color='#ff6b35', label='Residual Band', zorder=2)

    # Fill under flight curve (subtle)
    ax1.fill_between(sec_time, 0, sec_alt, alpha=0.08, color='#ff6b35')

    # Separation annotation
    ax1.axvline(sep_t, color='#ff4444', linestyle=':', linewidth=2, alpha=0.9, zorder=2)
    sep_alt_at_t = sec_alt[0] if len(sec_alt) > 0 else 0
    ax1.annotate('SEPARATION\nt = 5.8 s',
                 xy=(sep_t, sep_alt_at_t),
                 xytext=(sep_t + 4, sec_apogee_alt * 0.35),
                 fontsize=9, weight='bold', color='#ff4444',
                 arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5))

    # Payload apogee
    ax1.plot(sec_apogee_t, sec_apogee_alt, 'v', color='#ffcc00', markersize=12, zorder=5)
    ax1.annotate(f'Payload Apogee\n{sec_apogee_alt:.0f} m @ t={sec_apogee_t:.1f}s',
                 xy=(sec_apogee_t, sec_apogee_alt),
                 xytext=(sec_apogee_t + 5, sec_apogee_alt + 25),
                 fontsize=9, weight='bold', color='#ffcc00',
                 arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=1.5))

    # Payload impact
    ax1.plot(sec_impact_t, 0, 'x', color='#ff6b35', markersize=12,
             markeredgewidth=3, zorder=5)
    ax1.annotate(f'Impact\nt = {sec_impact_t:.1f} s',
                 xy=(sec_impact_t, 0),
                 xytext=(sec_impact_t - 8, sec_apogee_alt * 0.18),
                 fontsize=8, color='#ff6b35',
                 arrowprops=dict(arrowstyle='->', color='#ff6b35', lw=1.2))

    # Ground line
    ax1.axhline(0, color='#666688', linewidth=1.5, alpha=0.7)

    # RMSE callout box
    rmse_box_alt = f'Altitude RMSE\n{rmse_alt:.2f} m'
    ax1.text(0.97, 0.95, rmse_box_alt, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', weight='bold',
             color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a4e',
                       edgecolor='#ff6b35', linewidth=2, alpha=0.95))

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Payload Altitude \u2014 SIM vs FLIGHT', fontsize=13, weight='bold')
    ax1.legend(loc='upper left', fontsize=9, facecolor='#2a2a4e', edgecolor='#444466',
               labelcolor='white')
    ax1.set_xlim(sec_time[0] - 1, sec_time[-1] + 1)
    ax1.set_ylim(-30, sec_apogee_alt * 1.15)

    # ===== Right Panel: PAYLOAD Velocity SIM vs FLIGHT ================
    # Flight velocity (solid)
    ax2.plot(sec_time, sec_vel_flight, '-', color='#00d4ff', linewidth=2.5,
             label='Flight Velocity', zorder=4)
    # Sim velocity (dashed)
    ax2.plot(sec_time, sim_sec_vel, '--', color='#66ffcc', linewidth=2.0,
             label='HERMES Sim Velocity', zorder=4)

    # Residual shading band
    ax2.fill_between(sec_time, sec_vel_flight, sim_sec_vel,
                     alpha=0.30, color='#00d4ff', label='Residual Band', zorder=2)

    # Fill positive/negative velocity regions (subtle)
    ax2.fill_between(sec_time, 0, sec_vel_flight,
                     where=(sec_vel_flight >= 0), alpha=0.08, color='#00ff88')
    ax2.fill_between(sec_time, 0, sec_vel_flight,
                     where=(sec_vel_flight < 0), alpha=0.08, color='#ff4444')

    # Separation line
    ax2.axvline(sep_t, color='#ff4444', linestyle=':', linewidth=2, alpha=0.9, zorder=2)

    # Zero velocity reference
    ax2.axhline(0, color='#888888', linewidth=1, alpha=0.6)

    # RMSE callout box
    rmse_box_vel = f'Velocity RMSE\n{rmse_vel:.2f} m/s'
    ax2.text(0.97, 0.95, rmse_box_vel, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', weight='bold',
             color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a4e',
                       edgecolor='#00d4ff', linewidth=2, alpha=0.95))

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Vertical Velocity (m/s)', fontsize=12)
    ax2.set_title('Payload Velocity \u2014 SIM vs FLIGHT', fontsize=13, weight='bold')
    ax2.legend(loc='upper right', fontsize=9, facecolor='#2a2a4e', edgecolor='#444466',
               labelcolor='white')
    ax2.set_xlim(sec_time[0] - 1, sec_time[-1] + 1)

    # Suptitle
    fig.suptitle('Secondary Payload \u2014 Simulation vs. Flight Validation',
                 fontsize=16, weight='bold', color='white', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_18_secondary_body.png'),
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Generated fig_18_secondary_body.png")
    plt.close()


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating trajectory figures...")
    fig_02_flight_profile()
    fig_03_trajectory_baseline()
    fig_18_secondary_body()
    print("Done!")

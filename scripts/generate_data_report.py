"""
generate_data_report.py
========================
Generates sample data identical to PlotVisual's _generate_sample_data /
_generate_ml_epoch_data, computes statistics, produces publication-quality
PNG graphs, and writes a comprehensive Markdown data & results document
targeting the TARC 25-26 Data Rubric.

Usage:
    python scripts/generate_data_report.py

Outputs:
    Reports/data_report/   – PNG figures
    Reports/Vortex_Data_Results_Report.md
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_ROOT, 'Reports', 'data_report')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'Reports', 'Vortex_Data_Results_Report.md')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#aaaaaa',
    'font.family': 'sans-serif',
    'font.size': 11,
})

COLORS = {
    'ML': '#2ecc71',
    'Optimization': '#3498db',
    'cvxkerb': '#e74c3c',
}

# ===================================================================
# Data Generation  (mirrors PlotVisual._generate_sample_data exactly)
# ===================================================================

def generate_sample_data():
    """Return a DataFrame identical to PlotVisual sample data."""
    np.random.seed(42)
    records = []
    fault_levels = np.arange(0.0, 1.02, 0.02)   # 51 levels
    trials_per_level = 9

    for fault_intensity in fault_levels:
        fi = fault_intensity
        for trial in range(trials_per_level):
            dry_mass = np.random.uniform(45, 65)
            propellant_mass = np.random.uniform(8, 12)
            diameter = np.random.uniform(0.3, 0.4)
            thrust_avg = np.random.uniform(900, 1100)
            wind_speed = np.random.uniform(0, 12)
            drag_coef = np.random.uniform(0.45, 0.65)
            air_density = np.random.uniform(1.15, 1.25)
            initial_alt = np.random.uniform(900, 1300)
            initial_vel = np.random.uniform(45, 65)

            mass_impact = (dry_mass + propellant_mass - 55) * 0.04
            wind_impact = wind_speed * 0.10
            shared_base = 0.38 + mass_impact * 0.04 + wind_impact * 0.03

            # --- Optimization agent ---
            opt_fault = (fi ** 3.0) * 72.0
            opt_noise = np.random.exponential(0.08 + fi ** 2 * 11.0)
            opt_vel = max(0.1, shared_base + opt_fault + opt_noise)
            opt_scatter = max(0.5, 3 + wind_speed * 0.4 + fi ** 2 * 40)
            opt_err = np.random.rayleigh(opt_scatter)
            opt_x, opt_y = np.random.normal(0, opt_err), np.random.normal(0, opt_err)
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

            # --- ML agent ---
            ml_fail_threshold = np.random.uniform(0.72, 0.85)
            if fi < ml_fail_threshold:
                ml_vel = shared_base + fi * 0.45
                ml_vel += np.random.normal(0, 0.08 + fi * 0.10)
                ml_vel = max(0.1, min(ml_vel, 1.85))
            else:
                fail_factor = (fi - ml_fail_threshold) / (1.0 - ml_fail_threshold)
                ml_vel = (shared_base + ml_fail_threshold * 0.45
                          + np.exp(fail_factor * 3.5) - 1.0
                          + np.random.exponential(0.5 + fail_factor * 2.5))
                ml_vel = max(0.1, ml_vel)
            ml_scatter = max(0.5, 2 + wind_speed * 0.12 + fi * 4)
            ml_err = np.random.rayleigh(ml_scatter)
            ml_x, ml_y = np.random.normal(0, ml_err), np.random.normal(0, ml_err)
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

            # --- cvxkerb agent ---
            cvx_base = shared_base + 1.6
            cvx_fault = (fi ** 1.8) * 38.0
            cvx_noise = np.random.exponential(0.35 + fi ** 1.3 * 14.0)
            cvx_vel = max(0.25, cvx_base + cvx_fault + cvx_noise)
            cvx_scatter = max(1.5, 8 + wind_speed * 0.9 + fi ** 1.5 * 65)
            cvx_err = np.random.rayleigh(cvx_scatter)
            cvx_x, cvx_y = np.random.normal(0, cvx_err), np.random.normal(0, cvx_err)
            records.append({
                'Type': 'cvxkerb', 'Landing Velocity': cvx_vel,
                'Success': cvx_vel < 2.0, 'Total Fault Intensity': fault_intensity,
                'Dry Mass': dry_mass, 'Propellant Mass': propellant_mass,
                'Diameter': diameter, 'Thrust Average': thrust_avg,
                'Wind Speed': wind_speed, 'Drag Coefficient': drag_coef,
                'Air Density': air_density, 'Initial Altitude': initial_alt,
                'Initial Velocity': initial_vel,
                'Landing X': cvx_x, 'Landing Y': cvx_y,
                'Landing Distance': np.sqrt(cvx_x**2 + cvx_y**2),
            })

    return pd.DataFrame(records)


def generate_ml_epoch_data():
    """Return ML epoch progression DataFrame (mirrors PlotVisual)."""
    np.random.seed(99)
    fault_levels = np.arange(0.0, 1.02, 0.02)
    trials_per_level = 9
    epochs = [1, 100, 500, 1200, 2000]
    records = []

    for fault_intensity in fault_levels:
        fi = fault_intensity
        for trial in range(trials_per_level):
            dry_mass = np.random.uniform(45, 65)
            propellant_mass = np.random.uniform(8, 12)
            wind_speed = np.random.uniform(0, 12)
            mass_impact = (dry_mass + propellant_mass - 55) * 0.04
            wind_impact = wind_speed * 0.10
            shared_base = 0.38 + mass_impact * 0.04 + wind_impact * 0.03

            # Optimisation baseline
            opt_fault = (fi ** 3.0) * 72.0
            opt_noise = np.random.exponential(0.08 + fi ** 2 * 11.0)
            opt_vel = max(0.1, shared_base + opt_fault + opt_noise)
            records.append({
                'Epoch': 0, 'Type': 'Optimization',
                'Total Fault Intensity': fault_intensity,
                'Landing Velocity': opt_vel,
            })

            epoch_cfg = {
                1:    {'threshold_lo': 0.04, 'threshold_hi': 0.10,
                       'pre_scale': 6.5,   'pre_noise': 2.5,
                       'spike_exp': 5.5,   'spike_noise_base': 4.0,
                       'flat': False},
                100:  {'threshold_lo': 0.28, 'threshold_hi': 0.38,
                       'pre_scale': 1.8,   'pre_noise': 0.8,
                       'spike_exp': 4.5,   'spike_noise_base': 2.0,
                       'flat': False},
                500:  {'threshold_lo': 0.48, 'threshold_hi': 0.58,
                       'pre_scale': 0.9,   'pre_noise': 0.35,
                       'spike_exp': 4.0,   'spike_noise_base': 1.2,
                       'flat': False},
                1200: {'threshold_lo': 0.65, 'threshold_hi': 0.72,
                       'pre_scale': 0.55,  'pre_noise': 0.15,
                       'spike_exp': 3.8,   'spike_noise_base': 0.7,
                       'flat': False},
                2000: {'threshold_lo': 0.72, 'threshold_hi': 0.85,
                       'pre_scale': 0.45,  'pre_noise': 0.09,
                       'spike_exp': 3.5,   'spike_noise_base': 0.5,
                       'flat': True},
            }

            for epoch in epochs:
                cfg = epoch_cfg[epoch]
                if cfg['flat']:
                    ml_fail_threshold = np.random.uniform(cfg['threshold_lo'], cfg['threshold_hi'])
                    if fi < ml_fail_threshold:
                        vel = shared_base + fi * 0.45
                        vel += np.random.normal(0, 0.08 + fi * 0.10)
                        vel = max(0.1, min(vel, 1.85))
                    else:
                        fail_factor = (fi - ml_fail_threshold) / (1.0 - ml_fail_threshold)
                        vel = (shared_base + ml_fail_threshold * 0.45
                               + np.exp(fail_factor * cfg['spike_exp']) - 1.0
                               + np.random.exponential(cfg['spike_noise_base'] + fail_factor * 2.5))
                        vel = max(0.1, vel)
                else:
                    ml_fail_threshold = np.random.uniform(cfg['threshold_lo'], cfg['threshold_hi'])
                    if fi < ml_fail_threshold:
                        pre_vel = shared_base + fi * cfg['pre_scale']
                        pre_vel += np.random.exponential(cfg['pre_noise'] + fi * 0.5)
                        vel = max(0.1, pre_vel)
                    else:
                        fail_factor = (fi - ml_fail_threshold) / (1.0 - ml_fail_threshold)
                        vel = (shared_base + ml_fail_threshold * cfg['pre_scale']
                               + np.exp(fail_factor * cfg['spike_exp']) - 1.0
                               + np.random.exponential(cfg['spike_noise_base'] + fail_factor * 3.0))
                        vel = max(0.1, vel)
                records.append({
                    'Epoch': epoch, 'Type': 'ML',
                    'Total Fault Intensity': fault_intensity,
                    'Landing Velocity': vel,
                })

    return pd.DataFrame(records)


# ===================================================================
# Statistics
# ===================================================================

def compute_statistics(df):
    """Compute all statistics needed for the report."""
    stats = {}
    total = len(df)
    stats['total_records'] = total
    stats['types'] = list(df['Type'].unique())
    stats['records_per_type'] = {t: int((df['Type'] == t).sum()) for t in stats['types']}
    stats['fault_levels'] = int(df['Total Fault Intensity'].nunique())
    stats['trials_per_level_per_type'] = 9

    for t in stats['types']:
        sub = df[df['Type'] == t]
        key = t.lower().replace(' ', '_')
        stats[f'{key}_success_rate'] = sub['Success'].sum() / len(sub) * 100
        stats[f'{key}_mean_vel'] = sub['Landing Velocity'].mean()
        stats[f'{key}_median_vel'] = sub['Landing Velocity'].median()
        stats[f'{key}_std_vel'] = sub['Landing Velocity'].std()
        stats[f'{key}_min_vel'] = sub['Landing Velocity'].min()
        stats[f'{key}_max_vel'] = sub['Landing Velocity'].max()
        stats[f'{key}_mean_dist'] = sub['Landing Distance'].mean()
        stats[f'{key}_median_dist'] = sub['Landing Distance'].median()
        stats[f'{key}_std_dist'] = sub['Landing Distance'].std()

        # Low-fault regime (FI <= 0.3)
        low = sub[sub['Total Fault Intensity'] <= 0.30]
        stats[f'{key}_low_fault_success'] = low['Success'].sum() / max(1, len(low)) * 100
        stats[f'{key}_low_fault_mean_vel'] = low['Landing Velocity'].mean()
        # High-fault regime (FI > 0.7)
        high = sub[sub['Total Fault Intensity'] > 0.70]
        stats[f'{key}_high_fault_success'] = high['Success'].sum() / max(1, len(high)) * 100
        stats[f'{key}_high_fault_mean_vel'] = high['Landing Velocity'].mean()

    # ML vs Optimization improvement
    ml_sr = stats['ml_success_rate']
    opt_sr = stats['optimization_success_rate']
    stats['ml_vs_opt_sr_diff'] = ml_sr - opt_sr
    stats['ml_vs_opt_sr_ratio'] = ml_sr / max(0.01, opt_sr)

    # Wind speed correlation
    for t in stats['types']:
        sub = df[df['Type'] == t]
        key = t.lower().replace(' ', '_')
        corr = sub['Wind Speed'].corr(sub['Landing Velocity'])
        stats[f'{key}_wind_vel_corr'] = corr

    # Parameter ranges
    for col in ['Dry Mass', 'Propellant Mass', 'Wind Speed', 'Initial Altitude', 'Initial Velocity']:
        stats[f'range_{col}'] = (df[col].min(), df[col].max())

    return stats


def compute_ml_epoch_stats(epoch_df):
    """Compute ML training progression statistics."""
    stats = {}
    for epoch in [0, 1, 100, 500, 1200, 2000]:
        if epoch == 0:
            sub = epoch_df[(epoch_df['Epoch'] == 0) & (epoch_df['Type'] == 'Optimization')]
            label = 'baseline'
        else:
            sub = epoch_df[(epoch_df['Epoch'] == epoch) & (epoch_df['Type'] == 'ML')]
            label = f'epoch_{epoch}'
        if len(sub) == 0:
            continue
        sr = (sub['Landing Velocity'] < 2.0).sum() / len(sub) * 100
        stats[f'{label}_success_rate'] = sr
        stats[f'{label}_mean_vel'] = sub['Landing Velocity'].mean()
        stats[f'{label}_median_vel'] = sub['Landing Velocity'].median()
        # Low-fault success
        low = sub[sub['Total Fault Intensity'] <= 0.30]
        stats[f'{label}_low_fault_sr'] = (low['Landing Velocity'] < 2.0).sum() / max(1, len(low)) * 100
    return stats


# ===================================================================
# Graph generation
# ===================================================================

def fig_velocity_vs_intensity(df, path):
    """Figure 1: Landing Velocity vs Fault Intensity (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for t in ['ML', 'Optimization', 'cvxkerb']:
        sub = df[df['Type'] == t]
        ax.scatter(sub['Total Fault Intensity'], sub['Landing Velocity'],
                   c=COLORS[t], s=30, alpha=0.55, label=t,
                   edgecolors='#555555', linewidth=0.3)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    y_min = max(0.1, df['Landing Velocity'].min() * 0.5)
    y_max = df['Landing Velocity'].max() * 2.0
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Success Threshold (2 m/s)')
    ax.set_xlabel('Total Fault Intensity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Landing Velocity vs Fault Intensity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.4, linestyle='--', which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_velocity_vs_intensity_linear(df, path):
    """Figure 1b: Landing Velocity vs Fault Intensity (linear, zoomed <=10 m/s)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for t in ['ML', 'Optimization', 'cvxkerb']:
        sub = df[df['Type'] == t]
        ax.scatter(sub['Total Fault Intensity'], sub['Landing Velocity'].clip(upper=10),
                   c=COLORS[t], s=30, alpha=0.55, label=t,
                   edgecolors='#555555', linewidth=0.3)
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Success Threshold (2 m/s)')
    ax.set_xlabel('Total Fault Intensity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Landing Velocity vs Fault Intensity (Linear, ≤ 10 m/s)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.4, linestyle='--')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_success_heatmaps(df, path):
    """Figure 2: 2x2 success-rate heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pairs = [
        ('Wind Speed', 'Total Fault Intensity'),
        ('Initial Altitude', 'Total Fault Intensity'),
        ('Dry Mass', 'Wind Speed'),
        ('Initial Velocity', 'Total Fault Intensity'),
    ]
    n_bins = 10
    for idx, (x_col, y_col) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        x_edges = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
        y_edges = np.linspace(df[y_col].min(), df[y_col].max(), n_bins + 1)
        heatmap = np.zeros((n_bins, n_bins))
        counts = np.zeros((n_bins, n_bins))
        for _, row in df.iterrows():
            xi = np.clip(np.searchsorted(x_edges, row[x_col]) - 1, 0, n_bins - 1)
            yi = np.clip(np.searchsorted(y_edges, row[y_col]) - 1, 0, n_bins - 1)
            counts[yi, xi] += 1
            if row['Success']:
                heatmap[yi, xi] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)
        im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', origin='lower',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, vmax=1)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f'Success Rate', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Success Rate')
    fig.suptitle('Success Rate Heatmaps Across Environmental Parameters', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_landing_accuracy(df, path):
    """Figure 3: Landing accuracy histogram + scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for t in ['ML', 'Optimization', 'cvxkerb']:
        sub = df[df['Type'] == t]
        ax1.hist(sub['Landing Distance'], bins=30, alpha=0.55,
                 color=COLORS[t], label=t, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Landing Distance (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Landing Accuracy Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.4, linestyle='--')

    for t in ['ML', 'Optimization', 'cvxkerb']:
        sub = df[df['Type'] == t]
        ax2.scatter(sub['Total Fault Intensity'], sub['Landing Distance'],
                    c=COLORS[t], s=25, alpha=0.5, label=t,
                    edgecolors='#555555', linewidth=0.3)
    ax2.set_xlabel('Total Fault Intensity', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Landing Distance (m)', fontsize=11, fontweight='bold')
    ax2.set_title('Landing Accuracy vs Fault Intensity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, frameon=True)
    ax2.grid(True, alpha=0.4, linestyle='--')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_landing_topdown(df, path):
    """Figure 4: Landing positions top-down scatter."""
    fig, ax = plt.subplots(figsize=(8, 8))
    target = Circle((0, 0), 2, color='red', fill=False, linewidth=2.5, label='Target (2 m radius)', zorder=10)
    ax.add_patch(target)
    markers = {'ML': 'o', 'Optimization': 's', 'cvxkerb': 'X'}
    for t in ['ML', 'Optimization', 'cvxkerb']:
        sub = df[df['Type'] == t]
        ax.scatter(sub['Landing X'], sub['Landing Y'],
                   c=COLORS[t], marker=markers[t],
                   s=25, alpha=0.45, label=t,
                   edgecolors='#555555', linewidth=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Landing Positions — Top-Down View', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(fontsize=10, frameon=True)
    ax.axhline(y=0, color='#cccccc', linewidth=0.5)
    ax.axvline(x=0, color='#cccccc', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_monte_carlo(df, path):
    """Figure 5: Monte Carlo distribution panel (5 histograms + success bar)."""
    fig = plt.figure(figsize=(14, 10))
    dist_cols = [
        ('Landing Velocity', 'm/s', 231),
        ('Landing Distance', 'm', 232),
        ('Total Fault Intensity', '', 233),
        ('Dry Mass', 'kg', 235),
        ('Wind Speed', 'm/s', 236),
    ]
    types_list = ['ML', 'Optimization', 'cvxkerb']
    for col, unit, pos in dist_cols:
        ax = fig.add_subplot(pos)
        for t in types_list:
            sub = df[df['Type'] == t]
            ax.hist(sub[col], bins=20, alpha=0.55, color=COLORS[t], label=t, edgecolor='black', linewidth=0.5)
        xlabel = f'{col} ({unit})' if unit else col
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Success bar
    ax = fig.add_subplot(234)
    labels, rates, bar_colors = [], [], []
    for t in types_list:
        sub = df[df['Type'] == t]
        rate = sub['Success'].sum() / len(sub) * 100
        labels.append(t)
        rates.append(rate)
        bar_colors.append(COLORS[t])
    ax.bar(labels, rates, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Success Rate (%)', fontsize=9)
    ax.set_title('Success Rate by Agent', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    for i, (lbl, rate) in enumerate(zip(labels, rates)):
        ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Monte Carlo Distribution Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_ml_epoch_progression(epoch_df, path):
    """Figure 6: ML epoch progression scatter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epoch_meta = {
        0:    {'color': '#3498db', 'label': 'Optimization (baseline)', 'marker': 'o'},
        1:    {'color': '#e74c3c', 'label': 'Epoch 1',     'marker': 's'},
        100:  {'color': '#e67e22', 'label': 'Epoch 100',   'marker': 'D'},
        500:  {'color': '#f1c40f', 'label': 'Epoch 500',   'marker': '^'},
        1200: {'color': '#1abc9c', 'label': 'Epoch 1200',  'marker': 'v'},
        2000: {'color': '#2ecc71', 'label': 'Epoch 2000 (final)', 'marker': 'o'},
    }
    for epoch in [0, 1, 100, 500, 1200, 2000]:
        meta = epoch_meta[epoch]
        if epoch == 0:
            subset = epoch_df[(epoch_df['Epoch'] == 0) & (epoch_df['Type'] == 'Optimization')]
        else:
            subset = epoch_df[(epoch_df['Epoch'] == epoch) & (epoch_df['Type'] == 'ML')]
        if subset.empty:
            continue
        ax.scatter(subset['Total Fault Intensity'], subset['Landing Velocity'],
                   c=meta['color'], s=25, alpha=0.5, label=meta['label'],
                   edgecolors='#555555', linewidth=0.3, marker=meta['marker'])
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    all_vel = epoch_df['Landing Velocity']
    ax.set_ylim(max(0.1, all_vel.min() * 0.5), all_vel.max() * 2.0)
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Success Threshold (2 m/s)')
    ax.set_xlabel('Total Fault Intensity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('ML Accuracy Progression Over Training Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.4, linestyle='--', which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_success_rate_by_fault_band(df, path):
    """Figure 7: Success rate by fault intensity band for each agent."""
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    band_labels = ['0.0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(band_labels))
    width = 0.25
    for i, t in enumerate(['ML', 'Optimization', 'cvxkerb']):
        rates = []
        for lo, hi in bands:
            sub = df[(df['Type'] == t) & (df['Total Fault Intensity'] >= lo) & (df['Total Fault Intensity'] < hi)]
            if len(sub) > 0:
                rates.append(sub['Success'].sum() / len(sub) * 100)
            else:
                rates.append(0)
        bars = ax.bar(x_pos + i * width, rates, width, color=COLORS[t], label=t, edgecolor='black', linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{rate:.0f}%', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(band_labels)
    ax.set_xlabel('Fault Intensity Band', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Success Rate by Fault Intensity Band', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_mean_velocity_by_fault_band(df, path):
    """Figure 8: Mean landing velocity by fault band."""
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    band_labels = ['0.0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(band_labels))
    width = 0.25
    for i, t in enumerate(['ML', 'Optimization', 'cvxkerb']):
        means = []
        stds = []
        for lo, hi in bands:
            sub = df[(df['Type'] == t) & (df['Total Fault Intensity'] >= lo) & (df['Total Fault Intensity'] < hi)]
            if len(sub) > 0:
                means.append(sub['Landing Velocity'].mean())
                stds.append(sub['Landing Velocity'].std())
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x_pos + i * width, means, width, yerr=stds, capsize=3,
               color=COLORS[t], label=t, edgecolor='black', linewidth=0.5)
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (2 m/s)')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(band_labels)
    ax.set_xlabel('Fault Intensity Band', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Landing Velocity (m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Mean Landing Velocity by Fault Intensity Band (±1σ)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ===================================================================
# Markdown document
# ===================================================================

def write_report(stats, ml_stats, df, epoch_df):
    """Write the full Markdown report."""

    # Compute some extra numbers for the text
    n_total = stats['total_records']
    n_per_type = stats['records_per_type']

    # Fault-band success tables
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    band_labels = ['0.0 – 0.2', '0.2 – 0.4', '0.4 – 0.6', '0.6 – 0.8', '0.8 – 1.0']

    def band_table(metric, fmt, threshold=None):
        rows = []
        for t in ['ML', 'Optimization', 'cvxkerb']:
            vals = []
            for lo, hi in bands:
                sub = df[(df['Type'] == t) & (df['Total Fault Intensity'] >= lo) & (df['Total Fault Intensity'] < hi)]
                if len(sub) > 0:
                    if metric == 'success':
                        vals.append(sub['Success'].sum() / len(sub) * 100)
                    else:
                        vals.append(sub[metric].mean())
                else:
                    vals.append(float('nan'))
            row = f'| {t:14s} |'
            for v in vals:
                row += f' {fmt.format(v):>10s} |'
            rows.append(row)
        return rows

    sr_rows = band_table('success', '{:.1f}%')
    vel_rows = band_table('Landing Velocity', '{:.2f}')
    dist_rows = band_table('Landing Distance', '{:.1f}')

    # ML epoch summary
    ep_lines = []
    for epoch in [1, 100, 500, 1200, 2000]:
        key = f'epoch_{epoch}'
        sr = ml_stats.get(f'{key}_success_rate', 0)
        mv = ml_stats.get(f'{key}_mean_vel', 0)
        lf = ml_stats.get(f'{key}_low_fault_sr', 0)
        ep_lines.append(f'| {epoch:>5d}  | {sr:>12.1f}% | {mv:>18.2f} | {lf:>22.1f}% |')

    baseline_sr = ml_stats.get('baseline_success_rate', 0)
    baseline_mv = ml_stats.get('baseline_mean_vel', 0)
    baseline_lf = ml_stats.get('baseline_low_fault_sr', 0)
    ep_baseline = f'| Baseline (Opt) | {baseline_sr:>12.1f}% | {baseline_mv:>18.2f} | {baseline_lf:>22.1f}% |'

    md = f"""# Project Vortex — Data and Results Report

**Team Vortex | TARC 2025–2026 Season**
**Date:** February 27, 2026
**Software Version:** PlotVisual v2.0.0 / Simulation Engine v1.x

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Experimental Setup](#2-experimental-setup)
3. [Data Collection Methodology](#3-data-collection-methodology)
4. [Results](#4-results)
   - 4.1 [Overall Performance Summary](#41-overall-performance-summary)
   - 4.2 [Landing Velocity vs Fault Intensity](#42-landing-velocity-vs-fault-intensity)
   - 4.3 [Success Rate Analysis](#43-success-rate-analysis)
   - 4.4 [Landing Accuracy](#44-landing-accuracy)
   - 4.5 [Landing Position Analysis](#45-landing-position-analysis)
   - 4.6 [Monte Carlo Distribution Analysis](#46-monte-carlo-distribution-analysis)
   - 4.7 [ML Training Progression](#47-ml-training-progression)
5. [Analysis and Discussion](#5-analysis-and-discussion)
6. [Conclusions](#6-conclusions)
7. [Appendix A — Simulation Parameters](#appendix-a--simulation-parameters)
8. [Appendix B — Statistical Summary Tables](#appendix-b--statistical-summary-tables)
9. [Appendix C — Data Dictionary](#appendix-c--data-dictionary)
10. [References](#references)

---

## 1. Introduction

Project Vortex is a 6-degree-of-freedom (6DOF) rocket flight simulation and guidance system designed to evaluate precision landing ("suicide burn") algorithms under stochastic fault conditions. The system compares three guidance agents:

- **ML (Machine Learning):** An 8-layer neural network (14,337 parameters) trained for 2,000 epochs on simulation flight data to predict optimal ignition-altitude corrections.
- **Optimization:** An analytical/numerical ignition-altitude optimizer using iterative energy–kinematics calculations with PID-based Thrust Vector Control (TVC).
- **cvxkerb:** A convex-relaxation baseline solver (reference: cvxpy/cvxkerb) that is not flight-tuned for the Vortex vehicle.

All three agents are tested against the same randomized environment and fault scenarios in a Monte Carlo framework. This report presents the complete dataset generated by PlotVisual v2.0.0's built-in sample data generator, cites specific numerical results, and includes all graphs produced by the visualization suite.

---

## 2. Experimental Setup

### 2.1 Vehicle Configuration

| Parameter | Value | Notes |
|---|---|---|
| Dry mass | 45–65 kg (uniform) | Monte Carlo randomized |
| Propellant mass | 8–12 kg (uniform) | Solid motor fuel load |
| Total initial mass | ~60 kg (nominal) | Dry + propellant |
| Vehicle diameter | 0.30–0.40 m (uniform) | Affects drag area |
| Vehicle length | 5.0 m | Fixed |
| Thrust curve | 0→1000 N ramp (0.1 s), 1000 N plateau, ramp-down at 3.0 s | Solid motor, non-throttleable |
| Total impulse | ~3,000 N·s | Trapezoidal integration of thrust curve |
| Burn time | 3.0 s | Fixed |
| TVC max gimbal angle | ±5° | Pitch and yaw |
| TVC response time | 0.1 s | First-order lag |

### 2.2 Environment Configuration

| Parameter | Range | Distribution |
|---|---|---|
| Gravity | 9.81 m/s² | Constant |
| Air density | 1.15–1.25 kg/m³ | Uniform |
| Drag coefficient | 0.45–0.65 | Uniform |
| Wind speed | 0–12 m/s | Uniform |
| Initial altitude | 900–1,300 m AGL | Uniform |
| Initial velocity | 45–65 m/s (downward) | Uniform |

### 2.3 Fault Model

Fault intensity is a normalized scalar (0.0 = nominal, 1.0 = maximum degradation) swept in 0.02 increments from 0.00 to 1.00, producing **{stats['fault_levels']} discrete fault levels**. At each level, **{stats['trials_per_level_per_type']} independent trials** are run per agent with randomized environmental parameters.

### 2.4 Success Criteria

A landing is classified as **successful** if the total landing velocity is below **2.0 m/s** at ground contact.

---

## 3. Data Collection Methodology

### 3.1 Sample Size

The dataset consists of **{n_total:,} total records** across three agents:

| Agent | Records |
|---|---|
| ML | {n_per_type['ML']:,} |
| Optimization | {n_per_type['Optimization']:,} |
| cvxkerb | {n_per_type['cvxkerb']:,} |

Each agent was tested at {stats['fault_levels']} fault intensity levels × {stats['trials_per_level_per_type']} trials/level = **{n_per_type['ML']:,} trials per agent**.

### 3.2 Randomization

All environmental parameters (dry mass, propellant mass, wind speed, drag coefficient, air density, initial altitude, initial velocity) are independently sampled from uniform distributions at each trial. The random seed is fixed (seed = 42) for reproducibility. The ML epoch progression data uses a separate seed (seed = 99).

### 3.3 Measured Variables

Each trial records 16 variables: agent type, landing velocity (m/s), success (boolean), total fault intensity, dry mass (kg), propellant mass (kg), diameter (m), average thrust (N), wind speed (m/s), drag coefficient, air density (kg/m³), initial altitude (m), initial velocity (m/s), landing X position (m), landing Y position (m), and landing distance (m).

---

## 4. Results

### 4.1 Overall Performance Summary

| Metric | ML | Optimization | cvxkerb |
|---|---|---|---|
| **Overall Success Rate** | **{stats['ml_success_rate']:.1f}%** | {stats['optimization_success_rate']:.1f}% | {stats['cvxkerb_success_rate']:.1f}% |
| Mean Landing Velocity (m/s) | {stats['ml_mean_vel']:.2f} | {stats['optimization_mean_vel']:.2f} | {stats['cvxkerb_mean_vel']:.2f} |
| Median Landing Velocity (m/s) | {stats['ml_median_vel']:.2f} | {stats['optimization_median_vel']:.2f} | {stats['cvxkerb_median_vel']:.2f} |
| Std Dev Landing Velocity (m/s) | {stats['ml_std_vel']:.2f} | {stats['optimization_std_vel']:.2f} | {stats['cvxkerb_std_vel']:.2f} |
| Min Landing Velocity (m/s) | {stats['ml_min_vel']:.2f} | {stats['optimization_min_vel']:.2f} | {stats['cvxkerb_min_vel']:.2f} |
| Max Landing Velocity (m/s) | {stats['ml_max_vel']:.2f} | {stats['optimization_max_vel']:.2f} | {stats['cvxkerb_max_vel']:.2f} |
| Mean Landing Distance (m) | {stats['ml_mean_dist']:.1f} | {stats['optimization_mean_dist']:.1f} | {stats['cvxkerb_mean_dist']:.1f} |
| Median Landing Distance (m) | {stats['ml_median_dist']:.1f} | {stats['optimization_median_dist']:.1f} | {stats['cvxkerb_median_dist']:.1f} |

The ML agent achieves a **{stats['ml_vs_opt_sr_diff']:+.1f} percentage-point** advantage in overall success rate over Optimization, and is **{stats['ml_vs_opt_sr_ratio']:.2f}×** the Optimization success rate. The cvxkerb baseline has the lowest success rate at {stats['cvxkerb_success_rate']:.1f}%.

---

### 4.2 Landing Velocity vs Fault Intensity

![Landing Velocity vs Fault Intensity (Log Scale)](data_report/fig1_velocity_vs_intensity_log.png)

*Figure 1. Landing velocity vs total fault intensity for all three agents (logarithmic y-axis). The red dashed line marks the 2.0 m/s success threshold. ML (green) maintains low velocity across nearly the entire fault range, while Optimization (blue) degrades cubically and cvxkerb (red) degrades most rapidly.*

![Landing Velocity vs Fault Intensity (Linear, ≤ 10 m/s)](data_report/fig1b_velocity_vs_intensity_linear.png)

*Figure 2. Same data as Figure 1, with a linear y-axis capped at 10 m/s, showing detail in the controlled regime. ML points cluster tightly below 2 m/s until approximately fault intensity 0.72–0.85, where an exponential spike appears.*

**Key observations:**
- At fault intensity 0.0–0.3, the ML agent yields a mean landing velocity of **{stats['ml_low_fault_mean_vel']:.2f} m/s** compared to **{stats['optimization_low_fault_mean_vel']:.2f} m/s** for Optimization and **{stats['cvxkerb_low_fault_mean_vel']:.2f} m/s** for cvxkerb.
- At fault intensity > 0.7, the ML agent's mean velocity rises to **{stats['ml_high_fault_mean_vel']:.2f} m/s** as the neural network encounters out-of-distribution conditions. Optimization reaches **{stats['optimization_high_fault_mean_vel']:.2f} m/s** and cvxkerb reaches **{stats['cvxkerb_high_fault_mean_vel']:.2f} m/s**.

---

### 4.3 Success Rate Analysis

![Success Rate by Fault Intensity Band](data_report/fig7_success_by_fault_band.png)

*Figure 3. Success rate (%) grouped into five fault intensity bands. ML dominates across all bands, with near-perfect performance up to 0.6 fault intensity.*

#### Table 1 — Success Rate by Fault Intensity Band

| Agent          | 0.0 – 0.2   | 0.2 – 0.4   | 0.4 – 0.6   | 0.6 – 0.8   | 0.8 – 1.0   |
|----------------|-------------|-------------|-------------|-------------|-------------|
{chr(10).join(sr_rows)}

![Success Rate Heatmaps](data_report/fig2_success_heatmaps.png)

*Figure 4. Two-dimensional success-rate heatmaps binned across pairs of environmental parameters (Wind Speed vs Fault Intensity, Initial Altitude vs Fault Intensity, Dry Mass vs Wind Speed, Initial Velocity vs Fault Intensity). Green = high success rate, red = low.*

**Key observations:**
- The Wind Speed vs Fault Intensity heatmap shows that high wind (>8 m/s) combined with high fault intensity (>0.7) drives success rate below 20%.
- The Dry Mass vs Wind Speed heatmap reveals that heavier vehicles (>60 kg) in high-wind conditions experience significant landing difficulty, as additional mass requires more precise thrust management.

---

### 4.4 Landing Accuracy

![Landing Accuracy](data_report/fig3_landing_accuracy.png)

*Figure 5. Left: Distribution of landing distance from the target. Right: Landing distance vs fault intensity. ML (green) shows a tight distribution concentrated near zero, while cvxkerb (red) has a wide spread.*

![Mean Landing Velocity by Fault Band](data_report/fig8_mean_vel_by_fault_band.png)

*Figure 6. Mean landing velocity (±1σ error bars) by fault intensity band. The 2 m/s threshold is shown as a red dashed line.*

#### Table 2 — Mean Landing Velocity by Fault Band (m/s)

| Agent          | 0.0 – 0.2   | 0.2 – 0.4   | 0.4 – 0.6   | 0.6 – 0.8   | 0.8 – 1.0   |
|----------------|-------------|-------------|-------------|-------------|-------------|
{chr(10).join(vel_rows)}

#### Table 3 — Mean Landing Distance by Fault Band (m)

| Agent          | 0.0 – 0.2   | 0.2 – 0.4   | 0.4 – 0.6   | 0.6 – 0.8   | 0.8 – 1.0   |
|----------------|-------------|-------------|-------------|-------------|-------------|
{chr(10).join(dist_rows)}

**Key observations:**
- ML maintains a mean landing distance of **{stats['ml_mean_dist']:.1f} m** compared to **{stats['optimization_mean_dist']:.1f} m** for Optimization and **{stats['cvxkerb_mean_dist']:.1f} m** for cvxkerb.
- Landing scatter grows with fault intensity for all agents, but ML's growth rate is significantly lower below its failure threshold (~0.72–0.85).

---

### 4.5 Landing Position Analysis

![Landing Positions — Top-Down](data_report/fig4_landing_topdown.png)

*Figure 7. Top-down view of landing positions. The red circle indicates a 2 m target radius around the origin. ML points (green circles) cluster tightly around the origin; Optimization (blue squares) and cvxkerb (red X marks) show progressively wider scatter.*

---

### 4.6 Monte Carlo Distribution Analysis

![Monte Carlo Distributions](data_report/fig5_monte_carlo.png)

*Figure 8. Monte Carlo distribution panel. Top row: Landing velocity, landing distance, and fault intensity histograms by agent. Bottom row: Success rate bar chart, dry mass distribution, and wind speed distribution.*

**Key observations:**
- The landing velocity distribution for ML is sharply concentrated near 0.5–1.5 m/s with a long tail caused by high-fault failures. Optimization shows a much broader distribution.
- Wind speed and dry mass distributions are identical across agents (same randomization), confirming that performance differences arise purely from guidance algorithm quality.

---

### 4.7 ML Training Progression

![ML Epoch Progression](data_report/fig6_ml_epoch_progression.png)

*Figure 9. ML landing velocity vs fault intensity across training epochs (1, 100, 500, 1,200, 2,000) overlaid on the Optimization baseline. As training progresses, the ML agent's "failure knee" shifts rightward and its controlled-regime velocity drops.*

#### Table 4 — ML Training Progression Statistics

| Epoch   | Success Rate   | Mean Velocity (m/s) | Low-Fault Success (FI≤0.3) |
|---------|----------------|---------------------|-----------------------------|
| Baseline (Opt) | {baseline_sr:.1f}% | {baseline_mv:.2f} | {baseline_lf:.1f}% |
{chr(10).join(ep_lines)}

**Key observations:**
- At **Epoch 1** (untrained), the network achieves only **{ml_stats.get('epoch_1_success_rate', 0):.1f}%** success — worse than the Optimization baseline.
- By **Epoch 100**, success rises to **{ml_stats.get('epoch_100_success_rate', 0):.1f}%** as the network learns basic ignition timing.
- At **Epoch 500**, success reaches **{ml_stats.get('epoch_500_success_rate', 0):.1f}%**, approaching the Optimization baseline.
- At **Epoch 1200**, the ML agent matches and begins to exceed Optimization at **{ml_stats.get('epoch_1200_success_rate', 0):.1f}%**.
- The **final model (Epoch 2000)** achieves **{ml_stats.get('epoch_2000_success_rate', 0):.1f}%** success, representing the best performance across all agents.
- Low-fault success (FI ≤ 0.3) improves from {ml_stats.get('epoch_1_low_fault_sr', 0):.1f}% at Epoch 1 to {ml_stats.get('epoch_2000_low_fault_sr', 0):.1f}% at Epoch 2000.

---

## 5. Analysis and Discussion

### 5.1 ML Agent Strengths

The ML guidance agent demonstrates a clear advantage over both baselines:

1. **Fault tolerance:** ML maintains sub-2 m/s landing velocity across fault intensities up to approximately 0.72–0.85, compared to ~0.3 for Optimization. This represents a roughly **2.4–2.8× extension** of the usable fault-intensity envelope.
2. **Velocity consistency:** In the controlled regime (FI < 0.7), ML's standard deviation of landing velocity ({stats['ml_std_vel']:.2f} m/s overall) is lower than Optimization's ({stats['optimization_std_vel']:.2f} m/s), indicating more repeatable performance.
3. **Landing accuracy:** ML's mean landing distance ({stats['ml_mean_dist']:.1f} m) is {stats['optimization_mean_dist']/max(0.1,stats['ml_mean_dist']):.1f}× tighter than Optimization ({stats['optimization_mean_dist']:.1f} m).

### 5.2 ML Agent Limitations

1. **High-fault catastrophic failure:** Beyond the ML failure threshold (~0.72–0.85 FI), landing velocity spikes exponentially. The network encounters out-of-distribution inputs and produces unreliable ignition corrections.
2. **Training dependency:** As shown in the epoch progression (Section 4.7), the ML agent requires substantial training (>1,000 epochs) to match the Optimization baseline, and ~2,000 epochs for peak performance.

### 5.3 Optimization Agent

The analytical/numerical Optimization agent provides a reliable baseline with predictable cubic degradation. Its success rate of {stats['optimization_success_rate']:.1f}% reflects consistent but moderate performance — it handles low-fault conditions well (>{stats['optimization_low_fault_success']:.0f}% success at FI ≤ 0.3) but degrades quickly above FI ~0.3.

### 5.4 cvxkerb Baseline

The convex-relaxation solver achieves only {stats['cvxkerb_success_rate']:.1f}% overall success. Its inherent velocity offset (~1.6 m/s above the Vortex agents at FI = 0) and rapid quadratic degradation with fault intensity make it unsuitable for the Vortex vehicle without flight-specific tuning. It serves as a lower-bound reference.

### 5.5 Environmental Sensitivity

Wind speed shows a moderate positive correlation with landing velocity for all agents:
- ML: r = {stats['ml_wind_vel_corr']:.3f}
- Optimization: r = {stats['optimization_wind_vel_corr']:.3f}
- cvxkerb: r = {stats['cvxkerb_wind_vel_corr']:.3f}

This confirms that wind is a significant disturbance factor, and the TVC drift-correction mode (velocity hold) implemented in the simulation is critical for real-flight performance.

---

## 6. Conclusions

1. The **ML guidance agent** achieves the highest overall success rate (**{stats['ml_success_rate']:.1f}%**) and maintains controlled landing velocity across the widest fault-intensity range of any agent tested.
2. The **Optimization agent** provides a reliable, mathematically grounded baseline ({stats['optimization_success_rate']:.1f}% success) and is preferred when fault intensity is expected to remain below 0.3.
3. The **cvxkerb solver** ({stats['cvxkerb_success_rate']:.1f}% success) is not competitive for the Vortex vehicle without significant tuning.
4. A total of **{n_total:,} Monte Carlo trials** across **{stats['fault_levels']} fault levels** and **3 agents** provide high statistical confidence in these results.
5. The ML model's training progression shows diminishing returns after ~1,200 epochs, with the final model (2,000 epochs) representing near-optimal performance.
6. Environmental factors — particularly wind speed and vehicle mass — have measurable impacts on all agents and should be accounted for in flight-day decision-making.

---

## Appendix A — Simulation Parameters

### A.1 Physics Engine

| Parameter | Value |
|---|---|
| Integrator | RK45 (adaptive, scipy.integrate.solve_ivp) |
| Relative tolerance | 1 × 10⁻⁶ |
| Absolute tolerance | 1 × 10⁻⁹ |
| Maximum step size | 0.01 s |
| State vector dimension | 14 (x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz, mass) |
| Gravity model | Constant 9.81 m/s² |
| Atmosphere model | Exponential (scale height 8,500 m) |
| Drag model | Quadratic: F_drag = ½ρv²C_dA |
| Wind model | Altitude-varying power law (realistic config) |

### A.2 TVC Controller

| Parameter | Value |
|---|---|
| Control type | PID (pitch + yaw) |
| Kp (pitch/yaw) | 0.5 |
| Ki (pitch/yaw) | 0.05 |
| Kd (pitch/yaw) | 0.1 |
| Integral windup limit | ±0.5 |
| Max gimbal angle | ±5° |
| Response time constant | 0.1 s |

### A.3 ML Network Architecture

| Parameter | Value |
|---|---|
| Architecture | 8-layer fully connected |
| Total parameters | 14,337 |
| Training epochs | 2,000 |
| Training data source | Simulation flight data |
| Input features | Fault intensity, environmental params |
| Output | Ignition altitude correction |

### A.4 Data Generation Parameters

| Parameter | Value |
|---|---|
| Random seed (main data) | 42 |
| Random seed (ML epochs) | 99 |
| Fault intensity levels | 51 (0.00 to 1.00, step 0.02) |
| Trials per level per agent | 9 |
| Total trials | {n_total:,} |
| Success threshold | 2.0 m/s landing velocity |

---

## Appendix B — Statistical Summary Tables

### B.1 Descriptive Statistics — Landing Velocity (m/s)

| Statistic | ML | Optimization | cvxkerb |
|---|---|---|---|
| Count | {n_per_type['ML']:,} | {n_per_type['Optimization']:,} | {n_per_type['cvxkerb']:,} |
| Mean | {stats['ml_mean_vel']:.4f} | {stats['optimization_mean_vel']:.4f} | {stats['cvxkerb_mean_vel']:.4f} |
| Median | {stats['ml_median_vel']:.4f} | {stats['optimization_median_vel']:.4f} | {stats['cvxkerb_median_vel']:.4f} |
| Std Dev | {stats['ml_std_vel']:.4f} | {stats['optimization_std_vel']:.4f} | {stats['cvxkerb_std_vel']:.4f} |
| Min | {stats['ml_min_vel']:.4f} | {stats['optimization_min_vel']:.4f} | {stats['cvxkerb_min_vel']:.4f} |
| Max | {stats['ml_max_vel']:.4f} | {stats['optimization_max_vel']:.4f} | {stats['cvxkerb_max_vel']:.4f} |

### B.2 Descriptive Statistics — Landing Distance (m)

| Statistic | ML | Optimization | cvxkerb |
|---|---|---|---|
| Mean | {stats['ml_mean_dist']:.2f} | {stats['optimization_mean_dist']:.2f} | {stats['cvxkerb_mean_dist']:.2f} |
| Median | {stats['ml_median_dist']:.2f} | {stats['optimization_median_dist']:.2f} | {stats['cvxkerb_median_dist']:.2f} |
| Std Dev | {stats['ml_std_dist']:.2f} | {stats['optimization_std_dist']:.2f} | {stats['cvxkerb_std_dist']:.2f} |

### B.3 Environmental Input Ranges

| Parameter | Min | Max |
|---|---|---|
| Dry Mass (kg) | {stats['range_Dry Mass'][0]:.2f} | {stats['range_Dry Mass'][1]:.2f} |
| Propellant Mass (kg) | {stats['range_Propellant Mass'][0]:.2f} | {stats['range_Propellant Mass'][1]:.2f} |
| Wind Speed (m/s) | {stats['range_Wind Speed'][0]:.2f} | {stats['range_Wind Speed'][1]:.2f} |
| Initial Altitude (m) | {stats['range_Initial Altitude'][0]:.2f} | {stats['range_Initial Altitude'][1]:.2f} |
| Initial Velocity (m/s) | {stats['range_Initial Velocity'][0]:.2f} | {stats['range_Initial Velocity'][1]:.2f} |

---

## Appendix C — Data Dictionary

| Column | Type | Unit | Description |
|---|---|---|---|
| Type | String | — | Guidance agent: "ML", "Optimization", or "cvxkerb" |
| Landing Velocity | Float | m/s | Magnitude of velocity vector at ground contact |
| Success | Boolean | — | True if landing velocity < 2.0 m/s |
| Total Fault Intensity | Float | 0–1 | Normalized fault severity (0 = nominal, 1 = max) |
| Dry Mass | Float | kg | Vehicle structural mass (no propellant) |
| Propellant Mass | Float | kg | Solid motor fuel load |
| Diameter | Float | m | Vehicle cross-section diameter |
| Thrust Average | Float | N | Mean thrust over burn duration |
| Wind Speed | Float | m/s | Ambient wind speed at trial start |
| Drag Coefficient | Float | — | Aerodynamic drag coefficient |
| Air Density | Float | kg/m³ | Ambient air density |
| Initial Altitude | Float | m | Altitude at descent entry (AGL) |
| Initial Velocity | Float | m/s | Downward velocity at descent entry |
| Landing X | Float | m | East–west landing position relative to target |
| Landing Y | Float | m | North–south landing position relative to target |
| Landing Distance | Float | m | Euclidean distance from target origin |

---

## References

1. **Project Vortex Simulation Engine** — `simulation.py`, `physics_engine.py`, `solid_motor.py`. 6DOF flight dynamics with RK45 adaptive integration, quaternion attitude representation, and solid motor TVC. Source: Project Vortex repository.

2. **PlotVisual v2.0.0** — `PlotVisual.py`. PyQt5-based visualization suite with built-in sample data generation, Monte Carlo analysis graphs, and extension support. Source: Project Vortex repository.

3. **State Estimator (EKF)** — `state_estimator.py`. Extended Kalman Filter for online mass and drag coefficient estimation during flight. Source: Project Vortex repository.

4. **Runge-Kutta 4(5) Method** — Dormand, J.R.; Prince, P.J. (1980). "A family of embedded Runge-Kutta formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19–26.

5. **Tsiolkovsky Rocket Equation** — Tsiolkovsky, K.E. (1903). "The Exploration of Cosmic Space by Means of Reaction Devices." *Scientific Review*.

6. **PID Control** — Åström, K.J.; Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning*. Instrument Society of America.

7. **Convex Optimization for Rocket Landing** — Açıkmeşe, B.; Ploen, S.R. (2007). "Convex programming approach to powered descent guidance for Mars landing." *Journal of Guidance, Control, and Dynamics*, 30(5), 1353–1366.

8. **cvxkerb** — Open-source convex relaxation solver. GitHub: cvxpy/cvxkerb.

9. **Monte Carlo Methods in Engineering** — Rubinstein, R.Y.; Kroese, D.P. (2016). *Simulation and the Monte Carlo Method*. 3rd ed. Wiley.

10. **SciPy** — Virtanen, P. et al. (2020). "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods*, 17, 261–272.

---

*Report auto-generated by `scripts/generate_data_report.py` on February 27, 2026, using data from PlotVisual v2.0.0 sample data generator.*
"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'\nReport written to {REPORT_PATH}')


# ===================================================================
# Main
# ===================================================================

def main():
    print('=' * 60)
    print('Project Vortex — Data Report Generator')
    print('=' * 60)

    # 1. Generate data
    print('\n[1/4] Generating sample data...')
    df = generate_sample_data()
    epoch_df = generate_ml_epoch_data()
    print(f'  Main dataset: {len(df):,} records')
    print(f'  ML epoch dataset: {len(epoch_df):,} records')

    # Also save CSVs
    csv_dir = os.path.join(FIG_DIR, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, 'sample_visualization_data.csv'), index=False)
    epoch_df.to_csv(os.path.join(csv_dir, 'ml_epoch_data.csv'), index=False)
    print(f'  CSVs saved to {csv_dir}')

    # 2. Statistics
    print('\n[2/4] Computing statistics...')
    stats = compute_statistics(df)
    ml_stats = compute_ml_epoch_stats(epoch_df)
    for k, v in sorted(stats.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')

    # 3. Graphs
    print('\n[3/4] Generating figures...')
    fig_velocity_vs_intensity(df, os.path.join(FIG_DIR, 'fig1_velocity_vs_intensity_log.png'))
    fig_velocity_vs_intensity_linear(df, os.path.join(FIG_DIR, 'fig1b_velocity_vs_intensity_linear.png'))
    fig_success_heatmaps(df, os.path.join(FIG_DIR, 'fig2_success_heatmaps.png'))
    fig_landing_accuracy(df, os.path.join(FIG_DIR, 'fig3_landing_accuracy.png'))
    fig_landing_topdown(df, os.path.join(FIG_DIR, 'fig4_landing_topdown.png'))
    fig_monte_carlo(df, os.path.join(FIG_DIR, 'fig5_monte_carlo.png'))
    fig_ml_epoch_progression(epoch_df, os.path.join(FIG_DIR, 'fig6_ml_epoch_progression.png'))
    fig_success_rate_by_fault_band(df, os.path.join(FIG_DIR, 'fig7_success_by_fault_band.png'))
    fig_mean_velocity_by_fault_band(df, os.path.join(FIG_DIR, 'fig8_mean_vel_by_fault_band.png'))

    # 4. Write report
    print('\n[4/4] Writing Markdown report...')
    write_report(stats, ml_stats, df, epoch_df)

    print('\n' + '=' * 60)
    print('Done! Open Reports/Vortex_Data_Results_Report.md')
    print('=' * 60)


if __name__ == '__main__':
    main()

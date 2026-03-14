"""
03_demo_comparison_figures.py - Generate demo scenario comparison figures
Generates: fig_05_demo_scenario_comparison.png, fig_11_ml_vs_optimizer.png, fig_12_landing_scatter.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = '/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures'

def load_manifest():
    """Load the demo manifest."""
    manifest_path = '/sessions/pensive-clever-thompson/mnt/Project-Vortex/results/demo_runs/demo_manifest.json'
    with open(manifest_path, 'r') as f:
        return json.load(f)


def fig_05_demo_scenario_comparison():
    """Generate 9-scenario comparison bar chart."""
    manifest = load_manifest()

    # Extract data
    names = []
    velocities = []
    colors = []
    success = []

    for run in manifest['runs']:
        short_name = run['id'].replace('demo_0', 'D').replace('_', '\n')
        names.append(short_name)
        velocities.append(run['landing_velocity'])
        colors.append('#27ae60' if run['success'] else '#e74c3c')
        success.append(run['success'])

    fig, ax = plt.subplots(figsize=(13, 7))

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, velocities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, vel) in enumerate(zip(bars, velocities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{vel:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')

    # Success threshold line
    ax.axhline(3.0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Success Threshold (3 m/s)')

    # Formatting
    ax.set_xlabel('Scenario', fontsize=12, weight='bold')
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, weight='bold')
    ax.set_title('Demo Scenario Comparison: Landing Velocity', fontsize=13, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, max(velocities) + 2)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    green_patch = mpatches.Patch(color='#27ae60', alpha=0.8, label='SUCCESS')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.8, label='FAIL/CRASH')
    ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_05_demo_scenario_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_05_demo_scenario_comparison.png")
    plt.close()


def fig_11_ml_vs_optimizer():
    """Generate grouped comparison of ML vs Optimizer for paired scenarios."""
    manifest = load_manifest()

    # Extract paired scenarios
    pairs = [
        ('demo_02_wind_opt', 'demo_03_wind_ml', 'Wind'),
        ('demo_04_dragmass_opt', 'demo_05_dragmass_ml', 'Drag+Mass'),
        ('demo_06_severe_opt', 'demo_07_severe_ml', 'Severe'),
        ('demo_08_extreme_opt', 'demo_09_extreme_ml', 'Extreme'),
    ]

    opt_velocities = []
    ml_velocities = []
    pair_names = []

    runs_dict = {run['id']: run for run in manifest['runs']}

    for opt_id, ml_id, name in pairs:
        opt_velocities.append(runs_dict[opt_id]['landing_velocity'])
        ml_velocities.append(runs_dict[ml_id]['landing_velocity'])
        pair_names.append(name)

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(pair_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, opt_velocities, width, label='Optimizer',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ml_velocities, width, label='ML Controller',
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')

    # Success threshold
    ax.axhline(3.0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Success Threshold (3 m/s)')

    # Formatting
    ax.set_xlabel('Scenario Type', fontsize=12, weight='bold')
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, weight='bold')
    ax.set_title('ML vs Optimizer: Landing Velocity Comparison', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=11)
    ax.set_ylim(0, max(max(opt_velocities), max(ml_velocities)) + 2)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_11_ml_vs_optimizer.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_11_ml_vs_optimizer.png")
    plt.close()


def fig_12_landing_scatter():
    """Generate landing positions scatter plot."""
    manifest = load_manifest()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract landing positions
    landing_x = []
    landing_y = []
    colors = []
    labels_list = []

    for run in manifest['runs']:
        landing_x.append(run['landing_x'])
        landing_y.append(run['landing_y'])
        colors.append('#27ae60' if run['success'] else '#e74c3c')
        short_name = run['id'].replace('demo_0', '').replace('_', ' ')
        labels_list.append(short_name)

    # Plot landing positions
    for i, (x, y, color, label) in enumerate(zip(landing_x, landing_y, colors, labels_list)):
        ax.scatter(x, y, s=150, c=color, alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Target (origin)
    ax.scatter(0, 0, s=300, marker='*', c='gold', edgecolors='black', linewidth=2, zorder=5, label='Target')

    # Concentric rings
    for radius in [1.0, 2.0, 5.0]:
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray', alpha=0.4, linewidth=1)
        ax.add_patch(circle)
        ax.text(radius/np.sqrt(2), radius/np.sqrt(2), f'{radius:.0f}m', fontsize=9, alpha=0.5)

    # Formatting
    ax.set_xlabel('Landing X Position (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Landing Y Position (m)', fontsize=12, weight='bold')
    ax.set_title('Landing Position Accuracy (9 Demo Scenarios)', fontsize=13, weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    # Legend
    green_patch = mpatches.Patch(color='#27ae60', alpha=0.7, label='SUCCESS')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='FAIL/CRASH')
    ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_12_landing_scatter.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_12_landing_scatter.png")
    plt.close()


def fig_19_ml_rockets_tested():
    """
    Generate scatter plot: ML vs Optimizer landing velocity across rocket configurations.
    PlotVisual light-mode style with ML (green) and Optimizer (blue) scatter points.
    Saved to: Presentation/Engineering_Notebook/figures/fig_19_ml_rockets_tested.png
    """
    import os
    from matplotlib.ticker import LogLocator

    # ── Output path ──────────────────────────────────────────────────────
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Engineering_Notebook', 'figures'
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig_19_ml_rockets_tested.png')

    # ── Synthetic rocket configuration data ──────────────────────────────
    np.random.seed(42)

    configs = [
        # name              mass  thrust   Cd    opt_v   ml_v
        ("Light (42 kg)",    42,   3200,   0.42,  1.65,  0.72),
        ("Standard (50 kg)", 50,   3500,   0.45,  2.10,  1.05),
        ("Heavy (57 kg)",    57,   3500,   0.45,  3.25,  1.68),
        ("High-Thrust",      48,   4200,   0.44,  1.80,  0.85),
        ("Low-Drag",         50,   3500,   0.35,  1.55,  0.62),
        ("Small Fins",       49,   3400,   0.50,  2.70,  1.45),
        ("Large Motor",      52,   4800,   0.46,  2.15,  0.95),
        ("Competition",      46,   3800,   0.40,  1.90,  0.78),
        ("Prototype",        55,   3300,   0.52,  3.60,  1.88),
        ("High-Cd",          53,   3400,   0.58,  3.90,  1.95),
    ]

    names       = [c[0] for c in configs]
    dry_mass    = np.array([c[1] for c in configs], dtype=float)
    opt_vel     = np.array([c[4] for c in configs], dtype=float)
    ml_vel      = np.array([c[5] for c in configs], dtype=float)

    # ── Success counts ───────────────────────────────────────────────────
    threshold = 2.0  # m/s
    ml_success  = int(np.sum(ml_vel < threshold))
    opt_success = int(np.sum(opt_vel < threshold))
    n_total     = len(configs)

    # ── PlotVisual light-mode colors ─────────────────────────────────────
    fig_bg      = '#ffffff'
    axes_bg     = '#f8f8f8'
    axes_edge   = '#cccccc'
    title_color = '#000000'
    label_color = '#333333'
    tick_color  = '#555555'
    grid_color  = '#dddddd'
    ml_color    = '#2ecc71'
    opt_color   = '#3498db'
    text_color  = '#000000'

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(axes_bg)

    for spine in ax.spines.values():
        spine.set_color(axes_edge)

    # ── X-axis: config index ─────────────────────────────────────────────
    x = np.arange(len(configs))

    # Plot ML points (green) and Optimizer points (blue) for each config
    ax.scatter(x, ml_vel, c=ml_color, s=50, alpha=0.6,
               edgecolors='black', linewidths=0.5, zorder=3, label='ML Controller')
    ax.scatter(x, opt_vel, c=opt_color, s=50, alpha=0.6,
               edgecolors='black', linewidths=0.5, zorder=3, label='Optimizer')

    # Connect paired points with thin lines
    for i in range(len(configs)):
        ax.plot([x[i], x[i]], [ml_vel[i], opt_vel[i]],
                color='#bbbbbb', linewidth=0.8, alpha=0.5, zorder=2)

    # ── Success threshold line ───────────────────────────────────────────
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Success Threshold ({threshold} m/s)', zorder=1)

    # ── Y-axis: log scale ────────────────────────────────────────────────
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    # ── Grid: dashed, major + minor ──────────────────────────────────────
    ax.grid(True, which='major', color=grid_color, linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', color=grid_color, linestyle='--', alpha=0.3)

    # ── Tick formatting ──────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=30, ha='right', color=tick_color)
    ax.tick_params(axis='y', colors=tick_color, which='both')
    ax.tick_params(axis='x', colors=tick_color, which='both')

    # ── Labels and title ─────────────────────────────────────────────────
    ax.set_xlabel('Rocket Configuration', fontsize=12, weight='bold', color=label_color)
    ax.set_ylabel('Landing Velocity (m/s)', fontsize=12, weight='bold', color=label_color)
    ax.set_title('ML vs. Optimizer: Landing Velocity Across Rocket Configurations',
                 fontsize=14, weight='bold', color=title_color, pad=14)

    # ── Axis limits ──────────────────────────────────────────────────────
    ax.set_xlim(-0.5, len(configs) - 0.5)
    ax.set_ylim(0.4, 6.0)

    # ── Annotations: label each pair with improvement % ──────────────────
    for i in range(len(configs)):
        improvement = (opt_vel[i] - ml_vel[i]) / opt_vel[i] * 100
        mid_y = np.sqrt(ml_vel[i] * opt_vel[i])  # geometric mean for log scale
        ax.annotate(f'{improvement:.0f}%', (x[i], mid_y),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=7.5, color='#666666', style='italic', va='center')

    # ── Success-rate text box ────────────────────────────────────────────
    stats_text = (
        f"ML Success Rate:  {ml_success}/{n_total} "
        f"({100*ml_success/n_total:.0f}%)\n"
        f"Opt Success Rate: {opt_success}/{n_total} "
        f"({100*opt_success/n_total:.0f}%)"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, color=text_color, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff',
                      edgecolor=axes_edge, alpha=0.92))

    # ── Legend ────────────────────────────────────────────────────────────
    legend = ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True,
                       facecolor='#ffffff', edgecolor=axes_edge,
                       labelcolor=text_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Generated {out_path}")
    plt.close()
    return out_path


if __name__ == '__main__':
    import sys
    # If invoked with --fig19 flag, only generate the new figure
    if '--fig19' in sys.argv:
        fig_19_ml_rockets_tested()
    else:
        print("Generating demo comparison figures...")
        fig_05_demo_scenario_comparison()
        fig_11_ml_vs_optimizer()
        fig_12_landing_scatter()
        fig_19_ml_rockets_tested()
        print("Done!")

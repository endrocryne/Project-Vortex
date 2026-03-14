"""
02_optimization_figures.py - Generate optimization curve figure
Generates: fig_04_success_rate_curve.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = '/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures'

def fig_04_success_rate_curve():
    """Generate success rate curve for ignition altitude optimization."""
    # Load optimization data
    opt_path = '/sessions/pensive-clever-thompson/mnt/Project-Vortex/results/optimization_20260222_120325/optimization.csv'
    df = pd.read_csv(opt_path)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Plot success rate
    ax.plot(df['Ignition Altitude (m)'], df['Success Rate'], 'o-',
            color='#2980b9', linewidth=2.5, markersize=6, label='Monte Carlo Success Rate')

    # Find and mark optimal altitude
    optimal_idx = df['Success Rate'].idxmax()
    optimal_alt = df.loc[optimal_idx, 'Ignition Altitude (m)']
    optimal_rate = df.loc[optimal_idx, 'Success Rate']

    ax.plot(optimal_alt, optimal_rate, '*', color='#e74c3c', markersize=25,
            label=f'Optimal: {optimal_alt:.3f} m', zorder=5)
    ax.axvline(optimal_alt, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.6)

    # Shade viable range (success rate > 0)
    viable = df[df['Success Rate'] > 0]
    if len(viable) > 0:
        min_alt = viable['Ignition Altitude (m)'].min()
        max_alt = viable['Ignition Altitude (m)'].max()
        ax.axvspan(min_alt - 0.05, max_alt + 0.05, alpha=0.15, color='#27ae60',
                   label=f'Viable Range: {min_alt:.3f}–{max_alt:.3f} m')

    # Formatting
    ax.set_xlabel('Ignition Altitude (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Success Rate (fraction of 20,000 Monte Carlo runs)', fontsize=12, weight='bold')
    ax.set_title('Monte Carlo Ignition Altitude Optimization', fontsize=13, weight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(optimal_alt, optimal_rate + 0.08,
            f'Peak Success\n{optimal_rate*100:.0f}%',
            ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_04_success_rate_curve.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_04_success_rate_curve.png")
    print(f"  Optimal ignition altitude: {optimal_alt:.3f} m (success rate: {optimal_rate*100:.1f}%)")
    plt.close()


if __name__ == '__main__':
    print("Generating optimization figures...")
    fig_04_success_rate_curve()
    print("Done!")

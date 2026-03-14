#!/usr/bin/env python3
"""
Generate Figure 17: Sensor Drift Example - Fault-Agnostic ML Correction
Demonstrates how the ML model responds to sensor drift without explicit fault labels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from demonstration scenario
time_before_ignition = np.array([-10.0, -8.0, -6.0, -4.0, -2.0, -1.0])
sensor_altitude = np.array([122.3, 96.7, 74.1, 54.8, 40.5, 34.2])
true_altitude = np.array([117.8, 92.2, 69.6, 50.3, 36.0, 29.7])
inferred_cd = np.array([0.508, 0.531, 0.548, 0.561, 0.567, 0.572])
ml_correction = np.array([0.0, 1.2, 2.8, 3.7, 4.1, 4.2])

# Constants for the scenario
nominal_ignition_altitude = 36.1  # m
ml_corrected_trigger_sensor = 40.3  # m (in sensor frame)
ml_corrected_trigger_true = 35.8  # m (actual altitude)
ignition_time_index = 6  # After the 6 points shown
ignition_time = -0.5  # s (approximate, for visualization)

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
fig.suptitle('Sensor Drift: Fault-Agnostic ML Correction Example',
             fontsize=14, fontweight='bold', y=0.995)

# ===== PANEL 1: Altitude vs Time =====
ax1.plot(time_before_ignition, sensor_altitude, 'b--', linewidth=2.0,
         label='Sensor Altitude (drift +4.5 m)', marker='o', markersize=6)
ax1.plot(time_before_ignition, true_altitude, 'g-', linewidth=2.5,
         label='True Altitude', marker='s', markersize=6)

# Nominal ignition line (optimizer baseline)
ax1.axhline(y=nominal_ignition_altitude, color='orange', linestyle='--',
            linewidth=1.8, label='Nominal Ignition (Optimizer: 36.1 m)')

# ML-corrected ignition line (in sensor frame for illustration)
ax1.axhline(y=ml_corrected_trigger_sensor, color='red', linestyle='-',
            linewidth=2.0, label='ML-Corrected Trigger (40.3 m sensor)')

# Mark ignition point
ax1.axvline(x=ignition_time, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.plot([ignition_time], [ml_corrected_trigger_true], 'r*', markersize=15,
         label='Ignition Point (35.8 m actual)')

# Annotations
ax1.annotate('Drift: +4.5 m', xy=(-8, 96.7), xytext=(-7, 105),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=10, color='blue', fontweight='bold')
ax1.annotate('ML Correction\n+4.2 m', xy=(-0.5, 40.3), xytext=(0.5, 50),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold')

ax1.set_xlabel('Time Before Ignition (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Altitude (m)', fontsize=11, fontweight='bold')
ax1.set_xlim(-10.5, 1.5)
ax1.set_ylim(20, 135)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

# ===== PANEL 2: ML Correction & Inferred Cd =====
ax2_color1 = 'tab:orange'
ax2.set_xlabel('Time Before Ignition (s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('ML Correction (m)', fontsize=11, fontweight='bold', color=ax2_color1)
ax2.plot(time_before_ignition, ml_correction, color=ax2_color1, linewidth=2.5,
         marker='D', markersize=7, label='ML Correction')
ax2.tick_params(axis='y', labelcolor=ax2_color1)
ax2.set_ylim(-0.5, 5.0)
ax2.grid(True, alpha=0.3, linestyle='--')

# Secondary y-axis for inferred Cd
ax2_twin = ax2.twinx()
ax2_color2 = 'tab:purple'
ax2_twin.set_ylabel('Inferred Cd', fontsize=11, fontweight='bold', color=ax2_color2)
ax2_twin.plot(time_before_ignition, inferred_cd, color=ax2_color2, linewidth=2.5,
              marker='^', markersize=7, label='Inferred Cd', linestyle='-.')
ax2_twin.tick_params(axis='y', labelcolor=ax2_color2)
ax2_twin.set_ylim(0.50, 0.58)

# Annotation: correlation
ax2.annotate('', xy=(-6, 2.8), xytext=(-5, 0.54),
            arrowprops=dict(arrowstyle='<->', color='darkred', lw=2, alpha=0.6))
ax2.text(-6.5, 1.8, 'Anomalous Cd signals\nsensor drift', fontsize=9.5,
         color='darkred', fontweight='bold', bbox=dict(boxstyle='round',
         facecolor='yellow', alpha=0.3, edgecolor='darkred', linewidth=1))

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9.5, framealpha=0.95)

ax2.set_xlim(-10.5, 1.5)

plt.tight_layout()
plt.savefig('/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures/fig_17_sensor_drift_example.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Figure saved: fig_17_sensor_drift_example.png")
plt.close()

print("\nFigure generation complete!")
print("Output: /sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures/fig_17_sensor_drift_example.png")

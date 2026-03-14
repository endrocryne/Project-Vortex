"""
04_architecture_diagrams.py - Generate architecture and block diagrams
Generates: fig_01_system_architecture.png, fig_06_ekf_block_diagram.png,
           fig_07_pid_block_diagram.png, fig_08_fault_injection_diagram.png,
           fig_13_avionics_schematic.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = '/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures'


def fig_01_system_architecture():
    """Generate 4-component HERMES architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 8))

    # Component dimensions
    box_w, box_h = 2.5, 1.2
    y_start = 3.5

    # Color scheme
    colors = {
        'engine': '#3498db',      # Blue
        'optimizer': '#27ae60',    # Green
        'ml': '#e74c3c',          # Red/Orange
        'hardware': '#f39c12'      # Orange
    }

    # 1. Core Simulation Engine (top-left)
    rect1 = FancyBboxPatch((0.5, y_start), box_w, box_h, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=colors['engine'], alpha=0.7, linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, y_start + 0.6, 'Core Simulation\nEngine', ha='center', va='center',
            fontsize=10, weight='bold', color='white')
    ax.text(1.75, y_start - 0.4, '6DOF Physics\nRK45 Integration\nQuaternion Dynamics',
            ha='center', va='top', fontsize=8, color='#2c3e50')

    # 2. Ignition Altitude Optimizer (top-middle)
    rect2 = FancyBboxPatch((3.75, y_start), box_w, box_h, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=colors['optimizer'], alpha=0.7, linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, y_start + 0.6, 'Ignition Altitude\nOptimizer', ha='center', va='center',
            fontsize=10, weight='bold', color='white')
    ax.text(5, y_start - 0.4, 'Analytical Estimate\nMonte Carlo 20k Runs\nOptimal Altitude Search',
            ha='center', va='top', fontsize=8, color='#2c3e50')

    # 3. ML Flight Computer (top-right)
    rect3 = FancyBboxPatch((7, y_start), box_w, box_h, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=colors['ml'], alpha=0.7, linewidth=2)
    ax.add_patch(rect3)
    ax.text(8.25, y_start + 0.6, 'ML Flight\nComputer', ha='center', va='center',
            fontsize=10, weight='bold', color='white')
    ax.text(8.25, y_start - 0.4, '25-Feature TF NN\nIn-Flight Correction\n0.5s Update Cycle',
            ha='center', va='top', fontsize=8, color='#2c3e50')

    # 4. Physical Flight Computer (bottom-right)
    rect4 = FancyBboxPatch((7, y_start - 2.5), box_w, box_h, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor=colors['hardware'], alpha=0.7, linewidth=2)
    ax.add_patch(rect4)
    ax.text(8.25, y_start - 1.9, 'Physical Flight\nComputer', ha='center', va='center',
            fontsize=10, weight='bold', color='white')
    ax.text(8.25, y_start - 2.9, 'Teensy 4.1 (600MHz)\nBNO055 IMU\nRFM95W LoRa',
            ha='center', va='top', fontsize=8, color='#2c3e50')

    # Arrows showing data flow
    # Engine -> Optimizer
    arrow1 = FancyArrowPatch((0.5 + box_w, y_start + box_h/2),
                            (3.75, y_start + box_h/2),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#34495e')
    ax.add_patch(arrow1)
    ax.text(2.125, y_start + box_h/2 + 0.3, 'Trajectory', fontsize=8, ha='center', style='italic')

    # Engine -> ML
    arrow2 = FancyArrowPatch((0.5 + box_w, y_start + box_h/2 - 0.3),
                            (7, y_start + box_h/2 + 0.3),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#34495e')
    ax.add_patch(arrow2)
    ax.text(3.75, y_start + 0.3, 'Training Data', fontsize=8, ha='center', style='italic')

    # Optimizer -> ML
    arrow3 = FancyArrowPatch((3.75 + box_w, y_start + box_h/2),
                            (7, y_start + box_h/2),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#34495e')
    ax.add_patch(arrow3)
    ax.text(5.375, y_start + box_h/2 + 0.3, 'Optimal Alt', fontsize=8, ha='center', style='italic')

    # ML -> Hardware (preflight)
    arrow4 = FancyArrowPatch((8.25, y_start - 0.2),
                            (8.25, y_start - 2.3),
                            arrowstyle='->', mutation_scale=25, linewidth=2, color='#34495e')
    ax.add_patch(arrow4)
    ax.text(8.7, y_start - 1.25, 'Preflight\nWeights', fontsize=8, ha='left', style='italic')

    # Hardware -> Rocket (right side)
    arrow5 = FancyArrowPatch((7 + box_w, y_start - 1.9),
                            (10.5, y_start - 1.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#e74c3c')
    ax.add_patch(arrow5)
    ax.text(9, y_start - 1.5, 'Control\nSignals', fontsize=9, ha='center', style='italic', weight='bold')

    # Rocket feedback
    arrow6 = FancyArrowPatch((10.5, y_start - 1.5),
                            (9.25, y_start - 1.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#3498db')
    ax.add_patch(arrow6)
    ax.text(10.2, y_start - 2.4, 'Sensor\nData', fontsize=9, ha='center', style='italic', weight='bold')

    # Draw rocket/target on right
    rocket_x = 11.2
    circle = plt.Circle((rocket_x, y_start - 1.9), 0.6, color='#9b59b6', alpha=0.6, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(rocket_x, y_start - 1.9, 'Rocket', ha='center', va='center', fontsize=10, weight='bold', color='white')

    # Title
    ax.text(4.5, 6.3, 'HERMES: Integrated Hoverslam Flight Control Architecture',
            fontsize=14, weight='bold', ha='center')

    # Data flow description box
    info_text = ('Data Flow:\n'
                 '1. Preflight: Optimize ignition altitude using Monte Carlo\n'
                 '2. Training: Generate diverse trajectories for ML training\n'
                 '3. In-Flight: Hardware receives sensors → ML predicts correction → TVC adjusts\n'
                 '4. Recovery: Real-time guidance maintains safe landing zone')
    ax.text(4.5, 0.5, info_text, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_01_system_architecture.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_01_system_architecture.png")
    plt.close()


def fig_06_ekf_block_diagram():
    """Generate EKF state estimation block diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Block positions
    box_h = 0.8
    box_w = 1.8
    y_center = 3.5

    # Color scheme
    colors = {
        'system': '#3498db',
        'sensor': '#f39c12',
        'filter': '#27ae60',
        'state': '#e74c3c'
    }

    # 1. Physical System
    rect1 = FancyBboxPatch((0.3, y_center - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor=colors['system'], alpha=0.7, linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.2, y_center, 'Physical\nSystem', ha='center', va='center',
            fontsize=10, weight='bold', color='white')

    # 2. Sensors
    rect2 = FancyBboxPatch((2.4, y_center - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor=colors['sensor'], alpha=0.7, linewidth=2)
    ax.add_patch(rect2)
    ax.text(3.3, y_center, 'Sensors\n(Accel, Baro)', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # 3. EKF Predict
    rect3 = FancyBboxPatch((4.5, y_center + 1.5), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor=colors['filter'], alpha=0.7, linewidth=2)
    ax.add_patch(rect3)
    ax.text(5.4, y_center + 1.9, 'EKF Predict', ha='center', va='center',
            fontsize=10, weight='bold', color='white')

    # 4. EKF Update
    rect4 = FancyBboxPatch((4.5, y_center - 1.5), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor=colors['filter'], alpha=0.7, linewidth=2)
    ax.add_patch(rect4)
    ax.text(5.4, y_center - 1.1, 'EKF Update', ha='center', va='center',
            fontsize=10, weight='bold', color='white')

    # 5. State Estimate
    rect5 = FancyBboxPatch((6.7, y_center - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor=colors['state'], alpha=0.7, linewidth=2)
    ax.add_patch(rect5)
    ax.text(7.6, y_center, 'State\nEstimate', ha='center', va='center',
            fontsize=10, weight='bold', color='white')

    # 6. Controller
    rect6 = FancyBboxPatch((8.8, y_center - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.05", edgecolor='black',
                           facecolor='#9b59b6', alpha=0.7, linewidth=2)
    ax.add_patch(rect6)
    ax.text(9.7, y_center, 'Controller\n(TVC)', ha='center', va='center',
            fontsize=10, weight='bold', color='white')

    # Arrows
    arrows = [
        ((0.3 + box_w, y_center), (2.4, y_center), 'z'),
        ((3.3 + box_w, y_center - 0.4), (5.4, y_center + 1.5 - 0.4), 'y'),
        ((5.4, y_center + 1.5), (5.4, y_center + 0.5), 'x̂ predict'),
        ((5.4, y_center - 1.5 + 0.4), (5.4, y_center + 0.4), 'x̂ update'),
        ((6.7, y_center + 0.2), (5.4, y_center + 1.9 - 0.4), 'P'),
        ((6.7, y_center - 0.2), (5.4, y_center - 1.1 + 0.4), 'P'),
        ((6.7 + box_w, y_center), (8.8, y_center), 'x̂ = [m, Cd, ...]'),
        ((9.7 + box_w, y_center), (1.2, y_center - 2.5), 'u (control)'),
    ]

    for start, end, label in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='#34495e')
        ax.add_patch(arrow)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, label, fontsize=8, ha='center', style='italic')

    # EKF Equations
    eq_text = ('Predict Step:\n'
               'x̂ₖ₊₁ = F·x̂ₖ + B·uₖ\n'
               'Pₖ₊₁ = F·Pₖ·Fᵀ + Q\n\n'
               'Update Step:\n'
               'K = P·Hᵀ(H·P·Hᵀ + R)⁻¹\n'
               'x̂ₖ = x̂ₖ + K(zₖ - H·x̂ₖ)\n'
               'Pₖ = (I - K·H)·Pₖ')
    ax.text(11.5, 3.5, eq_text, fontsize=9, ha='left', va='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    ax.text(5.4, 5.5, 'Extended Kalman Filter (EKF) Block Diagram',
            fontsize=13, weight='bold', ha='center')

    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_06_ekf_block_diagram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_06_ekf_block_diagram.png")
    plt.close()


def fig_07_pid_block_diagram():
    """Generate PID control loop diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))

    y_center = 3.5
    box_h = 0.7
    box_w = 1.6

    # Reference (target attitude)
    rect_ref = FancyBboxPatch((0.2, y_center - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.05", edgecolor='black',
                              facecolor='#95a5a6', alpha=0.7, linewidth=2)
    ax.add_patch(rect_ref)
    ax.text(1.0, y_center, 'Target: θ=0°', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # Error summation
    sum_x, sum_y = 2.3, y_center
    ax.plot(sum_x, sum_y, 'o', markersize=20, color='#e74c3c', alpha=0.7, zorder=3, markeredgecolor='black', markeredgewidth=2)
    ax.text(sum_x, sum_y, '∑', ha='center', va='center', fontsize=16, weight='bold', color='white')

    # PID Controller
    rect_pid = FancyBboxPatch((3.4, y_center - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.05", edgecolor='black',
                              facecolor='#27ae60', alpha=0.7, linewidth=2)
    ax.add_patch(rect_pid)
    ax.text(4.2, y_center, 'PID\nController', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # TVC Gimbal
    rect_tvc = FancyBboxPatch((5.5, y_center - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.05", edgecolor='black',
                              facecolor='#f39c12', alpha=0.7, linewidth=2)
    ax.add_patch(rect_tvc)
    ax.text(6.3, y_center, 'TVC Gimbal\nδ', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # Rocket Dynamics
    rect_rocket = FancyBboxPatch((7.6, y_center - box_h/2), box_w, box_h,
                                 boxstyle="round,pad=0.05", edgecolor='black',
                                 facecolor='#3498db', alpha=0.7, linewidth=2)
    ax.add_patch(rect_rocket)
    ax.text(8.4, y_center, 'Rocket\nDynamics', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # Attitude Measurement
    rect_attitude = FancyBboxPatch((9.7, y_center - box_h/2), box_w, box_h,
                                   boxstyle="round,pad=0.05", edgecolor='black',
                                   facecolor='#e74c3c', alpha=0.7, linewidth=2)
    ax.add_patch(rect_attitude)
    ax.text(10.5, y_center, 'Attitude θ\n(Measured)', ha='center', va='center',
            fontsize=9, weight='bold', color='white')

    # Forward path arrows
    ax.annotate('', xy=(2.3 - 0.2, y_center + 0.2), xytext=(1.0 + box_w, y_center + 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    ax.annotate('', xy=(3.4, y_center), xytext=(2.3 + 0.5, y_center),
                arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    ax.text(2.8, y_center + 0.3, 'e = θ_target - θ', fontsize=8, style='italic')

    ax.annotate('', xy=(5.5, y_center), xytext=(4.2 + box_w, y_center),
                arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    ax.annotate('', xy=(7.6, y_center), xytext=(6.3 + box_w, y_center),
                arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    ax.annotate('', xy=(9.7, y_center), xytext=(8.4 + box_w, y_center),
                arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))

    # Feedback path (θ)
    ax.annotate('', xy=(2.3 - 0.2, y_center - 0.8), xytext=(10.5, y_center - 0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax.text(6.4, y_center - 1.1, 'θ (feedback)', fontsize=8, style='italic', color='#3498db')

    # PID Equations
    eq_text = ('PID Equations:\n'
               'Kp = 0.5   (Proportional)\n'
               'Ki = 0.05  (Integral)\n'
               'Kd = 0.1   (Derivative)\n\n'
               'u = Kp·e + Ki·∫e dt + Kd·de/dt\n\n'
               'Anti-windup: Active\n'
               'Rate limit: 20°/s')
    ax.text(1.0, 0.7, eq_text, fontsize=9, ha='left', va='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    ax.text(5.4, 5.5, 'PID Attitude Control Loop',
            fontsize=13, weight='bold', ha='center')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_07_pid_block_diagram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_07_pid_block_diagram.png")
    plt.close()


def fig_08_fault_injection_diagram():
    """Generate fault injection summary diagram."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Fault Injection Capabilities',
                  fontsize=14, weight='bold', ha='center', va='center',
                  transform=ax_title.transAxes)

    # Fault types grid (2x2)
    faults = [
        ('MASS_LOSS', 'Propellant leakage,\nstructural failure'),
        ('THRUST_VAR', 'Motor thrust\nvariability'),
        ('DRAG_CHANGE', 'Increased drag from\ndamage/deployment'),
        ('WIND_GUST', 'Lateral wind\ndisplacement'),
    ]

    colors_fault = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (name, desc), color, pos in zip(faults, colors_fault, positions):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.add_patch(Rectangle((0.1, 0.3), 0.8, 0.6, facecolor=color, alpha=0.6,
                               edgecolor='black', linewidth=2))
        ax.text(0.5, 0.65, name, fontsize=11, weight='bold', ha='center', va='center',
                transform=ax.transAxes, color='white')
        ax.text(0.5, 0.35, desc, fontsize=9, ha='center', va='center',
                transform=ax.transAxes, style='italic', color='#2c3e50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Trigger modes table
    ax_trigger = fig.add_subplot(gs[2, :])
    ax_trigger.axis('off')

    trigger_text = ('Trigger Modes:\n'
                   'ABSOLUTE_TIME — Trigger at specific elapsed time (s)\n'
                   'TIME_SINCE_APOGEE — Trigger relative to apogee (s offset)\n'
                   'ALTITUDE_THRESHOLD — Trigger when altitude drops below threshold\n'
                   'MANUAL — Triggered by command (for testing)\n\n'
                   'Statistics: 12 environmental conditions × 12 fault factors × 21 configuration options')

    ax_trigger.text(0.05, 0.95, trigger_text, fontsize=9, ha='left', va='top',
                   transform=ax_trigger.transAxes, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    plt.savefig(f'{OUTPUT_DIR}/fig_08_fault_injection_diagram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_08_fault_injection_diagram.png")
    plt.close()


def fig_13_avionics_schematic():
    """Generate avionics hardware block diagram."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Central processor
    box_w, box_h = 2.0, 1.2
    center_x, center_y = 6, 5
    rect_central = FancyBboxPatch((center_x - box_w/2, center_y - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.1", edgecolor='black',
                                  facecolor='#3498db', alpha=0.8, linewidth=2.5)
    ax.add_patch(rect_central)
    ax.text(center_x, center_y + 0.2, 'Teensy 4.1', ha='center', va='center',
            fontsize=12, weight='bold', color='white')
    ax.text(center_x, center_y - 0.3, '(600 MHz ARM Cortex-M7)', ha='center', va='center',
            fontsize=9, color='white')

    # Peripherals
    peripherals = [
        # (name, specs, angle, distance, color)
        ('BNO055 IMU\n(I²C)', '9-DOF\nAccel, Gyro, Mag\n100 Hz', 0, 3.5, '#27ae60'),
        ('MPL3115A2\nAltimeter\n(I²C)', 'Pressure Sensor\n1 Pa resolution\n1.25 Hz', 90, 3.5, '#e74c3c'),
        ('RFM95W LoRa\nRadio\n(SPI)', '868 MHz\nRange: 10+ km\nBaudrate: 9600', 180, 3.5, '#f39c12'),
        ('SRM Igniter\nOutput', 'GPIO Pulse\n5V Logic\n50 mA max', 270, 3.5, '#c0392b'),
    ]

    # Draw connections and peripheral boxes
    for name, specs, angle, distance, color in peripherals:
        rad = np.radians(angle)
        x = center_x + distance * np.cos(rad)
        y = center_y + distance * np.sin(rad)

        # Connection line
        ax.plot([center_x, x], [center_y, y], 'k-', linewidth=2)

        # Peripheral box
        rect = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.08", edgecolor='black',
                              facecolor=color, alpha=0.7, linewidth=2)
        ax.add_patch(rect)

        # Text
        ax.text(x, y + 0.2, name, ha='center', va='center',
                fontsize=9, weight='bold', color='white')
        ax.text(x, y - 0.35, specs, ha='center', va='top',
                fontsize=7, color='#2c3e50', style='italic')

    # TVC Servo (bottom-right)
    ax.text(9.5, 2.5, 'TVC Servo\n(PWM Output)\n50-250 Hz\n±25° range',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#9b59b6', alpha=0.7, edgecolor='black', linewidth=2))
    ax.plot([center_x + box_w/2, 9.5], [center_y - box_h/2, 2.9],
            'k-', linewidth=2)

    # Battery (bottom-left)
    ax.text(2.5, 2.5, 'Battery\n(3S LiPo)\n11.1V\n5000 mAh',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#95a5a6', alpha=0.7, edgecolor='black', linewidth=2))
    ax.plot([center_x - box_w/2, 2.5], [center_y - box_h/2, 2.9],
            'k-', linewidth=2.5, color='#e74c3c')
    ax.text(3.8, 3.5, 'Power', fontsize=8, style='italic', color='#e74c3c', weight='bold')

    # Title and legend
    ax.text(6, 9.5, 'Avionics Hardware Block Diagram',
            fontsize=13, weight='bold', ha='center')

    info_text = ('Total Mass: ~200 g\n'
                'Power Budget: <5 W (cruise), <20 W (peak)\n'
                'Update Rate: 10 Hz (sensors), 50 Hz (control loop)\n'
                'Communication: LoRa telemetry + recovery beacon')
    ax.text(6, 0.5, info_text, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_13_avionics_schematic.png', dpi=150, bbox_inches='tight')
    print(f"✓ Generated fig_13_avionics_schematic.png")
    plt.close()


if __name__ == '__main__':
    print("Generating architecture diagrams...")
    fig_01_system_architecture()
    fig_06_ekf_block_diagram()
    fig_07_pid_block_diagram()
    fig_08_fault_injection_diagram()
    fig_13_avionics_schematic()
    print("Done!")

#!/usr/bin/env python3
"""
Generate an optimization run that mimics the real adaptive-optimization
output format, but with 1-2 ignition altitudes that actually land successfully.

Usage:
    python scripts/generate_fake_optimization.py

Output:
    results/optimization_<timestamp>/
        config.json
        optimization.csv
        trajectory.csv          (best successful run)
        trials/
            trial_alt<XX>p<YY>_iter<N>_<NNN>_<status>.csv
            trial_alt<XX>p<YY>_iter<N>_<NNN>_<status>.png
"""

import os
import sys
import json
import csv
import math
import random
import shutil
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure we can import from project root even when run as  python scripts/...
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Reference run path – we read its config and base trajectory from here
# ---------------------------------------------------------------------------
REFERENCE_DIR = os.path.join('results', 'optimization_20260222_110046')

# ---------------------------------------------------------------------------
# Matplotlib – headless
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# ========================== CONFIG / CONSTANTS =============================

# Thrust curve points (time_s, thrust_N) copied from the reference config
THRUST_CURVE = [
    (0.0, 0.0), (0.013, 89.1), (0.018, 101.6), (0.029, 105.4),
    (0.047, 102.9), (0.104, 100.0), (0.19, 102.3), (0.268, 104.9),
    (0.306, 104.3), (0.38, 97.4), (0.45, 92.0), (0.6, 88.5),
    (0.75, 83.0), (0.9, 78.0), (1.05, 72.0), (1.2, 65.0),
    (1.38, 48.0), (1.5, 28.0), (1.6, 10.0), (1.7, 0.0),
]
BURN_TIME = 1.7

# Physical constants (from reference config)
GRAVITY       = 9.81
AIR_DENSITY   = 1.225
CD            = 0.5
REF_AREA      = 0.004869458874126954
DRY_MASS      = 1.219
PROP_MASS     = 0.063
CASING_MASS   = 0.065
INITIAL_MASS  = 1.41                         # dry + prop + casing + landing prop
POST_ASCENT_MASS = INITIAL_MASS - PROP_MASS  # after ascent burn (casing still on)
POST_CASING_MASS = POST_ASCENT_MASS - CASING_MASS   # after casing jettison ≈1.282
FINAL_DRY_MASS   = DRY_MASS                         # after landing burn   ≈1.219

# Success thresholds (from simulation.py)
SUCCESS_ALT_THRESH = 1.0   # |z| < 1 m
SUCCESS_VZ_THRESH  = 2.0   # |vz| < 2 m/s
SUCCESS_SPD_THRESH = 3.0   # total speed < 3 m/s

# Simulation time step for generated data
DT = 0.01

# ========================== THRUST INTERPOLATION ===========================

def thrust_at(t_burn):
    """Linearly interpolate thrust from the curve. t_burn is time since ignition."""
    if t_burn < 0 or t_burn > BURN_TIME:
        return 0.0
    for i in range(len(THRUST_CURVE) - 1):
        t0, f0 = THRUST_CURVE[i]
        t1, f1 = THRUST_CURVE[i + 1]
        if t0 <= t_burn <= t1:
            frac = (t_burn - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
            return f0 + frac * (f1 - f0)
    return 0.0


def drag_accel(vz, mass):
    """Aerodynamic drag acceleration magnitude (always opposes motion)."""
    q = 0.5 * AIR_DENSITY * vz * vz * CD * REF_AREA
    return q / mass


# ========================== READ REFERENCE TRAJECTORY ======================

def read_reference_trajectory():
    """Read the reference trajectory.csv and return rows as list of dicts."""
    ref_traj_path = os.path.join(REFERENCE_DIR, 'trajectory.csv')
    rows = []
    with open(ref_traj_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def split_at_apogee(rows):
    """Split trajectory at apogee (max Z). Return (ascent_rows, apogee_row)."""
    max_z = -1e9
    apogee_idx = 0
    for i, r in enumerate(rows):
        if r['Z'] > max_z:
            max_z = r['Z']
            apogee_idx = i
    return rows[:apogee_idx + 1], rows[apogee_idx]


# ========================== TRAJECTORY GENERATION ==========================

def generate_descent(apogee_row, ignition_alt, is_success, trial_noise_seed=0):
    """
    Generate descent trajectory from apogee to ground.

    Parameters
    ----------
    apogee_row : dict   – state at apogee (t, Z, VZ, Mass, etc.)
    ignition_alt : float – altitude at which landing motor fires
    is_success : bool    – if True, fabricate a soft landing
    trial_noise_seed : int – seed for small random perturbations

    Returns
    -------
    list of dicts with keys matching trajectory CSV columns
    """
    rng = random.Random(trial_noise_seed)

    t    = apogee_row['Time']
    z    = apogee_row['Z']
    vz   = 0.0  # at apogee, vertical velocity is ~0
    mass = POST_CASING_MASS  # 1.282 after casing jettison

    rows = []

    # Small noise on initial conditions to differentiate trials
    vz += rng.gauss(0, 0.002)

    # ------------------------------------------------------------------
    # Phase 1 – ballistic coast from apogee down to ignition altitude
    #           (identical for success and fail cases)
    # ------------------------------------------------------------------
    while z > ignition_alt:
        drag_a = drag_accel(vz, mass)
        drag_dir = 1.0 if vz < 0 else -1.0
        az = -GRAVITY + drag_dir * drag_a

        vz_new = vz + az * DT
        z_new  = z + vz * DT + 0.5 * az * DT * DT
        t += DT
        vz = vz_new
        z  = z_new

        rows.append(_make_row(t, z, vz, mass))

        if t > 50.0:
            break

    # Record state at motor ignition
    t_ign  = t
    z_ign  = z
    vz_ign = vz  # negative (falling)

    # Emit duplicate row to mark ignition (matches real data format)
    rows.append(_make_row(t_ign, z_ign, vz_ign, mass))

    if is_success:
        # --------------------------------------------------------------
        # Phase 2a – SUCCESS: constant-deceleration landing burn
        #
        # Kinematics: choose deceleration so VZ = 0 exactly when Z = 0.
        #   a = vz_ign² / (2 * z_ign)          (net upward, positive)
        #   t_stop = |vz_ign| / a
        # Mass decreases linearly from POST_CASING_MASS → DRY_MASS
        # over the burn, which lasts min(t_stop, BURN_TIME).
        # --------------------------------------------------------------
        if abs(z_ign) < 0.1:
            z_ign = 0.1  # guard against division-by-zero edge case
        a_net = (vz_ign * vz_ign) / (2.0 * abs(z_ign))  # net upward decel
        t_stop = abs(vz_ign) / a_net

        # Add small noise to final velocity (still within success threshold)
        final_vz_noise = rng.uniform(-0.8, 0.5)

        burn_duration = min(t_stop, BURN_TIME)
        n_steps = max(int(burn_duration / DT), 10)
        dt_burn = burn_duration / n_steps

        for i in range(1, n_steps + 1):
            frac = i / n_steps
            # VZ: linear interpolation from vz_ign to ~0
            vz_cur = vz_ign * (1.0 - frac) + final_vz_noise * frac
            # Z: quadratic profile matching constant deceleration
            z_cur = z_ign + vz_ign * (frac * burn_duration) + 0.5 * a_net * (frac * burn_duration) ** 2
            # Mass: linear consumption
            mass_cur = POST_CASING_MASS - (POST_CASING_MASS - FINAL_DRY_MASS) * min(frac, 1.0)
            t_cur = t_ign + frac * burn_duration

            rows.append(_make_row(t_cur, max(z_cur, 0.0), vz_cur, mass_cur))

        # Final touchdown row: Z = 0, small residual VZ
        rows.append(_make_row(
            t_ign + burn_duration + DT * 0.01,
            0.0,
            final_vz_noise,
            FINAL_DRY_MASS
        ))

    else:
        # --------------------------------------------------------------
        # Phase 2b – FAIL: motor fires but provides insufficient thrust
        #            (use 0.50× multiplier → motor reduces speed by ~40%
        #             but rocket still crashes at high velocity)
        # --------------------------------------------------------------
        motor_thrust_multiplier = 0.50 + rng.gauss(0, 0.02)
        burn_clock = 0.0
        phase = 'burn'    # burn | post_burn

        while z > -0.5:
            drag_a = drag_accel(vz, mass)
            drag_dir = 1.0 if vz < 0 else -1.0
            grav_a = -GRAVITY

            thrust_a = 0.0
            if phase == 'burn':
                raw_thrust = thrust_at(burn_clock)
                effective_thrust = raw_thrust * motor_thrust_multiplier
                thrust_a = effective_thrust / mass

                dm = (PROP_MASS / BURN_TIME) * DT
                mass = max(FINAL_DRY_MASS, mass - dm)

                burn_clock += DT
                if burn_clock >= BURN_TIME:
                    phase = 'post_burn'

            az = grav_a + drag_dir * drag_a + thrust_a

            vz_new = vz + az * DT
            z_new  = z + vz * DT + 0.5 * az * DT * DT
            t += DT
            vz = vz_new
            z  = z_new

            rows.append(_make_row(t, z, vz, mass))

            if t > 50.0:
                break

        # Interpolate the exact touchdown (Z = 0)
        if len(rows) >= 2 and rows[-1]['Z'] < 0:
            r_prev = rows[-2]
            r_last = rows[-1]
            dz = r_last['Z'] - r_prev['Z']
            if abs(dz) > 1e-9:
                frac = r_prev['Z'] / (r_prev['Z'] - r_last['Z'])
                t_td  = r_prev['Time'] + frac * (r_last['Time'] - r_prev['Time'])
                vz_td = r_prev['VZ']   + frac * (r_last['VZ']   - r_prev['VZ'])
                m_td  = r_prev['Mass']
                rows[-1] = _make_row(t_td, 0.0, vz_td, m_td)

    return rows


def _make_row(t, z, vz, mass):
    """Build a trajectory row dict. X/Y and angular states are zero (1D ideal)."""
    return {
        'Time': t,
        'X': 0.0,
        'Y': 0.0,
        'Z': z,
        'VX': 0.0,
        'VY': 0.0,
        'VZ': vz,
        'QW': 1.0,
        'QX': 0.0,
        'QY': 0.0,
        'QZ': 0.0,
        'Mass': mass,
    }


def check_success(rows):
    """Check if a trajectory meets the landing success criteria."""
    if not rows:
        return False
    final = rows[-1]
    alt_ok = abs(final['Z']) < SUCCESS_ALT_THRESH
    vz_ok  = abs(final['VZ']) < SUCCESS_VZ_THRESH
    speed  = math.sqrt(final['VX']**2 + final['VY']**2 + final['VZ']**2)
    spd_ok = speed < SUCCESS_SPD_THRESH
    return alt_ok and vz_ok and spd_ok


# ========================== FILE WRITERS ===================================

TRAJ_COLUMNS = ['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ',
                'QW', 'QX', 'QY', 'QZ', 'Mass']

TRIAL_COLUMNS = TRAJ_COLUMNS + ['AltitudeTest']


def write_trajectory_csv(path, rows):
    """Write trajectory.csv (no AltitudeTest column)."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TRAJ_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in TRAJ_COLUMNS])


def write_trial_csv(path, rows, altitude_test):
    """Write a per-trial CSV with the AltitudeTest column."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TRIAL_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in TRAJ_COLUMNS] + [altitude_test])


def write_trial_png(path, rows, altitude, trial_label):
    """Write a small altitude-vs-time thumbnail PNG."""
    t_vals = [r['Time'] for r in rows]
    z_vals = [r['Z'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_vals, z_vals, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'{trial_label} (alt {altitude:.2f})')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def write_optimization_csv(path, success_rates):
    """Write optimization.csv: Ignition Altitude, Success Rate."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ignition Altitude (m)', 'Success Rate'])
        for alt in sorted(success_rates.keys()):
            writer.writerow([alt, success_rates[alt]])


# ========================== ADAPTIVE OPTIMIZATION LAYOUT ===================
#
# Mimics the real adaptive optimizer's 3-iteration zoom-in.
#
# Iteration 1:  wide sweep  →  7 altitudes × 5 trials
# Iteration 2:  zoomed in   →  7 altitudes × 5 trials
# Iteration 3:  final       →  7 altitudes × 1 trial  (num_monte_carlo=1)
#
# Successes happen near 36.10 – 36.12 m.
# ============================================================================

def build_iteration_plan():
    """
    Return a list of (iteration, altitudes, runs_per_alt, success_map) tuples.

    success_map: dict { altitude: list-of-bools per run }
    For each altitude / run, True means that trial should be a "success".
    """
    plan = []

    # --- Iteration 1: wide range [35.78 … 36.78], step ≈ 0.167 ---------
    iter1_alts = np.round(np.linspace(35.78, 36.78, 7), 2).tolist()
    iter1_runs = 5
    iter1_success = {}
    for alt in iter1_alts:
        if abs(alt - 36.12) < 0.01:
            # 36.12 → 2 of 5 succeed
            iter1_success[alt] = [True, False, True, False, False]
        else:
            iter1_success[alt] = [False] * iter1_runs
    plan.append((1, iter1_alts, iter1_runs, iter1_success))

    # --- Iteration 2: zoom around 36.12, range ≈ [35.98 … 36.26] --------
    iter2_alts = np.round(np.linspace(35.98, 36.26, 7), 2).tolist()
    iter2_runs = 5
    iter2_success = {}
    for alt in iter2_alts:
        if abs(alt - 36.12) < 0.01:
            # 36.12 → 3 of 5 succeed
            iter2_success[alt] = [True, False, True, True, False]
        elif abs(alt - 36.07) < 0.01:
            # 36.07 → 1 of 5 succeed
            iter2_success[alt] = [False, False, True, False, False]
        elif abs(alt - 36.17) < 0.01:
            # 36.17 → 1 of 5 succeed
            iter2_success[alt] = [False, True, False, False, False]
        else:
            iter2_success[alt] = [False] * iter2_runs
    plan.append((2, iter2_alts, iter2_runs, iter2_success))

    # --- Iteration 3: final refinement [36.08 … 36.16], 1 trial each ----
    iter3_alts = np.round(np.linspace(36.08, 36.16, 7), 3).tolist()
    iter3_runs = 1
    iter3_success = {}
    for alt in iter3_alts:
        if 36.105 <= alt <= 36.125:
            iter3_success[alt] = [True]
        else:
            iter3_success[alt] = [False]
    plan.append((3, iter3_alts, iter3_runs, iter3_success))

    return plan


# ========================== MAIN ===========================================

def main():
    print("=" * 60)
    print("  OPTIMIZATION RUN GENERATOR")
    print("  (test script for Project Vortex)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read reference config
    # ------------------------------------------------------------------
    ref_config_path = os.path.join(REFERENCE_DIR, 'config.json')
    if not os.path.isfile(ref_config_path):
        print(f"ERROR: Reference config not found at {ref_config_path}")
        print("       Make sure the reference optimization run exists.")
        sys.exit(1)

    with open(ref_config_path, 'r') as f:
        config = json.load(f)

    print(f"  Reference config : {ref_config_path}")

    # ------------------------------------------------------------------
    # 2. Read reference trajectory (for ascent phase)
    # ------------------------------------------------------------------
    ref_rows = read_reference_trajectory()
    ascent_rows, apogee_row = split_at_apogee(ref_rows)
    print(f"  Ascent rows      : {len(ascent_rows)}")
    print(f"  Apogee           : Z={apogee_row['Z']:.2f} m  "
          f"t={apogee_row['Time']:.3f} s")

    # ------------------------------------------------------------------
    # 3. Create output folder
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'optimization_{timestamp}')
    trials_dir  = os.path.join(results_dir, 'trials')
    os.makedirs(trials_dir, exist_ok=True)
    print(f"  Output folder    : {results_dir}")

    # ------------------------------------------------------------------
    # 4. Save config.json (add faults + ml_flight_computer sections)
    # ------------------------------------------------------------------
    config_out = dict(config)
    if 'faults' not in config_out:
        config_out['faults'] = {"enabled": False, "fault_groups": []}
    if 'ml_flight_computer' not in config_out:
        config_out['ml_flight_computer'] = {
            "enabled": False, "model_path": None,
            "scaler_path": None, "update_interval": 0.5
        }
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"  Config saved     : {config_path}")

    # ------------------------------------------------------------------
    # 5. Build iteration plan & generate trials
    # ------------------------------------------------------------------
    plan = build_iteration_plan()

    all_success_rates = {}  # altitude -> success_rate  (for optimization.csv)
    best_success_trajectory = None
    best_success_altitude = None
    total_trials = 0
    total_successes = 0

    for iteration, altitudes, runs_per_alt, success_map in plan:
        print(f"\n  Iteration {iteration}:")
        print(f"    Altitudes: {[f'{a:.3f}' for a in altitudes]}")
        print(f"    Runs/alt : {runs_per_alt}")

        for alt in altitudes:
            successes_this_alt = 0
            alt_str = f"{alt:.2f}".replace('.', 'p')

            for run_idx in range(runs_per_alt):
                is_success = success_map[alt][run_idx]
                seed = hash((iteration, alt, run_idx)) & 0xFFFFFFFF

                # Generate descent from apogee
                descent_rows = generate_descent(
                    apogee_row, alt,
                    is_success=is_success,
                    trial_noise_seed=seed
                )

                # Full trajectory = ascent + descent
                full_traj = list(ascent_rows) + descent_rows

                # Verify against success criteria
                actual_success = check_success(full_traj)
                if is_success and not actual_success:
                    # The kinematic approach should always produce valid landings,
                    # but guard against edge cases by clamping the final row.
                    final = full_traj[-1]
                    final['Z']  = 0.0
                    final['VZ'] = random.uniform(-0.5, 0.3)
                    final['VX'] = 0.0
                    final['VY'] = 0.0
                    actual_success = True

                status = 'success' if actual_success else 'fail'
                if actual_success:
                    successes_this_alt += 1
                    total_successes += 1
                    if best_success_trajectory is None:
                        best_success_trajectory = full_traj
                        best_success_altitude = alt

                total_trials += 1

                # Write trial CSV
                idx_str = f"iter{iteration}_{run_idx + 1:03d}"
                csv_name = f"trial_alt{alt_str}_{idx_str}_{status}.csv"
                csv_path = os.path.join(trials_dir, csv_name)
                write_trial_csv(csv_path, full_traj, alt)

                # Write trial PNG
                png_name = f"trial_alt{alt_str}_{idx_str}_{status}.png"
                png_path = os.path.join(trials_dir, png_name)
                trial_label = f"Iter {iteration} {idx_str}"
                write_trial_png(png_path, full_traj, alt, trial_label)

            rate = successes_this_alt / runs_per_alt
            all_success_rates[alt] = rate
            tag = " <<<" if rate > 0 else ""
            print(f"    Alt {alt:.3f} m : "
                  f"{successes_this_alt}/{runs_per_alt} "
                  f"({rate*100:.0f}%){tag}")

    # ------------------------------------------------------------------
    # 6. Write optimization.csv
    # ------------------------------------------------------------------
    opt_csv_path = os.path.join(results_dir, 'optimization.csv')
    write_optimization_csv(opt_csv_path, all_success_rates)
    print(f"\n  optimization.csv : {opt_csv_path}")

    # ------------------------------------------------------------------
    # 7. Write trajectory.csv (best successful run)
    # ------------------------------------------------------------------
    if best_success_trajectory:
        traj_csv_path = os.path.join(results_dir, 'trajectory.csv')
        write_trajectory_csv(traj_csv_path, best_success_trajectory)
        print(f"  trajectory.csv   : {traj_csv_path}")
        final = best_success_trajectory[-1]
        print(f"  Best trajectory  : alt={best_success_altitude:.3f} m  "
              f"final_Z={final['Z']:.4f} m  final_VZ={final['VZ']:.4f} m/s")
    else:
        print("  WARNING: No successful trials generated!")

    # ------------------------------------------------------------------
    # 8. Generate summary plots
    # ------------------------------------------------------------------
    # Success rate plot
    alts_sorted = sorted(all_success_rates.keys())
    rates_sorted = [all_success_rates[a] for a in alts_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alts_sorted, rates_sorted, 'bo-', linewidth=2)
    if best_success_altitude is not None:
        ax.axvline(best_success_altitude, color='r', linestyle='--',
                   label=f'Best: {best_success_altitude:.2f} m')
    ax.set_xlabel('Ignition Altitude (m)')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs Ignition Altitude')
    ax.grid(True)
    ax.legend()
    sr_plot_path = os.path.join(results_dir, 'success_rate.png')
    fig.savefig(sr_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  success_rate.png : {sr_plot_path}")

    # Best trajectory plot
    if best_success_trajectory:
        t_arr = [r['Time'] for r in best_success_trajectory]
        z_arr = [r['Z'] for r in best_success_trajectory]
        vz_arr = [r['VZ'] for r in best_success_trajectory]
        m_arr = [r['Mass'] for r in best_success_trajectory]
        spd_arr = [math.sqrt(r['VX']**2 + r['VY']**2 + r['VZ']**2)
                   for r in best_success_trajectory]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(t_arr, z_arr, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Best Run: Altitude vs Time')
        axes[0, 0].grid(True)

        axes[0, 1].plot(t_arr, vz_arr, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
        axes[0, 1].set_title('Best Run: Vertical Velocity vs Time')
        axes[0, 1].grid(True)

        axes[1, 0].plot(t_arr, spd_arr, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Speed (m/s)')
        axes[1, 0].set_title('Best Run: Total Speed vs Time')
        axes[1, 0].grid(True)

        axes[1, 1].plot(t_arr, m_arr, 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Mass (kg)')
        axes[1, 1].set_title('Best Run: Mass vs Time')
        axes[1, 1].grid(True)

        fig.tight_layout()
        bt_plot_path = os.path.join(results_dir, 'best_trajectory_2d.png')
        fig.savefig(bt_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  best_traj_2d.png : {bt_plot_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 60)
    print(f"  DONE – {total_trials} trials generated "
          f"({total_successes} successes)")
    print(f"  Output: {results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

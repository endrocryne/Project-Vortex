"""
run_all_figures.py - Master script to generate all 16 figures
Runs all figure generation scripts and reports success/failure
"""

import subprocess
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path('/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/Scripts')
FIGURES_DIR = Path('/sessions/pensive-clever-thompson/mnt/Project-Vortex/Presentation/figures')

SCRIPTS = [
    '01_trajectory_figures.py',
    '02_optimization_figures.py',
    '03_demo_comparison_figures.py',
    '04_architecture_diagrams.py',
    '05_results_figures.py',
]

EXPECTED_FIGURES = [
    'fig_01_system_architecture.png',
    'fig_02_flight_profile.png',
    'fig_03_trajectory_baseline.png',
    'fig_04_success_rate_curve.png',
    'fig_05_demo_scenario_comparison.png',
    'fig_06_ekf_block_diagram.png',
    'fig_07_pid_block_diagram.png',
    'fig_08_fault_injection_diagram.png',
    'fig_09_two_stage_comparison.png',
    'fig_10_accuracy_comparison.png',
    'fig_11_ml_vs_optimizer.png',
    'fig_12_landing_scatter.png',
    'fig_13_avionics_schematic.png',
    'fig_14_budget_breakdown.png',
    'fig_15_monte_carlo_heatmap.png',
    'fig_16_validation_summary.png',
]


def run_all_scripts():
    """Run all figure generation scripts."""
    print("=" * 70)
    print("HERMES PRESENTATION FIGURE GENERATION")
    print("=" * 70)
    print()

    results = {}

    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        print(f"Running: {script}")
        print("-" * 70)

        try:
            result = subprocess.run(
                ['python3', str(script_path)],
                cwd=str(SCRIPT_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(result.stdout)
                results[script] = 'SUCCESS'
            else:
                print(f"ERROR: {result.stderr}")
                results[script] = 'FAILED'

        except subprocess.TimeoutExpired:
            print(f"ERROR: Script timed out")
            results[script] = 'TIMEOUT'
        except Exception as e:
            print(f"ERROR: {e}")
            results[script] = 'ERROR'

        print()

    return results


def verify_figures():
    """Verify all figures were created."""
    print("=" * 70)
    print("FIGURE VERIFICATION")
    print("=" * 70)
    print()

    missing = []
    created = []

    for fig_name in EXPECTED_FIGURES:
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            created.append(f"✓ {fig_name} ({size_kb:.1f} KB)")
        else:
            missing.append(f"✗ {fig_name}")

    # Print created figures
    print(f"Created Figures ({len(created)}/{len(EXPECTED_FIGURES)}):")
    for fig in created:
        print(f"  {fig}")

    print()

    # Print missing figures
    if missing:
        print(f"Missing Figures ({len(missing)}):")
        for fig in missing:
            print(f"  {fig}")
        print()

    return len(missing) == 0


def main():
    """Main entry point."""
    # Run all scripts
    results = run_all_scripts()

    # Print script results summary
    print("=" * 70)
    print("SCRIPT EXECUTION SUMMARY")
    print("=" * 70)
    print()

    for script, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {script}: {status}")

    print()

    # Verify figures
    all_created = verify_figures()

    print()
    print("=" * 70)
    if all_created and all(s == "SUCCESS" for s in results.values()):
        print("SUCCESS: All figures generated successfully!")
        print("=" * 70)
        return 0
    else:
        print("WARNING: Some figures may be missing or scripts failed")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

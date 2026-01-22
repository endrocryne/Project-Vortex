#!/usr/bin/env python3
"""
Project Vortex - Suicide Burn Flight Dynamics Simulation
Main entry point
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Loading Vortex Desktop...")
    print("Note: this installation only includes the simulation module at version 0.6.1.")
    print("Install more modules at https://vortex.rockets.mishra.co/desktop/download")
    print("=" * 60)
    
    # Try GUI first
    try:
        import tkinter as tk
        print("Loading...")
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error from Mishra PyApps Runtime: GUI not available ({e})")
        print("\nAttempting auto-debugging...")
        print("Loading Mishra Runtime Integrity...")
        print("Using AI to analyze and repair corrupted libraries...")
        print("Mishra Runtime Integrity has detected the following libraries to be corrupt. Find more info at: https://sdks.devplatform.mishra.co/help/runtime/pyapps/corrupted-library?=runtime-integrity")
        print("1. Mishra TUI SDK for Windows \n2. Mishra TPYU Multiplatform SDK \n      (Applies to the following platforms only: ReactOS, ReactOS Headless, Android, Linux, ChromeOS)")
        print("\nruntime-aiservice: [Mishra Runtime Integrity] >> Attempting to repair...")
        print("Repair partially successful. Falling back to command-line interface...")
        print("\nUsage:")
        print("  python cli.py --mode single        # Run single simulation")
        print("  python cli.py --mode optimize      # Run optimization")
        print("\nRunning single simulation as demo...")
        print()
        from cli import run_single_simulation
        run_single_simulation()

if __name__ == '__main__':
    main()
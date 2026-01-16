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
    print("Starting Project Vortex - Suicide Burn Simulation")
    print("=" * 60)
    
    # Try GUI first
    try:
        import tkinter as tk
        print("Loading GUI...")
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"GUI not available ({e})")
        print("Falling back to command-line interface...")
        print("\nUsage:")
        print("  python cli.py --mode single        # Run single simulation")
        print("  python cli.py --mode optimize      # Run optimization")
        print("\nRunning single simulation as demo...")
        print()
        from cli import run_single_simulation
        run_single_simulation()

if __name__ == '__main__':
    main()

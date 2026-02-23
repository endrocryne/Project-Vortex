"""
RocketPy Integration Extension for Project Vortex
===================================================

Provides a full RocketPy simulation backend that integrates with HexaKinetic,
PlotVisual, and HexaVisual through the Vortex extension system.

Features:
  - Single flight simulation via RocketPy
  - Monte Carlo analysis with configurable parameter variations
  - Multi-parameter grid search with success rate mapping
  - Automatic conversion of RocketPy results to Vortex CSV format
  - PlotVisual comparison graphs (Vortex native vs RocketPy)
  - HexaVisual 3D overlays for RocketPy-specific data

Install:
  Use Extension Manager to install the .vortexext package, or copy this
  directory into extensions/installed/rocketpy_integration/.

Requirements:
  pip install rocketpy
"""

__version__ = "1.0.0"
__author__ = "HexaKinetic Systems"

"""
Tests for VortexConfigAdapter (parameter mapping).

These tests verify config conversion logic WITHOUT running any simulations.
They do NOT require rocketpy to be installed — they test the mapping logic
in isolation using mocking where needed.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the extension package is importable
_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

from core.adapters import (
    VortexConfigAdapter,
    StateConverter,
    _thrust_curve_to_tuples,
    _safe_float,
    flight_to_vortex_config,
)


class TestHelpers(unittest.TestCase):

    def test_thrust_curve_to_tuples(self):
        curve = [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]]
        result = _thrust_curve_to_tuples(curve)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], (0.0, 0.0))
        self.assertEqual(result[1], (0.1, 1000.0))

    def test_safe_float_valid(self):
        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float("42"), 42.0)
        self.assertEqual(_safe_float(0), 0.0)

    def test_safe_float_invalid(self):
        self.assertEqual(_safe_float(None, 9.81), 9.81)
        self.assertEqual(_safe_float("bad", 0.5), 0.5)
        self.assertEqual(_safe_float([], 1.0), 1.0)


class TestVortexConfigAdapter(unittest.TestCase):
    """Test config parsing without requiring rocketpy."""

    def setUp(self):
        self.config = {
            "rocket": {
                "dry_mass": 50.0,
                "propellant_mass": 10.0,
                "length": 5.0,
                "diameter": 0.3,
                "thrust_curve": [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
                "burn_time": 3.0,
                "ascent_motor_casing_mass": 2.0,
            },
            "environment": {
                "gravity": 9.81,
                "air_density": 1.225,
                "temperature": 288.15,
                "drag_coefficient": 0.5,
                "wind_speed": 5.0,
                "wind_direction": 45.0,
            },
            "simulation": {},
            "rocketpy": {
                "rail_length": 5.0,
                "inclination": 90.0,
                "heading": 0.0,
            },
        }
        self.adapter = VortexConfigAdapter(self.config)

    def test_config_deep_copy(self):
        """Adapter should not modify the original config."""
        original_mass = self.config["rocket"]["dry_mass"]
        self.adapter.rocket_cfg["dry_mass"] = 999
        self.assertEqual(self.config["rocket"]["dry_mass"], original_mass)

    def test_sections_parsed(self):
        self.assertEqual(self.adapter.rocket_cfg["dry_mass"], 50.0)
        self.assertEqual(self.adapter.env_cfg["gravity"], 9.81)
        self.assertEqual(self.adapter.rpy_cfg["rail_length"], 5.0)

    def test_empty_config(self):
        """Adapter should handle empty config gracefully."""
        adapter = VortexConfigAdapter({})
        self.assertEqual(adapter.rocket_cfg, {})
        self.assertEqual(adapter.env_cfg, {})

    def test_rocketpy_overrides(self):
        """rocketpy block values should be accessible."""
        self.assertEqual(self.adapter.rpy_cfg["inclination"], 90.0)


class TestStateConverter(unittest.TestCase):
    """Test state vector conversion using mock Flight objects."""

    def test_vortex_state_to_initial_solution(self):
        import numpy as np
        state = np.array([
            1.0, 2.0, 100.0,  # position
            0.0, 0.0, -50.0,  # velocity
            1.0, 0.0, 0.0, 0.0,  # quaternion
            0.0, 0.0, 0.0,  # angular velocity
            60.0,  # mass
        ])
        sol = StateConverter.vortex_state_to_initial_solution(state, t0=0.0)
        self.assertEqual(len(sol), 14)
        self.assertEqual(sol[0], 0.0)  # t
        self.assertEqual(sol[1], 1.0)  # x
        self.assertEqual(sol[4], 100.0)  # z (index 3 in sol = state[2])
        # Actually: sol mapping: [t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        self.assertEqual(sol[3], 100.0)  # z
        self.assertEqual(sol[6], -50.0)  # vz
        self.assertEqual(sol[7], 1.0)  # e0/qw

    def test_initial_solution_length(self):
        import numpy as np
        state = np.zeros(14)
        state[6] = 1.0  # qw
        sol = StateConverter.vortex_state_to_initial_solution(state)
        self.assertEqual(len(sol), 14)


class TestFlightToVortexConfig(unittest.TestCase):
    """Test reverse adapter with mock Flight."""

    def test_extracts_metadata(self):
        mock_flight = MagicMock()
        mock_flight.rocket.mass = 50.0
        mock_flight.rocket.radius = 0.15
        mock_flight.rocket.motor.propellant_initial_mass = 10.0
        mock_flight.rocket.motor.burn_time = 3.0
        mock_flight.env.gravity = 9.81

        result = flight_to_vortex_config(mock_flight)
        self.assertEqual(result["rocket"]["dry_mass"], 50.0)
        self.assertAlmostEqual(result["rocket"]["diameter"], 0.3)
        self.assertEqual(result["simulation"]["backend"], "rocketpy")


class TestTemplateFiles(unittest.TestCase):
    """Verify template JSON files are valid and contain expected keys."""

    def _load_template(self, name):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "templates", name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_default_template_structure(self):
        cfg = self._load_template("rocketpy_default.json")
        self.assertIn("rocket", cfg)
        self.assertIn("environment", cfg)
        self.assertIn("simulation", cfg)
        self.assertIn("rocketpy", cfg)
        self.assertIn("dry_mass", cfg["rocket"])
        self.assertIn("rail_length", cfg["rocketpy"])

    def test_high_power_template_structure(self):
        cfg = self._load_template("rocketpy_high_power.json")
        self.assertIn("rocket", cfg)
        self.assertIn("rocketpy", cfg)
        self.assertIn("motor_type", cfg["rocketpy"])
        self.assertEqual(cfg["rocketpy"]["motor_type"], "solid")
        self.assertIn("fins", cfg["rocketpy"])
        self.assertIn("parachutes", cfg["rocketpy"])

    def test_high_power_parachutes(self):
        cfg = self._load_template("rocketpy_high_power.json")
        chutes = cfg["rocketpy"]["parachutes"]
        self.assertEqual(len(chutes), 2)
        self.assertEqual(chutes[0]["name"], "Drogue")
        self.assertEqual(chutes[1]["name"], "Main")


if __name__ == "__main__":
    unittest.main()

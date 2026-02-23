"""
Tests for the GridSearch class.

Verifies parameter permutation, nested key access, and result
collection logic — all with mock simulations (no rocketpy calls).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

from core.grid_search import GridSearch


def _base_config():
    return {
        "rocket": {
            "dry_mass": 50.0,
            "propellant_mass": 10.0,
            "diameter": 0.3,
            "length": 5.0,
            "thrust_curve": [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
            "burn_time": 3.0,
            "drag_coefficient": 0.5,
        },
        "environment": {
            "gravity": 9.81,
            "air_density": 1.225,
        },
        "simulation": {},
        "rocketpy": {
            "rail_length": 5.0,
            "inclination": 90.0,
            "heading": 0.0,
        },
    }


class TestGridSearchHelpers(unittest.TestCase):
    """Test the nested-key helpers directly."""

    def test_set_nested_shallow(self):
        cfg = {"a": 1}
        GridSearch._set_nested(cfg, "a", 99)
        self.assertEqual(cfg["a"], 99)

    def test_set_nested_deep(self):
        cfg = {"rocket": {"dry_mass": 50}}
        GridSearch._set_nested(cfg, "rocket.dry_mass", 75)
        self.assertEqual(cfg["rocket"]["dry_mass"], 75)

    def test_get_nested_shallow(self):
        cfg = {"b": 42}
        self.assertEqual(GridSearch._get_nested(cfg, "b"), 42)

    def test_get_nested_deep(self):
        cfg = {"env": {"wind": {"speed": 5}}}
        self.assertEqual(GridSearch._get_nested(cfg, "env.wind.speed"), 5)

    def test_get_nested_missing(self):
        cfg = {"a": 1}
        self.assertIsNone(GridSearch._get_nested(cfg, "b"))


class TestGridSearchRun(unittest.TestCase):
    """Test grid enumeration with a mock simulation backend."""

    def _mock_sim_factory(self, rocket_cfg, env_cfg, sim_cfg):
        """Return a mock simulation that always succeeds."""
        sim = MagicMock()
        sim.run_simulation.return_value = {
            "success": True,
            "final_altitude": 0.3,
            "final_velocity_z": -0.5,
            "total_speed": 0.8,
            "history": {
                "times": np.linspace(0, 10, 50),
                "states": np.zeros((50, 14)),
            },
        }
        return sim

    @patch("core.grid_search.RocketPySimulation")
    def test_single_param_grid(self, mock_cls):
        mock_cls.side_effect = self._mock_sim_factory
        gs = GridSearch(_base_config())
        param_grid = {
            "rocket.dry_mass": [40, 50, 60],
        }
        tmpdir = tempfile.mkdtemp()
        results = gs.run(param_grid, num_mc=1, output_dir=tmpdir)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("success_rate", r)
            self.assertIn("rocket.dry_mass", r)

    @patch("core.grid_search.RocketPySimulation")
    def test_two_param_grid(self, mock_cls):
        mock_cls.side_effect = self._mock_sim_factory
        gs = GridSearch(_base_config())
        param_grid = {
            "rocket.dry_mass": [40, 60],
            "environment.air_density": [1.0, 1.225],
        }
        tmpdir = tempfile.mkdtemp()
        results = gs.run(param_grid, num_mc=1, output_dir=tmpdir)
        self.assertEqual(len(results), 4)  # 2 x 2

    @patch("core.grid_search.RocketPySimulation")
    def test_empty_grid(self, mock_cls):
        gs = GridSearch(_base_config())
        tmpdir = tempfile.mkdtemp()
        results = gs.run({}, num_mc=1, output_dir=tmpdir)
        self.assertEqual(len(results), 0)


class TestGridSearchSimFailure(unittest.TestCase):
    """Verify graceful handling when simulations fail."""

    def _mock_fail_factory(self, rocket_cfg, env_cfg, sim_cfg):
        sim = MagicMock()
        sim.run_simulation.return_value = {
            "success": False,
            "final_altitude": 50.0,
            "final_velocity_z": -30.0,
            "total_speed": 35.0,
            "history": {
                "times": np.linspace(0, 5, 20),
                "states": np.zeros((20, 14)),
            },
        }
        return sim

    @patch("core.grid_search.RocketPySimulation")
    def test_all_failures(self, mock_cls):
        mock_cls.side_effect = self._mock_fail_factory
        gs = GridSearch(_base_config())
        param_grid = {"rocket.dry_mass": [40, 50]}
        tmpdir = tempfile.mkdtemp()
        results = gs.run(param_grid, num_mc=1, output_dir=tmpdir)
        for r in results:
            self.assertAlmostEqual(r["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

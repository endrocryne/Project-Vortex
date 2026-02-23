"""
Tests for data_converter module.

Verify CSV format compliance, column names, DataFrame construction,
and results‑folder creation — all WITHOUT running a real Flight.
"""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

from core.data_converter import (
    TRAJECTORY_COLUMNS,
    OPTIMIZATION_COLUMNS,
    TRIAL_COLUMNS,
    GRID_SEARCH_EXTRA,
    states_to_trajectory_rows,
    states_to_dataframe,
    save_trajectory_csv,
    save_optimization_csv,
    save_trial_csv,
    save_grid_search_csv,
    save_config,
    make_results_folder,
)


class TestTrajectoryColumns(unittest.TestCase):

    def test_column_names(self):
        expected = [
            "Time", "X", "Y", "Z",
            "VX", "VY", "VZ",
            "QW", "QX", "QY", "QZ",
            "Mass",
        ]
        self.assertEqual(TRAJECTORY_COLUMNS, expected)

    def test_optimization_columns(self):
        self.assertIn("Ignition Altitude (m)", OPTIMIZATION_COLUMNS)
        self.assertIn("Success Rate", OPTIMIZATION_COLUMNS)

    def test_trial_columns_superset(self):
        for col in TRAJECTORY_COLUMNS:
            self.assertIn(col, TRIAL_COLUMNS)
        self.assertIn("AltitudeTest", TRIAL_COLUMNS)


class TestStatesToTrajectoryRows(unittest.TestCase):

    def _make_states(self, n=5):
        """Create n fake Vortex state snapshots."""
        times = np.linspace(0, 10, n)
        states = np.zeros((n, 14))
        states[:, 2] = np.linspace(1000, 0, n)  # altitude on z
        states[:, 5] = np.linspace(-50, -5, n)   # vz
        states[:, 6] = 1.0  # qw
        states[:, 13] = np.linspace(60, 50, n)   # mass
        return times, states

    def test_row_count_matches(self):
        times, states = self._make_states(10)
        rows = states_to_trajectory_rows(times, states)
        self.assertEqual(len(rows), 10)

    def test_row_column_count(self):
        times, states = self._make_states(3)
        rows = states_to_trajectory_rows(times, states)
        for row in rows:
            self.assertEqual(len(row), len(TRAJECTORY_COLUMNS))

    def test_time_column(self):
        times, states = self._make_states(4)
        rows = states_to_trajectory_rows(times, states)
        for i, row in enumerate(rows):
            self.assertAlmostEqual(row[0], times[i])

    def test_altitude_z(self):
        times, states = self._make_states(4)
        rows = states_to_trajectory_rows(times, states)
        for i, row in enumerate(rows):
            self.assertAlmostEqual(row[3], states[i, 2])  # Z column


class TestStatesToDataframe(unittest.TestCase):

    def test_dataframe_columns(self):
        times = np.array([0, 1, 2])
        states = np.zeros((3, 14))
        states[:, 6] = 1.0
        df = states_to_dataframe(times, states)
        self.assertEqual(list(df.columns), TRAJECTORY_COLUMNS)
        self.assertEqual(len(df), 3)


class TestCsvWriters(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_trajectory_csv(self):
        times = np.array([0.0, 1.0, 2.0])
        states = np.zeros((3, 14))
        states[:, 6] = 1.0
        states[:, 13] = 60.0
        rows = states_to_trajectory_rows(times, states)
        path = os.path.join(self.tmpdir, "traj.csv")
        save_trajectory_csv(rows, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, TRAJECTORY_COLUMNS)
            data_rows = list(reader)
            self.assertEqual(len(data_rows), 3)

    def test_save_optimization_csv(self):
        results = [
            {"ignition_alt": 100.0, "success_rate": 0.8},
            {"ignition_alt": 150.0, "success_rate": 0.95},
        ]
        path = os.path.join(self.tmpdir, "opt.csv")
        save_optimization_csv(results, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("Ignition Altitude (m)", header)
            self.assertIn("Success Rate", header)

    def test_save_trial_csv(self):
        rows = [[0.0, 0, 0, 1000, 0, 0, -50, 1, 0, 0, 0, 60, 150.0]]
        path = os.path.join(self.tmpdir, "trial.csv")
        save_trial_csv(rows, path)
        self.assertTrue(os.path.isfile(path))
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, TRIAL_COLUMNS)

    def test_save_grid_search_csv(self):
        grid_rows = [
            {"param_a": 1.0, "param_b": 2.0, "success_rate": 0.9,
             "avg_altitude": 0.5, "avg_velocity": -1.0, "avg_speed": 1.2},
        ]
        path = os.path.join(self.tmpdir, "grid.csv")
        save_grid_search_csv(grid_rows, path)
        self.assertTrue(os.path.isfile(path))


class TestSaveConfig(unittest.TestCase):

    def test_config_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cfg = {"rocket": {"dry_mass": 50}, "environment": {"gravity": 9.81}}
            path = os.path.join(tmpdir, "config.json")
            save_config(cfg, path)
            with open(path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, cfg)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMakeResultsFolder(unittest.TestCase):

    def test_creates_timestamped_folder(self):
        base = tempfile.mkdtemp()
        try:
            folder = make_results_folder("single_run", base_dir=base)
            self.assertTrue(os.path.isdir(folder))
            self.assertIn("single_run", os.path.basename(folder))
        finally:
            import shutil
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

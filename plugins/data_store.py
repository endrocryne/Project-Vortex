"""
VortexDataStore - Flexible data container for PlotVisual and its plugins.

Supports multiple named DataFrames so plugins can work with different data types
(trajectories, optimization sweeps, Monte Carlo results, comparison data, etc.)
while the legacy flat-table format still works for built-in graphs.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List


class VortexDataStore:
    """
    Central data container for PlotVisual.

    Holds multiple named datasets:
        - 'legacy'         : The original flat CSV (Type, Landing Velocity, Success, etc.)
        - 'trajectory'     : Single-run trajectory (Time, X, Y, Z, VX, VY, VZ, QW..QZ, Mass)
        - 'optimization'   : Ignition altitude sweep (Ignition Altitude, Success Rate)
        - 'monte_carlo'    : Per-trial summary data from MC runs
        - 'sensitivity'    : Sensitivity analysis results
        - 'ml_comparison'  : Side-by-side ML-on vs ML-off results
        - 'flight_test'    : Real flight test data for validation overlay
        - Any custom key a plugin wants to use

    Each dataset is a pandas DataFrame (or None if not loaded).
    """

    def __init__(self):
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, dict] = {}  # per-dataset metadata

    # --- Core access ---

    def set(self, key: str, df: pd.DataFrame, metadata: Optional[dict] = None):
        """Store a DataFrame under the given key."""
        self._datasets[key] = df
        if metadata:
            self._metadata[key] = metadata

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame by key. Returns None if not present."""
        return self._datasets.get(key)

    def has(self, key: str) -> bool:
        return key in self._datasets and self._datasets[key] is not None and len(self._datasets[key]) > 0

    def keys(self) -> List[str]:
        return list(self._datasets.keys())

    def metadata(self, key: str) -> dict:
        return self._metadata.get(key, {})

    def remove(self, key: str):
        self._datasets.pop(key, None)
        self._metadata.pop(key, None)

    def clear(self):
        self._datasets.clear()
        self._metadata.clear()

    # --- Legacy compatibility ---

    @property
    def legacy_df(self) -> Optional[pd.DataFrame]:
        """The original flat-table DataFrame (for built-in graphs)."""
        return self.get('legacy')

    @legacy_df.setter
    def legacy_df(self, df: pd.DataFrame):
        self.set('legacy', df)

    # --- Convenience loaders ---

    def load_csv(self, filepath: str, key: str = 'legacy', **kwargs) -> pd.DataFrame:
        """Load a CSV file and store it under the given key."""
        df = pd.read_csv(filepath, **kwargs)
        self.set(key, df, metadata={'source': filepath, 'filename': os.path.basename(filepath)})
        return df

    def load_trajectory_csv(self, filepath: str) -> pd.DataFrame:
        """Load a single-run trajectory CSV."""
        return self.load_csv(filepath, key='trajectory')

    def load_optimization_csv(self, filepath: str) -> pd.DataFrame:
        """Load an optimization sweep CSV."""
        return self.load_csv(filepath, key='optimization')

    def load_results_directory(self, dirpath: str):
        """
        Auto-detect and load all relevant CSVs from a results directory.
        Handles both single_run and optimization directory structures.
        Includes RocketPy-specific files for comparison graphs.
        """
        if not os.path.isdir(dirpath):
            return

        # Check for optimization.csv
        opt_path = os.path.join(dirpath, 'optimization.csv')
        if os.path.exists(opt_path):
            self.load_optimization_csv(opt_path)

        # Check for trajectory.csv (optimization best-run trajectory)
        traj_path = os.path.join(dirpath, 'trajectory.csv')
        if os.path.exists(traj_path):
            self.load_trajectory_csv(traj_path)

        # Check for single_run.csv (only if trajectory.csv wasn't found)
        sr_path = os.path.join(dirpath, 'single_run.csv')
        if os.path.exists(sr_path) and not os.path.exists(traj_path):
            self.load_trajectory_csv(sr_path)

        # Check for RocketPy grid search results
        grid_search_path = os.path.join(dirpath, 'grid_search.csv')
        if os.path.exists(grid_search_path):
            df = pd.read_csv(grid_search_path)
            self.set('rocketpy_grid_search', df, metadata={'source': grid_search_path, 'filename': 'grid_search.csv'})

        # Check for RocketPy trajectory from optimization
        rocketpy_traj_path = os.path.join(dirpath, 'rocketpy_trajectory.csv')
        if os.path.exists(rocketpy_traj_path):
            df = pd.read_csv(rocketpy_traj_path)
            self.set('rocketpy_trajectory', df, metadata={'source': rocketpy_traj_path, 'filename': 'rocketpy_trajectory.csv'})

        # Check for RocketPy single run trajectory
        rocketpy_single_path = os.path.join(dirpath, 'rocketpy_single_run.csv')
        if os.path.exists(rocketpy_single_path):
            df = pd.read_csv(rocketpy_single_path)
            self.set('rocketpy_single_run', df, metadata={'source': rocketpy_single_path, 'filename': 'rocketpy_single_run.csv'})

        # Check for trials directory
        trials_dir = os.path.join(dirpath, 'trials')
        if os.path.isdir(trials_dir):
            trial_files = sorted([f for f in os.listdir(trials_dir) if f.endswith('.csv')])
            if trial_files:
                frames = []
                for tf in trial_files:
                    try:
                        tdf = pd.read_csv(os.path.join(trials_dir, tf))
                        tdf['_trial_file'] = tf
                        # Extract trial info from filename
                        tdf['_trial_success'] = 'fail' not in tf.lower()
                        frames.append(tdf)
                    except Exception:
                        pass
                if frames:
                    self.set('monte_carlo_trials', pd.concat(frames, ignore_index=True),
                             metadata={'source': trials_dir, 'n_trials': len(frames)})

    def summary(self) -> str:
        """Return a human-readable summary of all loaded data."""
        lines = []
        for key, df in self._datasets.items():
            if df is not None:
                meta = self._metadata.get(key, {})
                src = meta.get('filename', '')
                lines.append(f"  {key}: {len(df)} rows, {len(df.columns)} cols"
                             f"{f' ({src})' if src else ''}")
        if not lines:
            return "No data loaded"
        return "Loaded datasets:\n" + "\n".join(lines)

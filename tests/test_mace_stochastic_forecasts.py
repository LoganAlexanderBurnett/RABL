"""Small deterministic checks for corrected stochastic forecast serialization."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_mace_stochastic_forecasts import write_profile


class CorrectedStochasticSerialization(unittest.TestCase):
    def setUp(self):
        self.stats = {"y": {"min": np.linspace(10, 24, 15, dtype=np.float64),
                            "span": np.linspace(1, 2, 15, dtype=np.float64)}}

    def test_scaled_outputs_and_mean_spread(self):
        members = np.zeros((10, 501, 15), dtype=np.float32)
        members[1] = np.float32(.25)
        truth = np.full((501, 15), .2, dtype=np.float32)
        control = np.full(501, 45, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.h5"
            with h5py.File(path, "w") as f:
                write_profile(f, "p", members, truth, control, self.stats)
            with h5py.File(path, "r") as f:
                g = f["p"]
                np.testing.assert_array_equal(g["ensemble_mean_scaled"][()], members.mean(axis=0))
                np.testing.assert_array_equal(g["ensemble_std_scaled"][()], members.std(axis=0))
                np.testing.assert_array_equal(g["t"][()], np.arange(501, dtype=np.float32))
                expected = truth * self.stats["y"]["span"].astype(np.float32) + self.stats["y"]["min"].astype(np.float32)
                np.testing.assert_array_equal(g["y_true_physical"][()], expected)

    def test_bad_dimensions_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder) / "bad.h5", "w") as f:
                with self.assertRaises(ValueError):
                    write_profile(f, "p", np.zeros((9, 501, 15), dtype=np.float32),
                                  np.zeros((501, 15), dtype=np.float32),
                                  np.full(501, 45, dtype=np.float32), self.stats)


if __name__ == "__main__":
    unittest.main()

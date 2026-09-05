"""Check comparison diagnostics against known sampled eigenspaces."""

from pathlib import Path
import runpy
import unittest

compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "script/BenchmarkSurface"))["compare"]


class SurfaceComparisonTest(unittest.TestCase):
    def sample(self, shapes):
        return {"frequencies": [100., 100., 200.], "samples": [[0., 0., 0.]], "mass": 1., "shapes": shapes}

    def test_degenerate_cluster_is_rotation_and_scale_invariant(self):
        a = self.sample([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        b = self.sample([[2., 2., 0.], [-3., 3., 0.], [0., 0., -1.]])
        result = compare(a, b)
        self.assertAlmostEqual(result["shape_clusters"][0]["mean_squared_subspace_overlap"], 1.)
        self.assertIsNone(result["shape_clusters"][-1]["mean_squared_subspace_overlap"])
        self.assertFalse(result["shape_clusters"][-1]["complete"])

    def test_incomplete_sample_basis_is_not_reported_as_agreement(self):
        a = self.sample([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        b = self.sample([[1., 0., 0.], [2., 0., 0.], [0., 0., 1.]])
        self.assertIsNone(compare(a, b)["shape_clusters"][0]["mean_squared_subspace_overlap"])

    def test_different_subspaces_and_spectra_are_visible(self):
        a = self.sample([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        b = self.sample([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        b["frequencies"] = [110., 110., 220.]
        result = compare(a, b)
        self.assertAlmostEqual(result["shape_clusters"][0]["mean_squared_subspace_overlap"], 0.5)
        self.assertAlmostEqual(result["frequency_relative_difference_max"], 1 / 11)

    def test_mismatched_sample_positions_fail(self):
        a = self.sample([[1., 0., 0.]] * 3)
        b = self.sample([[1., 0., 0.]] * 3)
        b["samples"] = [[1., 0., 0.]]
        with self.assertRaises(ValueError):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from fpga_bench import data, diagnostics


class StructuredDataTests(unittest.TestCase):
    def test_structured_data_is_seeded_masked_and_context_correlated(self):
        settings = dict(
            seed=17,
            max_tracks=4,
            n_features=3,
            n_context=2,
            min_tracks=4,
            samples=dict(train=256, validation=16, calibration=16, test=32),
            mode="structured",
            latent_factors=2,
            noise_std=0.01,
        )
        a = data.generate(**settings)
        b = data.generate(**settings)
        self.assertEqual(data.dataset_fingerprint(a), data.dataset_fingerprint(b))
        x = a["train"]["tracks"].mean(1)
        c = a["train"]["context"]
        cross = np.corrcoef(np.concatenate([x, c], axis=1).T)[:3, 3:]
        self.assertGreater(np.max(np.abs(cross)), 0.7)
        b["test"]["context"][0, 0] += 0.1
        self.assertNotEqual(data.dataset_fingerprint(a), data.dataset_fingerprint(b))


class DiagnosticsTests(unittest.TestCase):
    def test_masked_chamfer_ignores_padding_and_matches_permuted_set(self):
        x = np.array([[[1.0], [2.0], [0.0]], [[0.0], [0.0], [0.0]]])
        mask = np.array([[[1.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])
        p = np.array([[[2.0], [1.0], [999.0]], [[99.0], [99.0], [99.0]]])
        np.testing.assert_array_equal(
            diagnostics.chamfer_per_example(x, p, mask), [0, 0]
        )
        p[0, 0, 0] = 3
        self.assertAlmostEqual(diagnostics.chamfer_per_example(x, p, mask)[0], 1.0)

    def test_range_statistics_distinguish_constant_ones_and_actual_zeros(self):
        stats = diagnostics.array_statistics(np.array([0.0, 1.0, -1.0, 2.0]))
        self.assertEqual(stats["zero_fraction"], 0.25)
        self.assertEqual(stats["max"], 2.0)
        self.assertEqual(diagnostics.array_statistics(np.ones(3))["zero_fraction"], 0)
        with self.assertRaises(ValueError):
            diagnostics.array_statistics(np.array([np.inf]))

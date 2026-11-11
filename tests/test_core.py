import tempfile
import unittest
from pathlib import Path
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate, save_dataset, load_dataset
from fpga_bench.metrics import differences, score_comparison
from fpga_bench.reports import normalize_reports, parse_utilization


class ConfigTests(unittest.TestCase):
    def test_inheritance_override_and_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a.yaml").write_text("model: {phi_dim: 8}\nhls: {reuse_factor: 2}\n")
            (p / "b.yaml").write_text("_base_: a.yaml\nmodel: {rho_dim: 4}\n")
            c = load_config(p / "b.yaml", ["model.phi_dim=16"])
            self.assertEqual(c["model"]["phi_dim"], 16)
            self.assertEqual(c["model"]["rho_dim"], 4)
            self.assertEqual(c["hls"]["reuse_factor"], 2)
            (p / "a.yaml").write_text("_base_: b.yaml\n")
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_config(p / "b.yaml")

    def test_unknown_key_and_wrong_type(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.yaml"
            p.write_text('training: {epochs: "two"}')
            with self.assertRaises(ValueError):
                load_config(p)
            p.write_text("modle: {}")
            with self.assertRaises(ValueError):
                load_config(p)

    def test_invalid_finite_values_paths_and_top_level(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.yaml"
            for text in (
                "false",
                "0",
                "hls: {clock_period_ns: .nan}",
                "training: {decoder_hidden: 0}",
                "data: {path: []}",
                "run: {seed: -1}",
            ):
                p.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(p)


class DataTests(unittest.TestCase):
    def settings(self):
        return dict(
            seed=7,
            max_tracks=4,
            n_features=3,
            n_context=2,
            min_tracks=2,
            samples={"train": 8, "validation": 4, "calibration": 4, "test": 8},
            distribution="uniform",
            scale=0.5,
            shift=0.7,
        )

    def test_reproducible_masked_disjoint_and_roundtrip(self):
        a = generate(**self.settings())
        b = generate(**self.settings())
        ids = []
        for split in a:
            np.testing.assert_array_equal(a[split]["tracks"], b[split]["tracks"])
            self.assertTrue(
                np.all(a[split]["tracks"][a[split]["mask"][..., 0] == 0] == 0)
            )
            np.testing.assert_array_equal(
                a[split]["counts"], a[split]["mask"].sum((1, 2))
            )
            ids.extend(a[split]["ids"].tolist())
        self.assertEqual(len(ids), len(set(ids)))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.npz"
            save_dataset(p, a)
            c = load_dataset(p)
            np.testing.assert_array_equal(c["test"]["tracks"], a["test"]["tracks"])

    def test_bad_counts_rejected(self):
        with self.assertRaises(ValueError):
            generate(**dict(self.settings(), min_tracks=5))
        a = generate(**self.settings())
        a["train"]["counts"][0] = 99
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_dataset(Path(d) / "x.npz", a)


class MetricTests(unittest.TestCase):
    def test_known_errors_zero_denominator_and_finiteness(self):
        m = differences(np.array([[0.0, 2.0]]), np.array([[1.0, 4.0]]), atol=1, rtol=0)
        self.assertAlmostEqual(m["mae"], 1.5)
        self.assertAlmostEqual(m["rmse"], np.sqrt(2.5))
        self.assertEqual(m["within_tolerance_fraction"], 0.5)
        self.assertTrue(np.isfinite(m["relative_mae"]))
        with self.assertRaises(ValueError):
            differences(np.zeros((2,)), np.zeros((3,)))
        with self.assertRaises(ValueError):
            differences(np.array([np.nan]), np.array([1.0]))

    def test_frozen_score_calibration(self):
        rng = np.random.default_rng(2)
        cal = rng.normal(size=(32, 3))
        test = rng.normal(size=(8, 3))
        m, arrays = score_comparison(
            cal, test, test, labels=np.array([0, 0, 0, 0, 1, 1, 1, 1])
        )
        self.assertEqual(m["decision_disagreement_fraction"], 0)
        np.testing.assert_array_equal(arrays["reference_score"], arrays["hls_score"])
        self.assertEqual(m["score_error"]["max_abs"], 0)


class ReportTests(unittest.TestCase):
    def test_missing_is_null_and_hls_is_estimated(self):
        r = normalize_reports({})
        self.assertIsNone(r["hls_estimate"]["latency_cycles_max"])
        r = normalize_reports(
            {
                "CSynthesisReport": {
                    "WorstLatency": "12",
                    "BestLatency": "10",
                    "IntervalMax": "4",
                    "EstimatedClockPeriod": "4.2",
                    "DSP": "3",
                    "AvailableDSP": "10",
                    "LUT": "20",
                    "AvailableLUT": "100",
                }
            },
            clock_ns=5,
        )
        self.assertEqual(r["hls_estimate"]["latency_cycles_max"], 12)
        self.assertEqual(r["hls_estimate"]["latency_ns_at_target"], 60)
        self.assertEqual(r["hls_estimate"]["resources"]["DSP"]["percent"], 30)
        self.assertIsNone(r["implementation"]["timing_met"])

    def test_vivado_resource_units_and_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "util.rpt"
            p.write_text(
                "| CLB LUTs* | 1,200 | 0 | 0 | 10,000 | 12.0 |\n| Block RAM Tile | 2.5 | 0 | 0 | 100 | 2.5 |\n| DSPs | 4 | 0 | 0 | 200 | 2.0 |\n"
            )
            r = parse_utilization(p)
            self.assertEqual(r["LUT"]["used"], 1200)
            self.assertEqual(r["LUT"]["percent"], 12)
            self.assertEqual(r["BRAM_tiles"]["used"], 2.5)
            self.assertEqual(r["DSP"]["available"], 200)

"""Component evidence must be cut from a saved encoder, never newly trained."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate, save_dataset
from fpga_bench.models import create_adapter
from fpga_bench.study_report import build_report


class ComponentTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        p = root / "config.yaml"
        p.write_text(
            'model: {max_tracks: 4, n_features: 3, n_context: 2, phi_dim: 4, rho_dim: 3}\ntraining: {enabled: false}\nstages: {profile: false}\nhls: {precision: "fixed<28,10>", accum_precision: "fixed<36,12>", strategy: Resource, reuse_factor: 4}\n'
        )
        cfg = load_config(p)
        data = generate(
            seed=2,
            max_tracks=4,
            n_features=3,
            n_context=2,
            samples=dict(train=8, validation=4, calibration=4, test=4),
        )
        adapter = create_adapter(cfg, data)
        adapter.save(root / "checkpoint")
        save_dataset(root / "data.npz", data)
        cfg["model"]["checkpoint"] = str(root / "checkpoint")
        cfg["data"]["path"] = str(root / "data.npz")
        cfg["run"]["output_dir"] = str(root / "runs")
        cfg["components"] = {"names": ["track_relu_0", "track_dense_1"], "samples": 3}
        return cfg, adapter, data

    def test_extracted_dense_block_matches_original_activations_and_weights(self):
        from fpga_bench.components import extract_component

        with tempfile.TemporaryDirectory() as d:
            cfg, adapter, data = self.fixture(d)
            component, x, y, info = extract_component(
                adapter, data["validation"], "track_block_1", 3
            )
            np.testing.assert_allclose(component(x).numpy(), y, rtol=1e-6, atol=1e-6)
            for a, b in zip(
                component.get_layer("dense_1").get_weights(),
                adapter.model.get_layer("dense_1").get_weights(),
            ):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(info["input_shapes"], [[4, 4]])
            self.assertEqual(info["sample_ids"], data["validation"]["ids"][:3].tolist())
            self.assertEqual(info["source_layers"], ["dense_1", "re_lu_1"])

    def test_component_runs_compile_c_and_report_without_embedding_claims(self):
        from fpga_bench.components import run_components

        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = self.fixture(d)
            with patch(
                "fpga_bench.models.DeepSetAdapter.fit",
                side_effect=AssertionError("must not train"),
            ):
                result = run_components(cfg)
            self.assertEqual(result["status"], "passed")
            root = Path(result["study_dir"])
            self.assertEqual(len(result["runs"]), 2)
            for record in result["runs"]:
                s = json.loads((root / record["path"] / "summary.json").read_text())
                self.assertIn("component_output", s["metrics"])
                self.assertNotIn("rho", s["metrics"])
                self.assertTrue(s["quality_within_tolerance"])
                self.assertEqual(s["stages"]["synthesize"]["status"], "not_requested")
                self.assertTrue((root / record["path"] / "component_data.npz").exists())
                self.assertTrue(record["identity_verified"])
            build_report(root, root / "report")
            text = (root / "report/summary.md").read_text()
            self.assertIn("isolated component", text.lower())
            self.assertIn("Component output error", text)
            self.assertNotIn("Numerical baseline accepted", text)

    def test_vendor_failure_keeps_next_component_and_records_raw_diagnostics(self):
        from fpga_bench.components import run_components

        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = self.fixture(d)
            cfg["stages"]["synthesize"] = True
            calls = []

            def vendor(path, cfg, stage):
                calls.append(path)
                (Path(path) / "synthesize.log").write_text(
                    "ERROR: isolated fixture failure\n"
                )
                if len(calls) == 1:
                    raise RuntimeError("vendor failed")

            with patch("fpga_bench.components.run_hls_stage", side_effect=vendor):
                result = run_components(cfg)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                [r["status"] for r in result["runs"]], ["failed", "passed"]
            )

    def test_checkpoint_data_and_valid_component_required_before_running(self):
        from fpga_bench.components import run_components

        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = self.fixture(d)
            cfg["training"]["enabled"] = True
            with self.assertRaisesRegex(ValueError, "training.enabled=false"):
                run_components(cfg)
            cfg["training"]["enabled"] = False
            cfg["model"]["checkpoint"] = None
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                run_components(cfg)
            cfg["model"]["checkpoint"] = str(Path(d) / "checkpoint")
            cfg["data"]["path"] = None
            with self.assertRaisesRegex(ValueError, "data.path"):
                run_components(cfg)
            cfg["data"]["path"] = str(Path(d) / "data.npz")
            cfg["components"]["names"] = ["not_a_component"]
            with self.assertRaisesRegex(ValueError, "Unknown component"):
                run_components(cfg)

    def test_full_encoder_overrides_are_scoped_to_the_extracted_component(self):
        from fpga_bench.components import run_components

        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = self.fixture(d)
            cfg["components"]["names"] = ["track_dense_1"]
            cfg["hls"]["layer_overrides"] = {
                "gamma": {"ReuseFactor": 8},
                "dense_1": {"ReuseFactor": 2},
            }
            result = run_components(cfg)
            self.assertEqual(result["status"], "passed")
            root = Path(result["study_dir"]) / result["runs"][0]["path"]
            s = json.loads((root / "summary.json").read_text())
            self.assertEqual(s["component"]["ignored_layer_overrides"], ["gamma"])
            self.assertEqual(set(s["requested_hls"]["layer_overrides"]), {"dense_1"})
            emitted = json.loads((root / "hls/emitted_precisions.json").read_text())
            dense = next(l for l in emitted if l["name"] == "dense_1")
            self.assertEqual(dense["reuse_factor"], 2)

    def test_failed_numerical_acceptance_blocks_vendor_synthesis(self):
        from fpga_bench.components import run_components

        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = self.fixture(d)
            cfg["components"]["names"] = ["track_dense_1"]
            cfg["metrics"].update(atol=0.0, rtol=0.0)
            cfg["stages"]["synthesize"] = True
            with patch(
                "fpga_bench.components.run_hls_stage",
                side_effect=AssertionError("must not synthesize inaccurate component"),
            ) as vendor:
                result = run_components(cfg)
            self.assertEqual(vendor.call_count, 0)
            root = Path(result["study_dir"]) / result["runs"][0]["path"]
            s = json.loads((root / "summary.json").read_text())
            self.assertFalse(s["quality_within_tolerance"])
            self.assertEqual(s["stages"]["synthesize"]["status"], "blocked")

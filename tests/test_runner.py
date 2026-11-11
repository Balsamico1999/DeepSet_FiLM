import tempfile
import unittest
from pathlib import Path
from fpga_bench.config import load_config
from fpga_bench.runner import run


class RunnerTests(unittest.TestCase):
    def test_graph_reference_run_is_isolated_and_reloadable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "c.yaml").write_text(
                "project_root: .\nrun: {output_dir: runs, name: test}\nmodel: {kind: graphnet, max_tracks: 3, n_features: 3, n_context: 2, hidden: 4, latent_dim: 2}\ndata: {samples: {train: 4, validation: 4, calibration: 4, test: 4}}\ntraining: {epochs: 1, batch_size: 4}\nstages: {compare: false, profile: false}\n"
            )
            cfg = load_config(p / "c.yaml")
            first = run(cfg)
            self.assertEqual(first["status"], "passed")
            self.assertTrue(
                (Path(first["run_dir"]) / "calibration_predictions.npz").exists()
            )
            self.assertTrue((Path(first["run_dir"]) / "checkpoint/model.pt").exists())
            cfg["model"]["checkpoint"] = str(Path(first["run_dir"]) / "checkpoint")
            cfg["training"]["enabled"] = False
            second = run(cfg)
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertEqual(second["status"], "passed")
            cfg["stages"]["compare"] = True
            third = run(cfg)
            self.assertEqual(third["stages"]["convert"]["status"], "unsupported")
            self.assertEqual(third["status"], "failed")
            self.assertIsNone(third["hardware"]["hls_estimate"]["latency_cycles_max"])

    def test_failed_model_run_still_saves_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "c.yaml").write_text(
                "project_root: .\nrun: {output_dir: runs}\nmodel: {checkpoint: absent}\n"
            )
            result = run(load_config(p / "c.yaml"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stages"]["model"]["status"], "failed")
            self.assertTrue((Path(result["run_dir"]) / "summary.json").exists())

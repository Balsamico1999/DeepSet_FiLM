import tempfile
import unittest
from pathlib import Path
import json
from fpga_bench import study


class StudyRunnerTests(unittest.TestCase):
    def config(self, root, axes="hls.reuse_factor: [1, 4]"):
        p = Path(root) / "c.yaml"
        p.write_text(
            'project_root: .\nrun: {output_dir: runs}\nmodel: {max_tracks: 4,n_features: 3,n_context: 2,phi_dim: 4,rho_dim: 3}\ndata: {mode: structured,samples: {train: 16,validation: 8,calibration: 8,test: 8}}\ntraining: {epochs: 1,batch_size: 8,decoder_hidden: 4}\nhls: {precision: "fixed<24,10>",strategy: Resource}\nstages: {compare: false,profile: false}\nsweep:\n  '
            + axes
            + "\n"
        )
        return p

    def test_study_freezes_weights_and_data_and_uses_validation(self):
        with tempfile.TemporaryDirectory() as d:
            result = study.run_study(self.config(d))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["runs"]), 2)
            root = Path(result["study_dir"])
            identities = []
            for point in result["runs"]:
                summary = json.loads(
                    (root / point["path"] / "summary.json").read_text()
                )
                identities.append(
                    (summary["weights_sha256"], summary["dataset_sha256"])
                )
                self.assertEqual(
                    summary["stages"]["training"]["status"], "not_requested"
                )
                self.assertEqual(summary["evaluation_split"], "validation")
                self.assertIn("reference_reconstruction", summary)
            self.assertEqual(identities[0], identities[1])
            self.assertTrue((root / "study.json").exists())

    def test_geometry_axes_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "hardware"):
                study.run_study(self.config(d, "model.phi_dim: [4,8]"))
            self.assertFalse((Path(d) / "runs").exists())

    def test_bad_precision_point_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(d, 'hls.precision: ["invalid"]')
            result = study.run_study(p, overrides=["stages.compare=true"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["runs"][0]["status"], "failed")
            self.assertTrue((Path(result["study_dir"]) / "study.json").exists())

    def test_invalid_setting_is_preserved_and_later_points_still_run(self):
        with tempfile.TemporaryDirectory() as d:
            result = study.run_study(self.config(d, "hls.reuse_factor: [1,0,4]"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                [p["status"] for p in result["runs"]], ["passed", "failed", "passed"]
            )
            bad = Path(result["study_dir"]) / result["runs"][1]["path"] / "summary.json"
            self.assertEqual(
                json.loads(bad.read_text())["stages"]["configuration"]["status"],
                "failed",
            )

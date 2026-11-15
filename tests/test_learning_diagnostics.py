import tempfile
import unittest
from pathlib import Path
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate
from fpga_bench.models import create_adapter
from fpga_bench.diagnostics import weights_fingerprint, reconstruction_metrics


class LearningDiagnosticsTests(unittest.TestCase):
    def test_training_records_updates_and_keeps_decoder_out_of_hardware_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text(
                "model: {max_tracks: 4,n_features: 3,n_context: 2,phi_dim: 8,rho_dim: 4}\ntraining: {epochs: 3,batch_size: 8,decoder_hidden: 8}"
            )
            cfg = load_config(p)
            data = generate(
                seed=9,
                max_tracks=4,
                n_features=3,
                n_context=2,
                mode="structured",
                samples=dict(train=32, validation=16, calibration=8, test=8),
            )
            a = create_adapter(cfg, data)
            a.fit(data, cfg["training"])
            self.assertTrue(hasattr(a, "diagnostics"), "Training diagnostics missing")
            self.assertGreater(a.diagnostics["last_encoder_delta_l2"], 0)
            self.assertGreater(a.diagnostics["optimizer_steps"], 0)
            self.assertIn("constant_validation_chamfer", a.diagnostics)
            self.assertIn("rho_validation", a.diagnostics)
            output = a.predict(data["test"])
            self.assertIn("reconstruction", output)
            np.testing.assert_allclose(
                output["reconstruction"], a.decode(output["rho"])
            )
            m, loss = reconstruction_metrics(a, data["test"], output["reconstruction"])
            self.assertTrue(np.isfinite(m["mean_chamfer"]))
            before = weights_fingerprint(a)
            a.save(Path(d) / "checkpoint")
            cfg["model"]["checkpoint"] = str(Path(d) / "checkpoint")
            loaded = create_adapter(cfg, data)
            self.assertEqual(weights_fingerprint(loaded), before)
            self.assertEqual(
                loaded.diagnostics["optimizer_steps"], a.diagnostics["optimizer_steps"]
            )
            self.assertTrue((Path(d) / "checkpoint/diagnostics.npz").exists())

    def test_research_architecture_builds_with_rho_output_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text(
                "model: {max_tracks: 32,phi_dim: 256,attention_hidden_dim: 16,rho_dim: 16,latent_dim: 16,track_layers: 2}"
            )
            cfg = load_config(p)
            data = generate(
                max_tracks=32,
                samples=dict(train=4, validation=4, calibration=4, test=4),
            )
            a = create_adapter(cfg, data)
            self.assertEqual(tuple(a.model.output_shape), (None, 16))
            self.assertEqual(a.model.get_layer("gamma").units, 256)
            self.assertEqual(a.model.get_layer("attention_hidden").units, 16)

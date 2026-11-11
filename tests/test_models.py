import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate
from fpga_bench.models import create_adapter, lower_broadcasts


class ModelTests(unittest.TestCase):
    def config(self, root, kind):
        p = Path(root) / "c.yaml"
        p.write_text(
            "model: {kind: "
            + kind
            + ", max_tracks: 4, n_features: 3, n_context: 2, phi_dim: 4, rho_dim: 3, hidden: 4, latent_dim: 2}\ntraining: {epochs: 1, batch_size: 4, decoder_hidden: 4}\n"
        )
        return load_config(p)

    def data(self):
        return generate(
            seed=2,
            max_tracks=4,
            n_features=3,
            n_context=2,
            samples=dict(train=8, validation=4, calibration=4, test=4),
        )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow optional")
    def test_deepset_train_reload_and_lowering(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.config(root, "deepset_film")
            data = self.data()
            adapter = create_adapter(cfg, data)
            history = adapter.fit(data, cfg["training"])
            self.assertTrue(np.isfinite(history[-1]["validation_loss"]))
            expected = adapter.predict(data["test"])["rho"]
            path = Path(root) / "checkpoint"
            adapter.save(path)
            cfg["model"]["checkpoint"] = str(path)
            loaded = create_adapter(cfg, data)
            np.testing.assert_allclose(expected, loaded.predict(data["test"])["rho"])
            lowered, rewrites = lower_broadcasts(adapter.model)
            actual = lowered(adapter.inputs(data["test"]), training=False).numpy()
            self.assertTrue(rewrites)
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
            empty = {k: np.array(v, copy=True) for k, v in data["test"].items()}
            empty["mask"][:] = 0
            empty["tracks"][:] = 0
            np.testing.assert_allclose(
                adapter.predict(empty)["rho"],
                lowered(adapter.inputs(empty)).numpy(),
                atol=1e-6,
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_graph_train_reload_without_root(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.config(root, "graphnet")
            data = self.data()
            adapter = create_adapter(cfg, data)
            history = adapter.fit(data, cfg["training"])
            self.assertTrue(np.isfinite(history[-1]["validation_loss"]))
            expected = adapter.predict(data["test"])
            path = Path(root) / "graph"
            adapter.save(path)
            cfg["model"]["checkpoint"] = str(path)
            loaded = create_adapter(cfg, data)
            np.testing.assert_allclose(
                expected["reconstruction"],
                loaded.predict(data["test"])["reconstruction"],
            )

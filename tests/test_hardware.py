import tempfile
import unittest
from pathlib import Path
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate
from fpga_bench.models import create_adapter
from fpga_bench.hardware import (
    convert_adapter,
    run_command,
    implementation_tcl,
    profile_model,
    validate_cosim_outputs,
)


class HardwareTests(unittest.TestCase):
    def test_full_deepset_compiled_c(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "c.yaml").write_text(
                'model: {max_tracks: 4, n_features: 3, n_context: 2, phi_dim: 4, rho_dim: 3}\ntraining: {epochs: 1, batch_size: 4, decoder_hidden: 4}\nhls: {precision: "fixed<24,10>"}\n'
            )
            cfg = load_config(p / "c.yaml")
            data = generate(
                seed=3,
                max_tracks=4,
                n_features=3,
                n_context=2,
                samples=dict(train=8, validation=4, calibration=4, test=4),
            )
            adapter = create_adapter(cfg, data)
            adapter.fit(data, cfg["training"])
            hls, lowered, rewrites = convert_adapter(
                adapter, cfg, p / "hls", data["test"]
            )
            hls.compile()
            actual = np.asarray(hls.predict(adapter.inputs(data["test"])))
            np.testing.assert_allclose(
                actual, adapter.predict(data["test"])["rho"], rtol=0.02, atol=0.01
            )
            self.assertTrue(rewrites)
            result = profile_model(adapter, lowered, hls, data["test"], p / "profiles")
            self.assertGreaterEqual(result["range_figures"], 3)
            self.assertTrue(result["matched_layer_names"])

    def test_vendor_command_failure_and_timeout(self):
        import sys

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                run_command(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    d,
                    Path(d) / "failure.log",
                    5,
                )
            with self.assertRaises(TimeoutError):
                run_command(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    d,
                    Path(d) / "timeout.log",
                    0.05,
                )
        script = implementation_tcl(False, 5)
        self.assertIn("out_of_context", script)
        self.assertIn("create_clock", script)
        self.assertNotIn("route_design", script)
        self.assertIn("route_design", implementation_tcl(True, 5))

    def test_cosim_checks_transactions_not_only_vendor_status(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "tb_data").mkdir()
            golden = np.array([[1.0, 2.0], [3.0, 4.0]])
            np.save(p / "testbench_outputs.npy", golden)
            results = p / "tb_data/rtl_cosim_results.log"
            np.savetxt(results, golden)
            validate_cosim_outputs(p)
            np.savetxt(results, golden + 1)
            with self.assertRaises(AssertionError):
                validate_cosim_outputs(p)
            np.savetxt(results, golden[:1])
            with self.assertRaises(ValueError):
                validate_cosim_outputs(p)

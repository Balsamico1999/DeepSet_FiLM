import tempfile
from pathlib import Path
import unittest
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.hardware import convert_model


class ControlledReLUTests(unittest.TestCase):
    def test_controlled_and_native_compiled_outputs_match_with_partial_lanes(self):
        import tensorflow as tf

        for reuse in (4, 100):
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as d:
                p = Path(d)
                (p / "c.yaml").write_text("stages: {profile: false}\n")
                cfg = load_config(p / "c.yaml")
                cfg["hls"].update(
                    precision="fixed<28,10>", reuse_factor=reuse, strategy="Resource"
                )
                x = tf.keras.Input((3, 5), name="features")
                model = tf.keras.Model(x, tf.keras.layers.ReLU(name="relu")(x))
                values = np.linspace(-2, 2, 30, dtype="float32").reshape(2, 3, 5)
                expected = model(values).numpy()
                results = []
                for mode in ("native", "controlled"):
                    cfg["hls"]["relu_implementation"] = mode
                    hls, lowered, _ = convert_model(
                        model, cfg, p / mode, [values], expected
                    )
                    np.testing.assert_array_equal(lowered(values).numpy(), expected)
                    hls.compile()
                    results.append(np.asarray(hls.predict(values)))
                    if mode == "controlled":
                        layer = next(l for l in hls.get_layers() if l.name == "relu")
                        self.assertGreater(
                            layer.get_attr("activation_parallel_lanes"), 0
                        )
                        self.assertLessEqual(layer.get_attr("reuse_factor"), 15)
                np.testing.assert_array_equal(*results)

    def test_controlled_rejects_nonstandard_relu(self):
        import tensorflow as tf
        from fpga_bench.models import lower_broadcasts

        x = tf.keras.Input((3, 5))
        model = tf.keras.Model(x, tf.keras.layers.ReLU(max_value=6)(x))
        with self.assertRaisesRegex(ValueError, "standard ReLU"):
            lower_broadcasts(model, relu_implementation="controlled")

    def test_controlled_streaming_is_rejected(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "c.yaml").write_text("stages: {profile: false}\n")
            cfg = load_config(p / "c.yaml")
            cfg["hls"].update(relu_implementation="controlled", io_type="io_stream")
            x = tf.keras.Input((3, 5))
            model = tf.keras.Model(x, tf.keras.layers.ReLU()(x))
            values = np.zeros((1, 3, 5), dtype="float32")
            with self.assertRaisesRegex(ValueError, "io_parallel"):
                convert_model(model, cfg, p / "hls", [values], values)

    def test_prepared_testbench_has_real_pairs_and_fails_without_files(self):
        import tensorflow as tf
        from fpga_bench.hardware import prepare_testbench, validate_csim_outputs

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "c.yaml").write_text("stages: {profile: false}\n")
            cfg = load_config(p / "c.yaml")
            cfg["hls"]["precision"] = "fixed<28,10>"
            x = tf.keras.Input((3,), name="features")
            model = tf.keras.Model(x, tf.keras.layers.ReLU()(x))
            values = np.array([[-1, 0.5, 2], [3, -2, 0.25]], dtype="float32")
            hls, _, _ = convert_model(
                model, cfg, p / "hls", [values], model(values).numpy()
            )
            hls.compile()
            prepare_testbench(hls, p / "hls", [values], 2)
            a = np.loadtxt(p / "hls/tb_data/tb_input_features.dat")
            b = np.loadtxt(p / "hls/tb_data/tb_output_predictions.dat")
            np.testing.assert_allclose(a, values)
            np.testing.assert_allclose(b, np.maximum(values, 0))
            text = (p / "hls/deepset_bench_test.cpp").read_text()
            self.assertNotIn("using default input", text)
            self.assertIn("return 1;", text)
            output = p / "hls/deepset_bench_prj/solution1/csim/build/tb_data"
            output.mkdir(parents=True)
            np.savetxt(output / "csim_results.log", b)
            self.assertEqual(validate_csim_outputs(p / "hls")["transactions"], 2)
            np.savetxt(output / "csim_results.log", np.zeros_like(b))
            with self.assertRaises(AssertionError):
                validate_csim_outputs(p / "hls")

    def test_failed_new_invocation_cannot_reuse_old_csim_success(self):
        import json
        from unittest.mock import patch
        from fpga_bench.hardware import run_hls_stage

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "cfg.yaml").write_text("stages: {profile: false}\n")
            cfg = load_config(p / "cfg.yaml")
            (p / "tb_data").mkdir()
            for name in ("tb_input_features.dat", "tb_output_predictions.dat"):
                (p / "tb_data" / name).write_text("1 2\n")
            np.save(p / "testbench_outputs.npy", np.array([[1.0, 2.0]]))
            old = p / "old_prj/solution1/csim/build/tb_data"
            old.mkdir(parents=True)
            (old / "csim_results.log").write_text("1 2\n")
            (p / "csim_validation.json").write_text('{"status":"passed"}')
            with (
                patch("fpga_bench.hardware.shutil.which", return_value="/fake/vitis"),
                patch(
                    "fpga_bench.hardware.run_command",
                    side_effect=TimeoutError("new attempt stopped before csim"),
                ),
            ):
                with self.assertRaises(TimeoutError):
                    run_hls_stage(p, cfg, "synthesize")
            self.assertNotEqual(
                json.loads((p / "csim_validation.json").read_text())["status"], "passed"
            )

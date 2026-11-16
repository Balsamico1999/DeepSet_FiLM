"""Broadcast lowering must preserve values without materializing replica tensors."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fpga_bench.config import load_config
from fpga_bench.data import generate
from fpga_bench.models import create_adapter, lower_broadcasts
from fpga_bench.hardware import convert_adapter


class IndexedBroadcastTests(unittest.TestCase):
    def test_float_lowering_removes_replica_layers_and_preserves_weights(self):
        cfg = load_config("benchmark_configs/deepset_research_train.yaml")
        data = generate(
            seed=7,
            max_tracks=32,
            n_features=8,
            n_context=8,
            samples=dict(train=4, validation=2, calibration=2, test=2),
        )
        model = create_adapter(cfg, data).model
        lowered, rewrites = lower_broadcasts(model)
        self.assertFalse(any("_replica_" in layer.name for layer in lowered.layers))
        self.assertEqual(
            sum(
                layer.__class__.__name__ == "IndexedBroadcast"
                for layer in lowered.layers
            ),
            4,
        )
        self.assertEqual(lowered.count_params(), model.count_params())
        for old in model.layers:
            for a, b in zip(
                old.get_weights(), lowered.get_layer(old.name).get_weights()
            ):
                np.testing.assert_array_equal(a, b)
        x = [data["test"]["tracks"], data["test"]["mask"], data["test"]["context"]]
        np.testing.assert_allclose(
            lowered(x).numpy(), model(x).numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(len(rewrites), 4)

    def test_compiled_broadcast_handles_axes_operand_order_and_partial_lanes(self):
        import tensorflow as tf
        from fpga_bench.indexed_broadcast import register_hls
        import hls4ml

        register_hls("Vitis")
        rng = np.random.default_rng(3)
        for shape, op, reverse, reuse in [
            ((1, 5), "multiply", False, 4),
            ((3, 1), "multiply", True, 4),
            ((1, 5), "add", True, 4),
            ((1, 1), "add", False, 100),
        ]:
            with (
                self.subTest(shape=shape, op=op, reverse=reverse),
                tempfile.TemporaryDirectory() as d,
            ):
                a = tf.keras.Input((3, 5), name="matrix")
                b = tf.keras.Input(shape, name="vector")
                args = [b, a] if reverse else [a, b]
                y = (
                    tf.keras.layers.Multiply()
                    if op == "multiply"
                    else tf.keras.layers.Add()
                )(args)
                original = tf.keras.Model([a, b], y)
                model, _ = lower_broadcasts(original)
                config = hls4ml.utils.config_from_keras_model(
                    model,
                    granularity="name",
                    backend="Vitis",
                    default_precision="fixed<28,10>",
                    default_reuse_factor=reuse,
                )
                hls = hls4ml.converters.convert_from_keras_model(
                    model,
                    hls_config=config,
                    backend="Vitis",
                    io_type="io_parallel",
                    output_dir=d,
                )
                hls.compile()
                x = [
                    rng.uniform(-1, 1, (2, 3, 5)).astype("float32"),
                    rng.uniform(-1, 1, (2, *shape)).astype("float32"),
                ]
                actual = np.asarray(hls.predict(x)).reshape(2, 3, 5)
                np.testing.assert_allclose(
                    actual, original(x).numpy(), atol=2e-5, rtol=2e-5
                )
                node = next(
                    n
                    for n in hls.get_layers()
                    if "IndexedBroadcast" in type(n).__name__
                )
                self.assertLessEqual(node.get_attr("reuse_factor"), 15)
                self.assertTrue(
                    (Path(d) / "firmware/nnet_utils/nnet_indexed_broadcast.h").exists()
                )

    def test_adapter_records_structural_reduction_and_retains_legacy_switch(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(
                "benchmark_configs/deepset_smoke.yaml", ["stages.profile=false"]
            )
            data = generate(
                seed=2, samples=dict(train=4, validation=2, calibration=2, test=2)
            )
            adapter = create_adapter(cfg, data)
            hls, _, _ = convert_adapter(
                adapter, cfg, Path(d) / "indexed", data["validation"]
            )
            info = json.loads((Path(d) / "indexed/lowering.json").read_text())
            self.assertEqual(info["implementation"], "indexed")
            self.assertEqual(info["structure"]["replica_layers"], 0)
            self.assertEqual(info["structure"]["indexed_broadcasts"], 4)
            cfg["hls"]["broadcast_implementation"] = "replicated"
            old, rewrites = lower_broadcasts(adapter.model, implementation="replicated")
            self.assertTrue(any("_replica_" in layer.name for layer in old.layers))
            x = adapter.inputs(data["validation"])
            new, _ = lower_broadcasts(adapter.model)
            np.testing.assert_allclose(
                new(x).numpy(), old(x).numpy(), rtol=1e-6, atol=1e-6
            )

    def test_streaming_rejected_before_emitting_incompatible_kernel(self):
        from fpga_bench.indexed_broadcast import register_hls
        import hls4ml
        import tensorflow as tf

        register_hls("Vitis")
        a = tf.keras.Input((3, 5))
        b = tf.keras.Input((1, 5))
        model, _ = lower_broadcasts(
            tf.keras.Model([a, b], tf.keras.layers.Add()([a, b]))
        )
        with (
            tempfile.TemporaryDirectory() as d,
            self.assertRaisesRegex(ValueError, "io_parallel"),
        ):
            hls4ml.converters.convert_from_keras_model(
                model,
                backend="Vitis",
                io_type="io_stream",
                output_dir=d,
                hls_config={"Model": {"Precision": "fixed<28,10>", "ReuseFactor": 4}},
            )

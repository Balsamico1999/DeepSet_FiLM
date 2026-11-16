"""Keras-v2 and AMD hls4ml extension for allocation-free rank-two broadcasting."""

from functools import lru_cache
from pathlib import Path
import tensorflow as tf


def broadcast_shape(shapes):
    if len(shapes) != 2 or any(
        len(s) != 2 or any(d is None or int(d) < 1 for d in s) for s in shapes
    ):
        raise ValueError(
            "Indexed broadcast requires two fixed positive rank-two shapes"
        )
    target = tuple(max(a, b) for a, b in zip(*shapes))
    if any(a not in (1, b) for s in shapes for a, b in zip(s, target)):
        raise ValueError(f"Illegal broadcast shapes: {shapes}")
    return target


@tf.keras.utils.register_keras_serializable(package="fpga_bench")
class IndexedBroadcast(tf.keras.layers.Layer):
    def __init__(self, operation, **kwargs):
        super().__init__(**kwargs)
        if operation not in ("add", "multiply"):
            raise ValueError("Indexed broadcast operation must be add or multiply")
        self.operation = operation

    def build(self, input_shape):
        broadcast_shape([tuple(s[1:]) for s in input_shape])
        super().build(input_shape)

    def call(self, inputs):
        return (
            inputs[0] * inputs[1]
            if self.operation == "multiply"
            else inputs[0] + inputs[1]
        )

    def get_config(self):
        return {**super().get_config(), "operation": self.operation}


@lru_cache(maxsize=1)
def _extension_types():
    from hls4ml.model.layers import Layer
    from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

    class IndexedBroadcastOp(Layer):
        def initialize(self):
            if self.model.config.get_config_value("IOType") != "io_parallel":
                raise ValueError("IndexedBroadcast currently supports io_parallel only")
            shapes = [
                tuple(self.get_input_variable(name).shape) for name in self.inputs
            ]
            shape = broadcast_shape(shapes)
            self.set_attr("input_shapes", shapes)
            self.add_output_variable(list(shape))

    class BroadcastConfig(LayerConfigTemplate):
        def __init__(self):
            super().__init__(IndexedBroadcastOp)

        def format(self, node):
            (ar, ac), (br, bc) = node.get_attr("input_shapes")
            rows, cols = broadcast_shape([(ar, ac), (br, bc)])
            n = rows * cols
            requested = int(node.get_attr("reuse_factor"))
            if requested < 1:
                raise ValueError("Indexed broadcast reuse must be positive")
            lanes = max(1, (n + requested - 1) // requested)
            steps = (n + lanes - 1) // lanes
            node.set_attr("reuse_factor", steps)
            node.set_attr("broadcast_parallel_lanes", lanes)
            values = dict(
                ar=ar,
                ac=ac,
                br=br,
                bc=bc,
                rows=rows,
                cols=cols,
                lanes=lanes,
                steps=steps,
            )
            fields = "\n".join(
                f"    static const unsigned {k} = {v};" for k, v in values.items()
            )
            multiply = "true" if node.get_attr("operation") == "multiply" else "false"
            return f"struct config{node.index} {{\n{fields}\n    static const bool multiply = {multiply};\n}};\n"

    class BroadcastFunction(FunctionCallTemplate):
        def __init__(self):
            super().__init__(
                IndexedBroadcastOp,
                include_header=["nnet_utils/nnet_indexed_broadcast.h"],
            )

        def format(self, node):
            a, b = [node.get_input_variable(name) for name in node.inputs]
            y = node.get_output_variable()
            return f"nnet::indexed_broadcast<{a.type.name}, {b.type.name}, {y.type.name}, config{node.index}>({a.name}, {b.name}, {y.name});"

    return IndexedBroadcastOp, BroadcastConfig, BroadcastFunction


_registered = set()
_parser_registered = False


def register_hls(backend_name):
    """Register once per backend; do not alter native hls4ml layer handlers."""
    global _parser_registered
    if backend_name not in ("Vitis", "Vivado"):
        raise ValueError("Indexed broadcast supports Vitis and Vivado backends only")
    if backend_name in _registered:
        return
    import hls4ml
    from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer

    ir, config, function = _extension_types()
    if not _parser_registered:

        def parse(keras_layer, input_names, input_shapes, data_reader):
            layer = parse_default_keras_layer(keras_layer, input_names)
            layer["operation"] = keras_layer["config"]["operation"]
            shape = broadcast_shape([tuple(s[1:]) for s in input_shapes])
            return layer, [input_shapes[0][0], *shape]

        hls4ml.converters.register_keras_v2_layer_handler("IndexedBroadcast", parse)
        hls4ml.model.layers.register_layer("IndexedBroadcast", ir)
        _parser_registered = True
    backend = hls4ml.backends.get_backend(backend_name)
    backend.register_template(config)
    backend.register_template(function)
    backend.register_source(
        (Path(__file__).parent / "hls/nnet_indexed_broadcast.h").resolve()
    )
    _registered.add(backend_name)

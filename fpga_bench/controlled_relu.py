"""Standard ReLU with explicit lane sharing; AMD io_parallel only."""

from functools import lru_cache
from pathlib import Path
import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="fpga_bench")
class ControlledReLU(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs)


@lru_cache(maxsize=1)
def extension_types():
    from hls4ml.model.layers import Layer
    from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

    class ControlledReLUOp(Layer):
        def initialize(self):
            if self.model.config.get_config_value("IOType") != "io_parallel":
                raise ValueError("ControlledReLU currently supports io_parallel only")
            shape = list(self.get_input_variable().shape)
            if not shape or any(not isinstance(x, int) or x < 1 for x in shape):
                raise ValueError("ControlledReLU requires a fixed positive shape")
            self.add_output_variable(shape)

    class ReLUConfig(LayerConfigTemplate):
        def __init__(self):
            super().__init__(ControlledReLUOp)

        def format(self, node):
            n = math.prod(node.get_input_variable().shape)
            requested = int(node.get_attr("reuse_factor"))
            if requested < 1:
                raise ValueError("ReLU reuse must be positive")
            lanes = max(1, (n + requested - 1) // requested)
            steps = (n + lanes - 1) // lanes
            node.set_attr("reuse_factor", steps)
            node.set_attr("activation_parallel_lanes", lanes)
            return f"struct config{node.index} {{\n static const unsigned n_in={n}, lanes={lanes}, steps={steps};\n}};\n"

    class ReLUFunction(FunctionCallTemplate):
        def __init__(self):
            super().__init__(
                ControlledReLUOp, include_header=["nnet_utils/nnet_controlled_relu.h"]
            )

        def format(self, node):
            x, y = node.get_input_variable(), node.get_output_variable()
            return f"nnet::controlled_relu<{x.type.name}, {y.type.name}, config{node.index}>({x.name}, {y.name});"

    return ControlledReLUOp, ReLUConfig, ReLUFunction


_registered = set()
_parser_registered = False


def register_hls(backend_name):
    global _parser_registered
    if backend_name not in ("Vitis", "Vivado"):
        raise ValueError("ControlledReLU supports Vitis/Vivado only")
    if backend_name in _registered:
        return
    import hls4ml
    from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer

    ir, config, function = extension_types()
    if not _parser_registered:

        def parse(layer, input_names, input_shapes, data_reader):
            return parse_default_keras_layer(layer, input_names), list(input_shapes[0])

        hls4ml.converters.register_keras_v2_layer_handler("ControlledReLU", parse)
        hls4ml.model.layers.register_layer("ControlledReLU", ir)
        _parser_registered = True
    backend = hls4ml.backends.get_backend(backend_name)
    backend.register_template(config)
    backend.register_template(function)
    backend.register_source(
        (Path(__file__).parent / "hls/nnet_controlled_relu.h").resolve()
    )
    _registered.add(backend_name)

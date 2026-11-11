"""Check installation without loading ROOT data or running a training script.

Default: Keras-2 profile. --hls also compiles a tiny QKeras model to C++.
This is an environment smoke check, not validation of the full DeepSet/GraphNet.
"""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-root', action='store_true', help='Check conversion only; repo data scripts still need ROOT')
    parser.add_argument('--graph', action='store_true', help='Check optional PyTorch GraphNet imports and scatter_reduce')
    parser.add_argument('--hls', action='store_true', help='Run tiny QKeras-to-HLS C++ simulation; no AMD tool required')
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    import qkeras
    import hls4ml
    import tf2onnx
    import onnx
    import onnxruntime as ort
    import yaml
    import umap
    from hls4ml.model.profiling import numerical
    from qkeras import QActivation, QDense
    for package in ['tensorflow', 'keras', 'qkeras', 'hls4ml', 'quantizers', 'numpy', 'onnx', 'onnxruntime', 'tf2onnx']:
        print(f'{package}: {metadata.version(package)}')
    tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    if not args.skip_root:
        import ROOT
        assert hasattr(ROOT, 'RDataFrame')
        assert int(ROOT.RDataFrame(3).Count().GetValue()) == 3
        print('PyROOT and RDataFrame: OK', ROOT.gROOT.GetVersion())
    if args.graph:
        import torch
        from GraphNet import GraphAutoEncoder
        pooled = torch.zeros(2).scatter_reduce(0, torch.tensor([0, 1]), torch.tensor([1., 2.]), reduce='sum')
        np.testing.assert_allclose(pooled.numpy(), [1., 2.])
        print('GraphNet import and scatter_reduce: OK', torch.__version__)
    from AutoEncoder import build_deepset_film, build_qkeras_deepset_film
    inputs = [np.zeros((2, 4, 8), np.float32), np.ones((2, 4, 1), np.float32), np.zeros((2, 8), np.float32)]
    model = build_deepset_film(4, 8, 8, phi_dim=8, attention_hidden_dim=4, rho_dim=4, vae_output=False)
    assert model(inputs).shape == (2, 4)
    quantized_model = build_qkeras_deepset_film(4, 8, 8)
    assert all(output.shape == (2, 8) for output in quantized_model(inputs))
    print('Repository float/quantized model construction and forward passes: OK')
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='environment-check-') as tmp:
        tmp = Path(tmp)
        x = tf.keras.Input((4,), name='input_features')
        dense = tf.keras.layers.Dense(2, name='output_features')
        tiny = tf.keras.Model(x, dense(x))
        dense.set_weights([np.full((4, 2), 0.125, np.float32), np.zeros(2, np.float32)])
        values = np.arange(12, dtype=np.float32).reshape(3, 4) / 16
        signature = (tf.TensorSpec((None, 4), tf.float32, name='input_features'),)
        proto, _ = tf2onnx.convert.from_keras(tiny, input_signature=signature, opset=17, output_path=str(tmp / 'tiny.onnx'))
        onnx.checker.check_model(proto)
        session = ort.InferenceSession(str(tmp / 'tiny.onnx'), providers=['CPUExecutionProvider'])
        result = session.run(None, {'input_features': values})[0]
        np.testing.assert_allclose(result, tiny(values).numpy(), rtol=1e-5, atol=1e-6)
        print('Tiny Keras -> ONNX -> ONNX Runtime parity: OK')
        if args.hls:
            if shutil.which('g++') is None:
                raise RuntimeError('g++ is required for the hls4ml compiled-C check')
            x = tf.keras.Input((4,), name='q_input')
            z = QActivation('quantized_bits(12,2,1)', name='q_input_cast')(x)
            qdense = QDense(2, kernel_quantizer='quantized_bits(8,1,1,alpha=1)', bias_quantizer='quantized_bits(8,1,1,alpha=1)', name='q_dense')
            z = qdense(z)
            z = QActivation('quantized_bits(12,4,1)', name='q_output_cast')(z)
            qt = tf.keras.Model(x, z)
            qdense.set_weights([np.full((4, 2), 0.125, np.float32), np.zeros(2, np.float32)])
            config = hls4ml.utils.config_from_keras_model(qt, granularity='name', backend='Vitis')
            hls = hls4ml.converters.convert_from_keras_model(qt, hls_config=config, backend='Vitis',
                    output_dir=str(tmp / 'hls'), part='xcu250-figd2104-2L-e', clock_period=5, io_type='io_parallel')
            hls.compile()
            actual = np.asarray(hls.predict(np.ascontiguousarray(values)))
            np.testing.assert_allclose(actual, qt(values).numpy(), rtol=0, atol=1e-3)
            print('Tiny QKeras -> Vitis-backend compiled-C parity: OK')
    for command in ['g++', 'make', 'dot', 'latex', 'dvipng', 'vivado', 'vitis_hls', 'vitis-run']:
        print(f'{command}: {shutil.which(command) or "not on PATH"}')
    print('Requested checks passed. Full model conversion and FPGA synthesis are separate checks.')


if __name__ == '__main__':
    main()

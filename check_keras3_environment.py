"""Small Keras-3/QKeras-v3 checks; no ROOT sample or AMD compiler needed.

--hls compiles a tiny quantized model, not the full repository inference graph.
Run in requirements-keras3-experimental.txt's separate Python 3.11 environment.
"""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hls', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    import keras
    import qkeras
    import hls4ml
    from qkeras import QDense, QActivation
    from AutoEncoder import build_deepset_film, build_qkeras_deepset_film
    if keras.backend.backend() != 'tensorflow':
        raise RuntimeError('Set KERAS_BACKEND=tensorflow before importing Keras')
    for package in ['tensorflow', 'keras', 'qkeras-v3', 'hls4ml', 'quantizers']:
        print(f'{package}: {metadata.version(package)}')
    inputs = [np.zeros((2, 4, 8), np.float32), np.ones((2, 4, 1), np.float32), np.zeros((2, 8), np.float32)]
    model = build_deepset_film(4, 8, 8, phi_dim=8, attention_hidden_dim=4, rho_dim=4, vae_output=False)
    assert model(inputs).shape == (2, 4)
    quant = build_qkeras_deepset_film(4, 8, 8)
    assert all(output.shape == (2, 8) for output in quant(inputs))
    with tf.GradientTape() as tape:
        output = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(output - 1.))
    gradients = tape.gradient(loss, model.trainable_weights)
    keras.optimizers.Adam().apply_gradients(zip(gradients, model.trainable_weights))
    print('Repository builders, forward passes and modern-Adam gradient update: OK')
    if args.hls:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='keras3-check-') as tmp:
            x = keras.Input((4,), name='q_input')
            z = QActivation('quantized_bits(12,2,1)', name='q_input_cast')(x)
            dense = QDense(2, kernel_quantizer='quantized_bits(8,1,1,alpha=1)',
                           bias_quantizer='quantized_bits(8,1,1,alpha=1)', name='q_dense')
            z = dense(z)
            z = QActivation('quantized_bits(12,4,1)', name='q_output_cast')(z)
            tiny = keras.Model(x, z)
            dense.set_weights([np.full((4, 2), .125, np.float32), np.zeros(2, np.float32)])
            config = hls4ml.utils.config_from_keras_model(tiny, granularity='name', backend='Vitis')
            hls = hls4ml.converters.convert_from_keras_model(tiny, hls_config=config, backend='Vitis',
                    part='xcu250-figd2104-2L-e', output_dir=str(Path(tmp) / 'hls'), io_type='io_parallel')
            hls.compile()
            values = np.arange(12, dtype=np.float32).reshape(3, 4) / 16
            expected = tiny(values).numpy()
            actual = np.asarray(hls.predict(values))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
            print('Tiny QKeras-v3 compiled-C maximum absolute error:', np.max(np.abs(actual - expected)))
    print('Requested checks passed. Training scripts still require optimizer/export migration.')


if __name__ == '__main__':
    main()

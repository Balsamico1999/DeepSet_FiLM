# DeepSet_FiLM

## GraphNet ONNX export

`train_GraphNet.py` saves the best checkpoint and `graphnet_ae.onnx` in the output
directory (default: `graphnet_run/`). The ONNX model accepts normalized, batched
graph tensors and returns the normalized track reconstruction and graph latent.
Its metadata includes the model configuration, normalization statistics, and
feature names. Training loads the exported model with ONNX Runtime and checks its
outputs against PyTorch (`rtol=1e-3`, `atol=1e-4`) to allow for small floating-point
differences between runtimes.

For inference, prepare and normalize a graph batch as during training, then load
`graphnet_ae.onnx` with `onnxruntime.InferenceSession` and provide arrays under
the named inputs `node`, `edge`, `senders`, `receivers`, `node_graph`,
`edge_graph`, `node_rank`, `glob`, `n_nodes`, `dec_senders`, `dec_receivers`,
and `dec_edge_graph`. The outputs are named `reconstruction` and `latent`.

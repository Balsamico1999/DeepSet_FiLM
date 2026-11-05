# DeepSet_FiLM

## GraphNet ONNX export

`train_GraphNet.py` saves the best checkpoint and `graphnet_ae.onnx` in the
configured `GraphNet.OUTPUT_DIR` (default: `graphnet_run/`). GraphNet training
parameters and data settings are read from the `GraphNet` section of
`config.yaml`; command-line options can override them. The ONNX model accepts
normalized, batched graph tensors and returns the normalized track reconstruction
and graph latent.
Its metadata includes the model configuration, normalization statistics, and
feature names. Training loads the exported model with ONNX Runtime and checks its
outputs against PyTorch (`rtol=1e-3`, `atol=1e-4`) to allow for small floating-point
differences between runtimes.

For inference, prepare and normalize a graph batch as during training, then load
`graphnet_ae.onnx` with `onnxruntime.InferenceSession` and provide arrays under
the named inputs `node`, `edge`, `senders`, `receivers`, `node_graph`,
`edge_graph`, `node_rank`, `glob`, `n_nodes`, `dec_senders`, `dec_receivers`,
and `dec_edge_graph`. The outputs are named `reconstruction` and `latent`.

## Comparing anomaly models

First train/export GraphNet and export the DeepSet + FiLM model as
`deepset_film.onnx`, then run `python study_model_comparison.py`. The study reads
the ROOT input, event limit, graph options, validation fraction, batch size, and
seed from the same `GraphNet` config used by training. LLP injection parameters
come from the separate top-level `LLP` section. The model paths are
`GraphNet.OUTPUT_DIR/graphnet_ae.onnx` and `deepset_film.onnx`; ROC curves,
score arrays, and AUC metrics are written under
`GraphNet.OUTPUT_DIR/model_comparison/`.

To compare another model, register an adapter in `study_model_comparison.py` that
implements `from_config(graphnet_settings, training_settings, processor)` and
`score(data)`. The adapter owns loading its model and returns one anomaly score
per nominal and injected jet.

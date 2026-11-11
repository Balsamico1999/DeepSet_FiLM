# Version 0.1 verification

Executed on 11 November 2026, Linux x86_64, Python 3.10.22. The installed legacy environment is TensorFlow 2.14.1, Keras 2.14.0, QKeras 0.9.0, hls4ml 1.3.0, NumPy 1.26.4, PyTorch 2.5.1 CPU. Dependency consistency passed using `uv pip check --python <test-environment-python>` (89 packages). The test environment was created with uv rather than conda; the supplied conda/pip commands target the same Python package profile. ROOT and AMD tools are absent.

## Executed checks

- `python -m unittest discover -s tests -v`: **16 passed**, including actual small-model training, checkpoint round trips, hls4ml C++ compilation/inference, multi-input profiling, malformed configuration/data, metrics/report fixtures and subprocess timeout/failure handling.
- `python generate_synthetic_data.py --config benchmark_configs/deepset_smoke.yaml --output benchmark_data/final-random.npz`: generated a valid fixture.
- `python -m fpga_bench run --config benchmark_configs/deepset_smoke.yaml`: **passed**, including training, reference inference, float lowering equivalence, C++ compilation/comparison and profiling. All outputs met the configured tolerance, `abs(error) <= 0.01 + 0.01*abs(reference)`.
- `python -m fpga_bench run --config benchmark_configs/graph_smoke.yaml`: **passed**, CPU training/reference only. Graph checkpoint reload and rejection of unsupported hardware conversion additionally passed in the test suite.
- `python -m fpga_bench sweep --config benchmark_configs/experiments/deepset_sweep.yaml`: **four runs completed**, with separate numerical quality failures preserved.
- Ruff undefined/unused-import checks and source formatting passed for the added Python modules/tests.
- Independent code review found a missing explicit C/RTL-output validation check; it was corrected. Regression checks reject changed values and missing transactions even when a vendor status might report success.

## Illustrative output errors

These runs use random bounded data and small models. Values qualify the software workflow, not a trained physics model or FPGA design.

| Configuration | Epochs | Rho MAE | Rho maximum absolute error | Numerical tolerance |
|---|---:|---:|---:|---|
| Smoke, fixed<24,10>, reuse 1 | 2 | 0.000267560 | 0.006044388 | Pass |
| Sweep, fixed<16,6>, reuse 1 | 1 | 0.004074822 | 0.111248255 | Fail |
| Sweep, fixed<16,6>, reuse 4 | 1 | 0.004074822 | 0.111248255 | Fail |
| Sweep, fixed<20,8>, reuse 1 | 1 | 0.001208956 | 0.030681849 | Fail |
| Sweep, fixed<20,8>, reuse 4 | 1 | 0.001208956 | 0.030681849 | Fail |

The smoke and sweep use different epoch counts; this table is not a controlled precision study. Within the sweep, training seeds/settings are fixed. Identical C++ outputs across reuse factors are expected; hardware scheduling effects require synthesis. Activation ranges and per-layer trace comparisons are retained under `examples/deepset/profiles/`.

## Evidence and limits

`examples/` contains actual summaries, metrics, histories, package/source manifests, a sweep CSV, profiles and execution logs. Absolute run paths in these historical records describe this workspace; they are not paths to use on another host. Fresh runs write your own paths. Generated vendor projects/binaries and checkpoints are intentionally excluded from these small example records; the runnable pipeline regenerates them.

No HLS synthesis, RTL co-simulation, Vivado synthesis or place-and-route was executed. Their command construction, failure handling and report parsing have software/fixture checks only. Timing/resource fields are null in these examples. These stages require qualification against the user's installed AMD release. No device is programmed.

Quantization-aware training with an equivalent QKeras/Brevitas topology, Keras 3 migration, graph HLS lowering, streaming I/O and production physics data are future work. This version uses post-training fixed-point conversion of the floating DeepSet encoder and short reconstruction training for smoke tests.

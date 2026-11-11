> Implementation status: this is the design reference. See ../BENCHMARKING.md for delivered version 0.1 scope and verification.md for executed checks; future design items are not implied to be implemented.

# Proposed synthetic-data and FPGA benchmarking framework

11 November 2026 — implementation design for the existing DeepSet_FiLM repository.

**Status:** proposed scope for approval. No framework or synthetic-data generator has been implemented yet. Existing environment requirements and smoke checks are already available. The SSH server remains inaccessible from this workspace, so implementation will be delivered as an update patch and project archive.

## 1. Intended result

A reproducible command-line research workflow that generates random, shape-valid data, constructs or loads a network, optionally trains it briefly, compares framework inference against hls4ml compiled C inference, produces numerical profiles, and optionally runs AMD tool stages to report latency, initiation interval, utilization and routed timing.

Start with DeepSet + FiLM. Include synthetic training/inference for the existing GraphNet immediately, with an adapter boundary for its later hardware conversion. Do not represent an unsupported graph conversion as a successful synthesis run.

Random inputs are functional test fixtures, not physics simulation. Training on them establishes that gradients, checkpoints, exports and comparisons work; its losses/AUC do not measure LLP sensitivity.

## 2. Approaches considered

| Approach | Benefit | Cost/limitation |
|---|---|---|
| Extend existing top-level training scripts directly | Fewer initial entry points | Inherits ROOT coupling, incompatible data contracts and output collisions; difficult to test stages independently |
| Add a small modular runner around existing model builders — recommended | Preserves reference models, bypasses ROOT for synthetic tests, provides independent stages and adapters | Requires a small new Python package and explicit data/checkpoint contracts |
| Adopt a full experiment/configuration platform immediately | Extensive orchestration and sweeps | Additional dependencies and framework conventions before model-conversion feasibility is established |

Recommendation: add an `fpga_bench` package, a small safe YAML composition layer, versioned NPZ fixtures, and per-run artifacts. Reuse `AutoEncoder.py`, `GraphNet.py` and the existing loss utilities where appropriate. Avoid importing the existing trainers, which execute work at import time. Keep real-data scripts intact.

The recommendation and defaults below are proposals, not choices already confirmed by the user.

## 3. Initial defaults

- Existing verified Keras 2 profile: Python 3.10, TensorFlow/Keras 2.14, QKeras 0.9, hls4ml 1.3 with the compatibility pins already delivered.
- Model: `deepset_film`, small smoke dimensions (8 tracks, 8 features, 8 context values, phi width 16, rho width 8). The original larger dimensions remain selectable.
- Quantization: begin with hls4ml fixed-point conversion of a floating reference (PTQ). Preserve architecture; do not substitute the existing smaller QKeras builder as an equivalent model. Equivalent QAT can be added separately after the conversion path works.
- Hardware boundary: encoder input tracks/mask/context to rho output. Host-side Mahalanobis scoring measures downstream numerical effects. Hardware results are labeled encoder-only; FPGA score logic is a later extension.
- Hardware toolchain: Vitis HLS backend plus optional Vivado implementation. Backend `Vivado` is an explicit legacy alternative for installations containing Vivado HLS.
- Illustrative FPGA part: `xcu250-figd2104-2L-e`, inherited from the earlier smoke script; configurable, not a claim that this is the user's hardware target.
- Clock constraint: 5 ns, configurable, never presented as achieved timing merely because requested.
- I/O: `io_parallel` initially; streaming is a separate feasibility/optimization configuration.
- Expensive AMD stages are off by default. Pure Python and compiled-C tests can run without Vivado/Vitis installed.
- Keras 3 remains a separate environment/adapter qualification rather than silently replacing the existing default.

## 4. Synthetic data

A command generates deterministic NPZ files using one explicit NumPy random generator and a recorded seed. No ROOT installation is required for this route.

Canonical arrays:

- `tracks`: float32 [samples, max_tracks, track_features].
- `mask`: float32 [samples, max_tracks, 1], exactly zero or one.
- `context`: float32 [samples, context_features].
- `counts`: integer [samples], equal to mask population.
- `sample_id`: stable integer identity; distinct IDs across split roles.
- Optional `label`: artificial nominal/shifted-distribution labels for pipeline checks only.

Generate valid counts and padded rows explicitly. Defaults use finite bounded random values with zero padding, rather than unconstrained extreme numbers that trivially overflow small fixed-point types. Gaussian/uniform distributions, feature scales, sample counts and multiplicity ranges are configurable. Optional boundary fixtures exercise all-valid, minimum-count and empty-set behavior where supported.

Generate independent train, validation, calibration and test splits. Fit normalization from training only, guard zero scales, and persist the transform. Generate shifted artificial examples only for a labeled toy anomaly diagnostic; do not invent classification accuracy from an arbitrary embedding.

A graph adapter turns the same valid track sets into arrays accepted by `GraphData`: node/edge/global features, local senders/receivers, counts, event/jet identities and feature names. Complete directed connectivity is the first option. Use a documented consistent ranking. Build these arrays with NumPy directly rather than importing ROOT-coupled DataProcessor.

The graph profile requires at least two real tracks by default. An empty-set DeepSet boundary fixture must not accidentally reach a graph stage that cannot accept it.

## 5. Model and checkpoint adapters

A common adapter contract covers construction, optional training, checkpoint save/load, reference prediction, inference input/output names, score calculation and declared conversion capabilities.

### DeepSet adapter

Reuse the actual shared MLP -> context FiLM -> masked value/gate -> sum -> rho topology. Train through the existing set decoder and masked Chamfer objective for a short smoke run. Decoder and encoder checkpoints are both retained so training can be resumed; only the encoder is selected for initial hardware conversion. VICReg coefficients are explicit and logged, including zero values.

Support constructing an untrained network, training a new network, or loading a previously saved checkpoint. Persist architecture, ordered feature schema, normalization, seed and checkpoint identity together. Validate loaded input/output shapes before conversion. Do not use a missing checkpoint as permission to create a fresh random model silently.

If stock hls4ml rejects the DeepSet's broadcast/mask/reduction graph, the proposed first-delivery scope includes mathematically equivalent fixed-shape lowering (explicit replication, mask expansion or reduction wiring) and a small custom hls4ml layer/kernel if necessary. Preserve trained weights, gates, nonlinearities, sum semantics and input/output contracts. Validate the rewritten floating graph against the original before evaluating fixed-point differences. Do not replace learned gating with mean pooling or present a different network as the converted reference. Successful compiled-C comparison of a small full DeepSet is an acceptance criterion; a failed conversion produces diagnostics and an explicit incomplete status rather than a fabricated success.

### GraphNet adapter

Reuse GraphAutoEncoder and its mean per-graph reconstruction loss. Run small CPU training/checkpoint/inference tests on the synthetic graph arrays. Record reconstruction and latent outputs separately. Decoder is part of inference for this model's reconstruction anomaly score.

Checkpoint loading is limited to the existing supported builders and this framework's recorded native Keras/PyTorch checkpoint contracts in the first release; arbitrary third-party model files are outside that initial promise.

Expose a capability result explaining unsupported HLS operations. The first release provides graph software coverage and an explicit hardware-conversion feasibility stage; it does not claim the current dynamic scatter/gather topology converts automatically. A faithful fixed-shape graph or custom HLS engine is a later, separately tested implementation.

## 6. Ordered stages

1. Resolve and validate configuration; save its fully merged form and provenance.
2. Generate or load versioned data and split manifests.
3. Build or load the model; optionally train; save checkpoint/history.
4. Run reference predictions on the exact evaluation fixtures and save named outputs.
5. Convert the selected inference graph with hls4ml; save emitted layer precisions/configuration.
6. Compile its C++ prediction library and run the identical fixtures.
7. Compute numerical differences and optional toy anomaly-score changes.
8. Produce hls4ml weight/activation profiles and selected layer traces.
9. Optionally run vendor C simulation, HLS synthesis and C/RTL co-simulation.
10. Optionally run Vivado RTL synthesis and place-and-route.
11. Parse reports and write summary artifacts even when a later stage fails.

Each stage has status `not_requested`, `passed`, `failed`, `blocked` or `unsupported`, plus an explanation and relevant log paths. Required failures return a nonzero exit status. Missing tools do not become fake zero-valued resource/timing results. Successful HLS estimates do not override a failed RTL co-simulation or timing-closure result.

Vendor stages run with configurable time limits and captured logs. Use fresh isolated output directories; do not reuse stale reports from another precision/clock/model run. Resume requires matching manifest hashes.

## 7. Numerical and task metrics

For every named output, save elementwise differences and aggregate MAE, RMSE, maximum absolute error, error percentiles, relative errors with an explicit denominator floor, and configured tolerance pass fraction. Report shape/dtype and nonfinite counts. Save framework and HLS predictions in NPZ so metrics can be recomputed.

For embeddings, report per-coordinate errors and optional cosine similarity with explicit zero-vector handling. Do not apply argmax and call it classification accuracy.

For the DeepSet's optional downstream score, fit one regularized covariance and mean on reference calibration embeddings, then apply those same constants to framework and HLS test outputs. Record covariance rank and method. Compare score errors, rank correlation and fixed-threshold decision disagreement. If artificial nominal/shifted labels exist, report toy ROC-AUC and efficiency/FPR at predeclared thresholds. Label these synthetic diagnostics prominently. Do not claim tail probabilities finer than the available background sample supports.

Keep frozen-calibration comparisons separate from any experiment that recalibrates using HLS outputs. Otherwise the calibration change can conceal conversion errors.

For GraphNet, compare reconstruction and latent outputs plus reconstruction-score differences when a hardware adapter becomes available. Maintain sample/node alignment explicitly.

## 8. Profiling

Use hls4ml's profiling functions to save weight/activation range plots before and after conversion/optimization. Trace selected HLS and reference layers on a bounded fixture subset. Respect optimized/fused graph differences: align only layers with a known semantic mapping, not layers at matching numeric positions.

Report profiler failures separately from core numerical results; an optional plotting problem must not prevent saved metrics/checkpoints. Run headlessly and close figures. Save actual emitted precision types, not only the requested model-wide default.

## 9. Timing and utilization evidence

| Stage | Outputs | Interpretation |
|---|---|---|
| Python/C++ execution | Host wall time | Software diagnostic, never FPGA latency |
| HLS synthesis | Estimated latency, II, clock estimate and resource counts | HLS estimates |
| RTL co-simulation | Functional pass/fail, transaction latency/II | Simulated RTL behavior on supplied vectors |
| Vivado synthesis | Mapped LUT/FF/DSP/BRAM/URAM | RTL synthesis result |
| Vivado place-and-route | Implemented utilization, WNS/TNS and available timing fields | Routed implementation under stated constraints |

For one jet per transaction, convert cycles to ns using the recorded clock and identify whether timing was met. Report theoretical throughput from top-level jet II; a track-loop II of one is not one jet per cycle. Do not describe target-period-derived throughput as achieved when routed timing fails.

Keep raw hls4ml/AMD reports and report provenance. Preserve missing fields as null with a reason. Record device capacity when available and then calculate utilization percentages; do not hardcode capacity from an unverified part string. Treat BRAM18/BRAM36 units explicitly.

Vitis is recommended for new AMD work. Vivado's later synthesis/place-and-route is a distinct stage. Its out-of-context hls4ml flow constrains internal register paths rather than full board I/O timing. No physical board throughput or electrical-power claims are made.

## 10. YAML composition and overrides

Proposed contract:

- `_base_` accepts a path or list of YAML paths relative to the declaring YAML file.
- Load bases in listed order; later bases override earlier ones; the child overrides all bases.
- Recursively merge mappings; replace scalar and list values. Lists are not silently concatenated.
- Detect include cycles and missing files and use safe YAML loading.
- CLI overrides such as `--set hls.reuse_factor=8` apply last and are parsed as typed YAML values.
- Reject unknown schema keys and incompatible types with a useful message. Allow a declared raw hls4ml layer-configuration mapping where appropriate.
- Paths for data/checkpoints/outputs resolve from an explicit project root; no unannounced dependence on the caller's working directory.
- Save resolved YAML and the complete source/override chain per run.

Illustrative configuration (design example, not an implemented configuration file):

```yaml
_base_: [../base.yaml, ../models/deepset.yaml]
run:
  name: deepset_smoke
  seed: 42
model:
  kind: deepset_film
  max_tracks: 8
  phi_dim: 16
  rho_dim: 8
training:
  enabled: true
  epochs: 2
hls:
  backend: Vitis
  part: xcu250-figd2104-2L-e
  clock_period_ns: 5.0
  io_type: io_parallel
  precision: fixed<16,6>
  reuse_factor: 4
stages:
  compare: true
  profile: true
  synthesize: false
  cosim: false
  vivado_synthesis: false
  place_and_route: false
```

Provide a small optional sweep command over dot-path axes for precision, reuse and width, with a separate run manifest and row per combination. Validate reuse factors against the selected model/backend; do not promise every integer is legal. A basic ordered sweep is sufficient initially; distributed scheduling is outside the first version.

## 11. Proposed interface and artifacts

Illustrative entry points, subject to implementation:

- `python -m fpga_bench generate --config ...`
- `python -m fpga_bench run --config ... --set training.epochs=1`
- `python -m fpga_bench sweep --config ...`
- `python -m fpga_bench report --run-dir ...`

A run directory contains resolved config, source/environment/data/model hashes, stage statuses, history, checkpoints, predictions, numeric metrics, profiles, HLS project, raw vendor reports, and summary JSON/CSV. Run names are unique and outputs are never overwritten by default. Export-ready figures can be regenerated from saved arrays.

Synthetic inputs, generated runs and large vendor artifacts are excluded from Git. Source code, templates, small test fixtures and configuration files are committed. Fix the broad `*.txt` ignore interaction for the known requirements files without exposing arbitrary data files.

## 12. Validation and first delivery

Before claiming the first version works:

- Verify YAML precedence, typed overrides, path resolution and cycle/error handling.
- Verify reproducibility, masks/counts, padding, finite arrays and disjoint split IDs.
- Run a finite-loss short DeepSet training/checkpoint reload test.
- Run a finite-loss short GraphNet training/checkpoint reload test when PyTorch is installed.
- Exercise a trained small DeepSet through compiled HLS C inference and meaningful numerical metrics.
- Exercise profiling and model-loading paths, not just random model creation.
- Parse fixture reports for absent fields, resource units, latency and timing-closure failures.
- Validate clear errors for missing vendor tools and unsupported graph hardware conversion.
- Run Vitis/Vivado stages only where installed; otherwise deliver tested invocation/report handling with an explicit vendor-verification limit.

The first delivery is an incremental patch against the environment update already applied on the user's server, plus a complete updated project archive and exact commands. No existing training/model behavior is changed without a documented reason.

## References

- [hls4ml Vitis/Vivado backends and implementation](https://fastmachinelearning.org/hls4ml/backend/vitis.html)
- [hls4ml profiling](https://fastmachinelearning.org/hls4ml/advanced/profiling.html)
- [hls4ml report APIs](https://fastmachinelearning.org/hls4ml/autodoc/hls4ml.report.html)
- Actual installed hls4ml 1.3 source signatures and report keys were inspected alongside the uploaded model/loss code.

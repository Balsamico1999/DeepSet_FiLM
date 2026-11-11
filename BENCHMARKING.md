# Synthetic training and FPGA benchmarking (version 0.1)

This adds a ROOT-free workflow around the existing DeepSet–FiLM and GraphNet models. Original model/data/training modules are unchanged. The new smoke trainer is deliberately reconstruction-only: masked Chamfer for the DeepSet set decoder and per-graph MSE for GraphNet. It does not reproduce the old scripts' augmentation/VICReg experiments.

**Synthetic inputs and shifted test labels are software fixtures, not particle-physics simulation or LLP sensitivity measurements.**

## Start

Run from `/home/a/DeepSet_FiLM` in the verified Python 3.10 / Keras 2 environment. ROOT is unnecessary for these commands. If no environment has been created yet, install/initialize Miniforge first (the `conda` command must work), then:

```bash
conda create --name deepset-bench --override-channels --channel conda-forge \
    --strict-channel-priority python=3.10 numpy=1.26.4 pip
conda activate deepset-bench
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-linux-py310.txt
# Optional for GraphNet and the full test suite:
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-graph.txt
```

Use g++/make from your OS (`build-essential` on Ubuntu/Debian). No extra Python packages were needed beyond the previously supplied environment profiles. If your environment has another name, substitute it below.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deepset-bench
python -m pip check
python -m unittest discover -s tests -v
```

The optional GraphNet path needs the CPU/CUDA PyTorch installation described in SETUP_HLS4ML.md. The tests include C++ compilation; use a host with g++ and make. TensorFlow/NumPy thread counts can be limited on small VMs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
```

Current runner/lowering/profiling are qualified for the delivered Keras-2/hls4ml-1.3 profile. Keras 3 remains a separate migration and is not silently selected by this runner.

## Generate data and run models

```bash
python generate_synthetic_data.py \
    --config benchmark_configs/deepset_smoke.yaml \
    --output benchmark_data/random.npz

python -m fpga_bench run \
    --config benchmark_configs/deepset_smoke.yaml \
    --set data.path=benchmark_data/random.npz

python -m fpga_bench run --config benchmark_configs/graph_smoke.yaml
```

`generate` refuses to overwrite an existing file unless `--overwrite` is provided. Omitting data.path during `run` generates and saves a new deterministic fixture inside that run's directory. The NPZ file uses schema version 1, requires no pickles, and stores each split's tracks, mask, context, counts, IDs and labels.

Shapes are [N,T,F] tracks, [N,T,1] masks, and [N,C] contexts. Masks are exact binary values; padded tracks are zero. Uniform bounded or clipped-normal values are configurable. Training, validation and calibration are nominal random samples; the second half of the test split receives a shift in its first feature and label 1. Sample IDs do not overlap between splits. Normalization is fitted on real training tracks and training context only, with constant-channel protection.

The data file is for the new runner, rather than a replacement ROOT file for the legacy top-level trainers. Graph inputs are derived from these arrays with directed complete edges. Tracks are ranked by feature-vector norm; this is a documented synthetic ordering, not a physics pT claim.

## Configuration composition

```yaml
_base_: [../base.yaml, ../models/deepset.yaml]
training:
  epochs: 1
hls:
  precision: fixed<20,8>
  reuse_factor: 4
```

Bases are relative to their declaring file. Later bases override earlier ones; the child overrides all bases; `--set key=value` applies last. Mappings merge recursively and lists replace. Overrides use YAML scalar/list types. Missing includes, cycles, unknown keys and invalid types are errors. Data/checkpoint/output paths are relative to project_root, which is resolved where declared, independent of your terminal directory.

`resolved.yaml` records the final configuration. `manifest.json` records source hashes, environment versions and the include/override chain. The supplied smoke model is intentionally small: 8 tracks, 8 input features, 8 context features, phi width 16 and rho width 8. Its conservative output precision is fixed<24,10>. Change dimensions/precision in a child YAML or via `--set`.

### Sweeps

```bash
python -m fpga_bench sweep \
    --config benchmark_configs/experiments/deepset_sweep.yaml
```

The example sweeps 16/20-bit precision and reuse 1/4. A sweep is ordered, not distributed, and limited to 16 combinations unless `--max-runs` is explicitly increased. Each combination has an isolated run directory. The sweep writes combined JSON/CSV and keeps failed configurations. Reuse factors can be adjusted by hls4ml to legal values; inspect emitted scheduling values/logs rather than assuming the requested integer was honored unchanged.

## What runs

1. Validate configuration; generate/load and validate data.
2. Construct a model or load its checkpoint; optionally train and save history/checkpoint.
3. Save reference test and calibration predictions.
4. For DeepSet, create an equivalent graph with explicit fixed-shape replication where hls4ml cannot handle broadcasting. Copy trained weights and retain all FiLM/gate/value/sum operations. Test floating outputs against the original before conversion.
5. Convert the lowered graph and compile C++; compare the same test inputs.
6. Save hls4ml range plots and reference/HLS activation traces.
7. Run requested vendor stages and collect their evidence.

The DeepSet hardware boundary is **raw tracks/mask/context to rho**, including its embedded normalization but excluding the training decoder, synthetic-data generation and host Mahalanobis scoring. Input transfer/board I/O is not measured. GraphNet reference inference includes its decoder because reconstruction is its score.

The existing dynamic GraphNet has software training/checkpoint support. Hardware conversion is explicitly unsupported in version 0.1: requesting it records an unsupported stage and exits nonzero. It needs a later fixed-shape/custom graph adapter; installing hls4ml or Brevitas does not supply one automatically. Pair-model training and equivalent-topology QAT are also outside version 0.1.

## Numerical results

`metrics.json` records per-output MAE, RMSE, maximum/percentile errors, relative error with a denominator floor, coordinate errors, cosine similarity and tolerance pass fraction. Nonfinite results fail the comparison instead of being silently averaged. `predictions.npz` preserves reference/HLS outputs, IDs and artificial labels.

For optional task diagnostics, one mean and regularized covariance are fitted on reference calibration embeddings. Both reference and HLS test embeddings use those same constants and threshold. Score differences, rank correlation, threshold disagreement and toy AUC/FPR/TPR are recorded. This keeps conversion effects separate from recalibration. Background count and empirical FPR resolution are included.

A stage marked `passed` means its operation completed. Numerical acceptance is a separate `quality_within_tolerance` field; successful compilation is not proof a low-precision model is accurate. `summary.json` can therefore have status passed but quality_within_tolerance false. Sweeps retain these informative low-precision results.

## Profiling

The runner uses hls4ml weight/activation range plotting, quantized-type overlays and HLS layer traces. It works around hls4ml 1.3's combined-profiler assumption that all inputs fit in one contiguous array: weight/reference plots and multi-input HLS traces are obtained separately. Layers are compared only by matching names in the lowered graph and compatible element counts. Unmatched optimized layers are not paired by numeric position.

Profiles are bounded to 16 test examples. `profiles/layer_traces.npz`, layer_errors.json and range PNGs are retained. A plotting failure does not erase numeric results; it is recorded as a separate failed stage.

## Vendor timing and resources

First complete a software/C++ smoke run. Then source your actual AMD tool settings and select the exact benchmark part:

```bash
python -m fpga_bench run \
    --config benchmark_configs/experiments/deepset_vendor.yaml \
    --set hls.part=xcu250-figd2104-2L-e
```

Co-simulation enables hls4ml C/RTL logfile validation and independently checks RTL transaction count, finiteness and values against the saved compiled-C outputs. A vendor Pass status alone is insufficient. The independent comparison permits only documented text-serialization rounding (rtol 1e-5, atol 1e-6); validation details are saved in cosim_validation.json.

The default Vitis backend supports `vitis-run` or a Vitis HLS `vitis_hls` launcher using its generated Tcl. `--set hls.backend=Vivado` requires the older `vivado_hls`, not just Vivado. The framework runs commands as subprocesses with captured logs and a configurable timeout; timed-out process groups are terminated.

Vendor stages are independently configurable. HLS synthesis is added automatically when co-simulation or Vivado implementation needs it. Co-simulation uses multiple deterministic input transactions and compiled-C golden outputs. Vivado implementation uses a separate generated out-of-context Tcl script with an explicit clock; it does not assume the installed hls4ml exposes the newer build(pnr=True) API.

| Evidence | Meaning |
|---|---|
| HLS latency/II/resources | Compiler estimates |
| RTL co-simulation | Functional/transaction simulation results |
| Vivado synthesis utilization | Mapped RTL resources |
| Routed WNS and utilization | Implementation evidence under the chosen clock and out-of-context boundary |
| Stage host_wall_seconds | Time taken on the host; never FPGA latency |

Missing fields are null, not zero. LUT/FF/DSP/BRAM/URAM counts and percentages are reported when available. BRAM18K units and Vivado BRAM tiles are kept distinct. Target-period latency/throughput is labeled as such; a failed routed clock constraint does not become an achieved-throughput claim. No electrical-power or physical-board throughput measurement is made.

Actual AMD execution is not available in the development workspace; vendor stages/report parsing require qualification on your installed tool release. Check raw logs/reports. A tool exiting without its expected report is a failure.

## Checkpoints and outputs

A run directory includes:

- resolved.yaml, manifest.json, summary.json and summary.csv;
- data.npz, history.json, reference_predictions.npz, calibration_predictions.npz, metrics.json and predictions.npz when their stages run;
- checkpoint/ with model.json, normalization arrays and native weights;
- hls/ with generated source/config, explicit-lowering description, emitted precision/scheduling, testbench data and vendor reports;
- profiles/ and per-stage logs.

Load a checkpoint created by this runner:

```bash
python -m fpga_bench run --config benchmark_configs/deepset_smoke.yaml \
    --set model.checkpoint=benchmark_runs/YOUR_RUN/checkpoint \
    --set training.enabled=false
```

The saved architecture is authoritative; input dimensions must match your data config. Missing/incompatible checkpoints fail rather than substituting random weights. Checkpoints do not preserve optimizer state: training from loaded weights creates a new optimizer. Arbitrary external ONNX/.keras/PyTorch checkpoints need a future explicit import adapter. Existing runs are never resumed/overwritten; loading weights creates a fresh run.

Regenerate the CSV or print saved results with `python -m fpga_bench report --run-dir ...`.

## Code map

| Module | Responsibility |
|---|---|
| config.py | YAML composition, validation and path resolution |
| data.py | Random fixtures, validation, NPZ schema and normalization |
| models.py | Existing-model adapters, smoke training, checkpoint loading, broadcast lowering |
| metrics.py | Output errors and frozen-calibration toy scores |
| hardware.py | Conversion, profiling, bounded vendor commands and implementation Tcl |
| reports.py | Normalize reports while preserving evidence source/missing fields |
| runner.py | Stage dependencies, run manifests, summaries and sweeps |
| __main__.py | CLI argument handling and exit status |

Tests cover configuration precedence/errors, dataset masks/IDs/reproducibility, numerical metrics, report absence, model reload, graph training, full small-DeepSet compiled-C inference, profiling and command timeout/failure handling. They do not replace real AMD synthesis/co-simulation or physics validation.

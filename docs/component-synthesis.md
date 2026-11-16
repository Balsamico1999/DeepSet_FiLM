# Isolated component synthesis

Purpose: identify the compiler/memory costs of individual trained encoder
operations before another expensive full-network synthesis. Component estimates
are diagnostic, not additive full-encoder estimates or end-to-end performance.

Implementation plan:
1. Extract supported single-input Dense/ReLU blocks from a saved DeepSet encoder;
   capture actual floating intermediate inputs/outputs on validation examples.
2. Convert/compile/compare each block independently using the same hls4ml settings
   and HLS-stage runner. Do not train, change widths/tracks or run the decoder.
3. Run bounded sequential HLS tests; retain partial failures and continue to the
   next selected component. Require numerical acceptance before synthesis.
4. Save component boundaries/shapes/weights/input fingerprints, original encoder
   identity, predictions and one combined automatic report/review.zip.
5. Verify extraction/weight identity, actual compiled C, failure isolation,
   checkpoint/data requirements, reporting and the existing full suite.

Default target: track_relu_0, with a 600-second HLS timeout. This is the wide
32x256 activation implicated by the design-size report. Additional targets are
selectable, rather than automatically launching many long builds. Only Vitis HLS
synthesis is offered by this diagnostic command; profiling, co-simulation,
physical mapping/routing and streaming are separate tests.

## Available boundaries

| Name | Extracted operation | Research-sized input -> output |
|---|---|---|
| track_relu_0 | First track ReLU | 32x256 -> 32x256 |
| track_relu_1 | Second track ReLU | 32x256 -> 32x256 |
| track_dense_0 | First learned track Dense, before ReLU | 32x8 -> 32x256 |
| track_dense_1 | Second learned track Dense, before ReLU | 32x256 -> 32x256 |
| track_block_0 | First track Dense and ReLU | 32x8 -> 32x256 |
| track_block_1 | Second track Dense and ReLU | 32x256 -> 32x256 |
| track_phi | Track normalization and complete track stem, before FiLM | 32x8 -> 32x256 |
| context_gamma | Context-to-gamma projection | 16 -> 256 |
| context_beta | Context-to-beta projection | 16 -> 256 |

Actual shapes and layer names are read from the restored model. Hidden-layer
indices are zero-based. A shape is per example; the sample batch is omitted.
The second wide ReLU has the same operation/shape as the first, so the default
starts with only the first rather than automatically repeating similar builds.
The context projections are exposed because Dense-resource functions were also
substantial in the latest compiler-size report; this is not a claim those source
functions were conclusively mapped to one particular projection.

## How it works

`components.py` loads your saved checkpoint/data once and never calls training or
the reconstruction decoder. It discovers the linear track stem and supports the
explicit cuts above. A TensorFlow probe obtains the exact input and output
activations at the cut for the first selected validation examples. The selected
layers are cloned with their existing configuration and weights. Their floating
output must match the original-model probe within 1e-6 absolute/relative tolerance.

`hardware.convert_model` is the common conversion entry point shared by normal
full-encoder runs and component runs. Each component has its own project directory,
compiled-C simulation, numerical comparison and optional HLS synthesis. Reuse
legality is resolved exactly as for the full model. A component with failed
numerical acceptance does not proceed to HLS synthesis. A vendor failure/timeout
is recorded and does not prevent the next selected component from running.

`hls.layer_overrides` are scoped to layers present in each component. Overrides
for unrelated source-encoder layers are recorded as ignored in component.json;
they do not cause a false unknown-layer failure. Each point retains its effective
resolved YAML and emitted precision/reuse settings.

Saved evidence includes component.json (source layers, shapes, parameter count,
input/reference/weight fingerprints, sample IDs and source checkpoint/model),
component_data.npz (captured inputs/reference outputs), predictions.npz, stage
logs, raw vendor reports and summary.json. The group study.json records each
component's relative path and source-weight identity check. The automatic report
uses **component_output** metrics, not rho, reconstruction or anomaly metrics.
It includes one combined review.zip with bounded vendor tails, sampled-memory
summaries and any available compiler design-size reports.

The default captures 16 validation examples. This affects software simulation
work, not the per-example 32-track hardware shape. Every slot is included, even
when a particular example has fewer valid tracks. Captured floating activations
are quantized at the component boundary; they are not promised to match the
upstream fixed-point activations of the integrated full network bit-for-bit.

## First test on the user's machine

Apply the accompanying patch after the indexed-broadcast update. No new Python
dependencies are required. Activate the existing swap and Xilinx environment:

```bash
cd ~/DeepSet_FiLM
swapon --show
# Only if /swap-vivado is absent from that list:
sudo swapon /swap-vivado

source ~/Xilinx/Vitis_HLS/2024.1/settings64.sh
conda activate deepset-bench
TRAIN_RUN=benchmark_runs/deepset_research_train_20261115T133211_11ae354f

python -m fpga_bench components \
    --config benchmark_configs/deepset_components.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz"
```

This runs only track_relu_0 and gives its Vitis HLS subprocess at most 600 seconds.
Extraction/C++ compilation/reporting add host time outside that limit. Progress
is printed at the start/end of each stage. Vivado mapping/routing, co-simulation,
profiling and training remain disabled. Upload the printed report/review.zip.
No full encoder conversion/synthesis is requested by this command.

For a software-only check first, add `--set stages.synthesize=false`.
For a selected subsequent component, repeat the command with, for example:

```bash
--set 'components.names=[track_dense_1]'
```

Or select several sequential diagnostics (600 seconds EACH, not a group limit):

```bash
--set 'components.names=[track_relu_0,track_dense_0,track_dense_1,context_gamma]'
```

Use the same two checkpoint/data overrides in each command. The maximum selection
is 12 distinct components. Captured sample count is configurable from 1 to 128
via components.samples. HLS timeout is hls.timeout_seconds. Existing report flags
--no-pdf, --no-report and --json work on the components command. Existing groups
can be re-reported with `fpga_bench report --study-dir GROUP --overwrite`.

## Interpretation

A small isolated component that times out identifies an expensive standalone
compilation path worth investigating. A successful component does not prove its
integrated network will synthesize: compiler interactions, interfaces, scheduling
and storage differ. Do not add isolated LUT/DSP counts or latency to predict the
full encoder. Keep raw reports and compare the same target part, clock, precision,
strategy, resolved reuse and source weights.

No AMD tool is available in the implementation workspace. Actual extraction,
compiled-C inference and reporting were executed, while vendor-failure continuation
is verified using a controlled test double. No hardware utilisation/latency or
component synthesis success is claimed here.

## Executed verification

Final full suite: **56 tests passed** in 224.834 seconds. Tests include actual
compiled-C component inference, extraction/weight identity, no-training enforcement,
checkpoint/data requirements, full-encoder override scoping, numerical gating of
vendor stages, failed-component continuation, old full-encoder conversion,
calibration tracing and real pdflatex compilation. Independent review found no
actionable issues in this increment.

A real CLI software-only group exercised four components of the trained research-
sized encoder, using 16 validation examples at precision28,10 / accumulator36,12,
requested reuse128. All passed numerical acceptance and source-weight identity.
The checkpoint is the earlier reduced-sample verification checkpoint (512 training
examples), not the user's separately trained weights, which were not uploaded.

| Component | Parameters | Maximum absolute output error |
|---|---:|---:|
| track_relu_0 | 0 | 0.000003814 |
| track_dense_0 | 2,304 | 0.000013538 |
| track_dense_1 | 65,792 | 0.000055432 |
| context_gamma | 4,352 | 0.000018790 |

The combined report compiled to a nine-page PDF and its first page was rendered
and inspected. No Vitis synthesis was available in the implementation workspace.

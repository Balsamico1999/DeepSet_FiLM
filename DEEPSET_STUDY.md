# Floating DeepSet encoder study

This increment keeps the qualified Python 3.10 / TensorFlow-Keras 2.14 / hls4ml 1.3 environment. It studies a trained floating model converted to fixed point. It does not add QKeras/Brevitas QAT, GraphNet hardware support or change the original model builders.

## Architecture and boundaries

The research profile follows the supplied track architecture:

| Research setting | Benchmark setting | Value |
|---|---|---:|
| MAX_TRACKS | model.max_tracks | 32 |
| PHI_DIM | model.phi_dim | 256 |
| ATTENTION_HIDDEN_DIM | model.attention_hidden_dim | 16 |
| RHO_DIM | model.rho_dim | 16 |
| N_HIDDEN_LAYERS | model.track_layers | 2 |
| LATENT_DIM | model.latent_dim | 16, inactive with vae_output=False |
| MAX_TRACK_PAIRS | Deferred pair model | 16, not an input limit for this encoder |

Synthetic tracks have eight features; context has eight features. These dimensions match the original eight track columns and four event plus four jet columns, but generated values do not represent real physics features. The existing builder retains its fixed width-16 context hidden layer.

The hardware boundary is raw tracks/mask/context through embedded normalization, FiLM, gating and pooling to rho. The reconstruction decoder and anomaly scoring run on the host and are excluded from FPGA resource/latency counts. Decoder size/training hyperparameters are benchmark choices, not additional encoder layers.

Keep `deepset_smoke.yaml` for fast checks. Its 8-track/phi16 encoder is much smaller than the research model. No full-width C++ or AMD synthesis is performed by automated tests: the research builder shape is tested, and real compiled-C tests use small networks.

## Reports are automatic

CLI `run` and `study` now finish with readable numerical results and generate a
report automatically. Open `report/report.html` or `report/report.pdf`. To share
results for review, upload just **`report/review.zip`**; it includes the readable
report and compact evidence, without checkpoints, arrays or vendor logs.

Use `--no-pdf` if LaTeX is unavailable, `--no-report` to skip report creation, or
`--json` for machine-readable completion. The `report` command also generates
these outputs by default for existing runs; add `--overwrite` if a report exists.
See [docs/automatic-reporting.md](docs/automatic-reporting.md).

## Setup

Activate `deepset-bench` as before. No additional Python dependencies are required beyond the existing default lock (optional torch is required for the old GraphNet tests). Host `g++` and `make` are needed for compiled C. On Ubuntu/Debian, PDF report generation additionally needs:

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended lmodern
```

The reporter uses pdflatex without shell escape, saves its log, and fails clearly if the compiler or required packages are missing. Markdown, TeX sources and figures are retained on compilation failure. HTML remains readable when PDF compilation fails. It never substitutes a different PDF renderer and calls that compiled LaTeX.

For HLS estimates, source Vitis HLS. Source Vivado too only when you request mapped synthesis/implementation. All study profiles disable Vivado synthesis/routing.

## 1. Check the small study first

```bash
python -m unittest discover -s tests -v
python -m fpga_bench study \
    --config benchmark_configs/experiments/deepset_study_smoke.yaml
```

This trains one small floating model and host decoder, then uses the exact same weights, normalization and arrays for four compiled-C precision/reuse points. It does not run AMD tools. The CLI prints the study directory.

```bash
python -m fpga_bench report \
    --study-dir benchmark_runs/STUDY_DIRECTORY \
    --output benchmark_runs/STUDY_DIRECTORY/report \
    --compile-latex
```

Replace STUDY_DIRECTORY with the printed name. Report destinations must be empty unless `--overwrite` is explicit. Overwrite replaces known report outputs, including removing an old PDF before rebuilding; it does not recursively delete unrelated files.

## 2. Train the research encoder

```bash
python -m fpga_bench run \
    --config benchmark_configs/deepset_research_train.yaml
```

This profile uses structured synthetic data: three shared latent factors generate correlated bounded tracks/context with noise. The same seeded projection is used across disjoint splits. Independent random mode remains available via `--set data.mode=iid`.

Default split sizes are train 4096, validation 512, calibration 512, test 1024. Training uses eight epochs, batch 64 and learning rate 0.0001. These sample/training choices can be overridden independently of architecture. Training is reconstruction-only masked Chamfer, with the original set decoder. It is not the VICReg/augmentation objective from legacy trainers.

Record the printed run directory, for example:

```bash
TRAIN_RUN=benchmark_runs/YOUR_TRAINING_RUN
python -m fpga_bench report --run-dir "$TRAIN_RUN" \
    --output "$TRAIN_RUN/report" --compile-latex
```

Inspect `checkpoint/training_diagnostics.json`, `history.json` and the report. Diagnostics include initial validation loss, selected best epoch, constant-mean reconstruction baseline, optimizer-step count, weight-change norms, embedding variance and valid-track gate activity. A best epoch of zero is retained and explicitly warned about; it means training did not improve validation. A collapsed embedding or a reconstruction result worse than the constant baseline is a warning, not hidden by a successful conversion status.

The selected checkpoint stores encoder and decoder weights, normalization, history, numeric diagnostics and arrays used for plots. Optimizer state is not saved. Loading an older v0.1 checkpoint still works, but unavailable training diagnostics are not invented.

## 3. Profile calibration data and compare outputs

```bash
python -m fpga_bench run \
    --config benchmark_configs/deepset_research_profile.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz" \
    --set training.enabled=false \
    --set evaluation.split=validation
```

This uses compiled C, not FPGA synthesis. The profile includes at most 128 calibration examples; change profiling.max_samples if needed. profile_info.json records role and sample IDs. ranges.json stores numeric reference/HLS ranges and representable limits when available. Empirical coverage on calibration inputs is not a guaranteed overflow bound.

Profiling figures include native hls4ml plots and matched-layer error traces. Constant one-valued weights are not actual zero weights; use the numeric statistics and zero fractions rather than interpreting every native profiler message as a failure.

Reconstruction comparison sends reference and HLS embeddings through the same frozen floating decoder. It records absolute/signed/relative changes in masked Chamfer, separately from embedding MAE/RMSE/tolerance and frozen-calibration host scores.

The default research Resource strategy/reuse128 is a starting candidate, not a proven fit or optimum. Research-width compilation can be substantially more expensive than smoke runs.

## 4. Run a controlled Vitis HLS study

Start with a small subset, retaining the same checkpoint and data:

```bash
python -m fpga_bench study \
    --config benchmark_configs/experiments/deepset_research_study.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz" \
    --set training.enabled=false \
    --set 'sweep={hls.precision: ["fixed<24,10>"], hls.reuse_factor: [128,256]}'
```

Add `--set stages.synthesize=false` for numerical screening without HLS synthesis. The complete supplied grid has three precisions (24/20/16 total bits with integer width held at 10) and four requested reuse factors (32/64/128/256), for 12 points. It is a screening grid, not a claim these formats are appropriate after profiling. Layer overrides can implement mixed precision.

`study` always screens validation data. It trains/loads once into a parent run, saves its checkpoint and dataset, then disables training in every child. Semantic array/weight fingerprints must match across points. Allowed axes are hardware precision, accumulator precision, reuse, strategy and layer overrides. Architecture/data/training axes are rejected; those require separate trained models. Failed points remain in study.json and the report. A point can execute successfully while failing numerical tolerance.

Reuse is applied at layer granularity and explicit overrides take precedence. Legal Dense reuse is resolved before conversion. In hls4ml 1.3, rank-3 Dense layers become pointwise Conv1D; the parallel Resource kernel requires reuse <= input channels. Hence a global request128 does not mean every layer uses128. Read:

- hls/requested_hls_config.yaml — requested name-level settings;
- hls/resolved_hls_config.yaml and reuse_adjustments.json — legal pre-conversion settings;
- hls/emitted_precisions.json — emitted layer output types, strategies and effective reuse.

The input, normalization and copy/sum layers are retained. No claim is made that the broadcast-lowering representation is optimal; investigate generated hardware overhead before scaling further.

## 5. Produce the study report

```bash
python -m fpga_bench report --study-dir benchmark_runs/YOUR_STUDY \
    --output benchmark_runs/YOUR_STUDY/report --compile-latex
```

Outputs include results.csv, report.json, summary.md, tables/*.tex, report.tex, report.pdf, latex-build.log and PNG/PDF figures. Available evidence drives the plots: missing HLS/implementation results remain missing rather than zeros.

Figures cover training/baseline, learned weights, embedding activity, encoder errors/scatter/CDF, reconstruction change, precision/reuse heatmaps, hardware estimates, resource-quality trade-offs, build time, calibration ranges and matched-layer errors. ROC curves appear only when artificial test labels contain both classes; validation samples are nominal, so a validation study cannot claim an anomaly ROC result. Effective reuse sets are reported separately from the requested global x-axis. Accumulator precision, strategy and layer overrides remain in tables and partition plots so different arithmetic settings are not silently merged. Missing summaries and identity mismatches are explicit report diagnostics.

HLS timing/resources are estimates. ns/throughput at the target clock are conditional values. Host wall time is not FPGA latency. Do not mix columns from HLS estimates and routed reports or interpret an absent implementation result as zero utilization.

## 6. Evaluate selected settings on held-out test inputs

After selecting precision/reuse using validation/calibration, run the chosen candidate with the same checkpoint/data and `evaluation.split=test`. For example, start from deepset_research_profile.yaml, disable stages.profile, and supply the selected hls.precision/reuse_factor/layer_overrides. This preserves the encoder-only boundary. It does not automatically add HLS/Vivado stages.

Do not use the test split to choose the settings and then report it as untouched test performance. Different training seeds/models should be separate studies; do not average different hardware configurations and call that seed uncertainty.

## Optional physical validation and host resource measurements

The optional Vivado script now loads clock XDC before synthesis and defaults to one thread. hls.clock_source can specify the intended external clock-buffer site. Without it, the out-of-context clock assumption remains incomplete; a positive setup slack alone is not full board-level timing signoff. Inspect hold checks, unconstrained paths and complete reports. No board is programmed.

Vendor subprocesses save *.resources.json beside logs: sampled Linux process-group PSS/RSS, process count, system available RAM and swap use. PSS is proportional shared-memory accounting; RSS sums can double-count shared pages. Sampling can miss peaks, processes outside the group are not covered, and system swap includes other applications. These are measurements with stated scope, not a calculated minimum-RAM requirement. Failed main processes trigger process-group cleanup to avoid orphan Vivado workers.

Each run/study is isolated. General vendor-stage resumption and importing manual retry results into an old summary are not implemented. Preserve raw reports/checkpoints. No vendor synthesis was available in the implementation workspace; the user must qualify those stages against the installed AMD release.

Executed verification and its limits are recorded in [docs/study-verification.md](docs/study-verification.md).

## Indexed broadcast conversion

The default is now `hls.broadcast_implementation=indexed`. Direct indexed
broadcast kernels replace four fixed-copy Dense layers and four broadcast
transposes. Learned weights, the network architecture, activation boundaries,
normalization and pooling remain as before. The existing pooling transpose is
retained. This is a hardware implementation change, not quantization-aware
training or a streaming conversion.

Use `benchmark_configs/deepset_indexed_check.yaml` with your saved checkpoint,
saved data, training disabled and validation evaluation for the first numerical
check. Only after that passes, request `--set stages.synthesize=true`. Reports
include the selected implementation and conversion graph counts. The graph
counts do not predict FPGA resource use or HLS compiler instruction counts.

The original conversion is selectable with
`--set hls.broadcast_implementation=replicated`. A controlled software comparison
is supplied in `benchmark_configs/experiments/deepset_broadcast_comparison.yaml`.
It freezes checkpoint/data once and compares both implementations without vendor
synthesis by default. They remain separate report/plot variants.

See [docs/indexed-broadcast.md](docs/indexed-broadcast.md) for equations, reuse
semantics, test evidence, limitations and commands. The extension currently
supports Vitis/Vivado `io_parallel` only; it rejects streaming explicitly.
Vendor log tails, compact sampled-memory statistics and an available design-size
report are now included automatically in `report/review.zip`.

## Isolated component diagnostics

Use `python -m fpga_bench components --config benchmark_configs/deepset_components.yaml`
with the saved checkpoint and data overrides. It starts with only the first wide
ReLU and a 600-second HLS timeout. Components use real captured floating intermediate
activations and restored weights, with no training or decoder execution. Numerical
comparison must pass before vendor synthesis. Other cuts, sequential selections,
software-only checks and report interpretation are documented in
[docs/component-synthesis.md](docs/component-synthesis.md). Component results are
not additive full-encoder resource/timing estimates. No vendor execution was
available in the implementation workspace.

## Controlled ReLU experiment

The default ReLU remains native. The profile
`benchmark_configs/deepset_components_controlled_relu.yaml` selects one wide
ReLU with explicit reuse-controlled lanes and the same600-second HLS cap. See
[docs/controlled-relu.md](docs/controlled-relu.md) for the comparison command and
implementation. Component and full-encoder vendor testbenches now require real
paired inputs/compiled-C references and have no zero-input fallback. Vendor
C-simulation output validation is retained in review.zip when available.

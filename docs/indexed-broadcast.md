# Indexed broadcast conversion — design and qualification

Scope approved on 2026-11-16: first optimize the hardware conversion, preserving
the trained encoder, weights and architecture. Streaming, quantized training,
smaller models and pooling redesign are separate experiments.

Approach: replace only unequal-shape rank-two Add/Multiply operations in the
conversion graph with an IndexedBroadcast layer. It computes elementwise output
by indexing singleton dimensions at zero. The HLS extension uses a pipelined
outer loop and a bounded unrolled lane loop, controlled by layer reuse. It does
not allocate replicated operands or transpose them. Keep the existing graph
lowering under hls.broadcast_implementation=replicated for controlled comparison;
the new default is indexed. Support Vitis/Vivado io_parallel; reject other IOs
instead of silently generating incompatible code.

Implementation sequence:
- Regression tests first: exact floating equivalence and learned-weight identity,
  direct C++ broadcasting on both axes/both operand orders/partial lane groups,
  old-mode selection, unsupported streaming rejection.
- Add Keras/IR/parser/templates/header through the public hls4ml extension API.
- Select lowering via typed YAML setting and record graph counts/effective lanes.
- Run full software suite and a research-width checkpoint comparison. AMD tools
  are unavailable here: FPGA runtime, memory, utilisation and latency improvements
  must be measured on the user's host, not inferred from removed graph nodes.
- Include bounded vendor log tails and memory summaries in review.zip so the next
  synthesis attempt is reviewable from one file.

## What changed and why

The original Keras encoder has 31 layers. The legacy hardware-conversion graph
has 39: it inserts four frozen all-ones Dense copy layers and four transposes to
expand operands to equal shapes. The indexed graph has 31 layers: the original
four unequal-shape merges are replaced in place, without those eight auxiliary
layers. Its transpose count falls from five to one. The remaining transpose is
the original track pooling representation and is intentionally unchanged.

For input X[T,F], context gamma[1,F], beta[1,F], mask M[T,1], and attention
A[T,1], the replacement computes:

- FiLM multiplication: Y[t,f] = X[t,f] * gamma[0,f].
- FiLM addition: Z[t,f] = Y[t,f] + beta[0,f].
- Masking: masked[t,f] = Z[t,f] * M[t,0].
- Attention weighting: weighted[t,f] = value[t,f] * A[t,0].

Singleton dimensions are indexed at zero; other dimensions use the output row
or column. Each original multiply/add remains a separate operation and result
cast. No arithmetic fusion, pruning, retraining or normalization change is made.
Broadcasts also work when the small operand comes first. Equal-shape native
merges continue to use hls4ml's own kernels.

Files:
- `models.py`: selects indexed or replicated replay of the conversion graph.
- `indexed_broadcast.py`: serializable Keras layer, hls4ml IR/parser and backend
  configuration/function templates. Registration occurs once per backend.
- `hls/nnet_indexed_broadcast.h`: actual synthesizable C++ indexed operation.
  hls4ml copies it into each generated project's firmware/nnet_utils directory.
- `hardware.py`: registers the extension, verifies float equivalence, records
  graph counts and emitted custom-layer lane counts.
- `study.py`, `study_report.py`, `plotting.py`, `presentation.py`: retain the
  implementation as a separate comparison axis and explain graph counts.
- `vendor_diagnostics.py`: bounded log tails (64 KiB/stage), compact memory
  summaries and available design-size report (128 KiB cap) for review.zip.

## Broadcast arithmetic scheduling

The new kernel does not fully unroll every broadcast element by default. For N
output elements and requested reuse R:

```
lanes = max(1, ceil(N / R))
steps = ceil(N / lanes)
```

The lane loop is unrolled; the step loop requests pipeline II=1. Tail guards
handle N that is not divisible by the lane count. Reuse greater than N resolves
to N one-lane steps. Effective steps and parallel lanes are recorded in
hls/emitted_precisions.json. For 32*256=8192 elements and reuse128, this means
64 arithmetic lanes over 128 groups, rather than requesting 8192 parallel
operations for the broadcast. The 32*16 attention weighting uses four lanes.

These are implementation directives, not measured FPGA cycle counts or a promise
that II=1 will be achieved. This explicitly shares broadcast arithmetic whereas
the previous native merge could fully unroll it. Consequently latency/throughput
can change. Treat this as a complete conversion variant; do not attribute every
hardware difference solely to the removal of transposes. Native Dense reuse
legality/resolution is unchanged. Precision/accumulator settings remain explicit.

## How to test on the user's host

Apply the incremental patch accompanying this guide to the checkout with the
previous automatic-reporting update. No Python dependency changes are needed.
Keep the existing floating checkpoint and data. Run the numerical check first:

```bash
conda activate deepset-bench
cd ~/DeepSet_FiLM
TRAIN_RUN=benchmark_runs/deepset_research_train_20261115T133211_11ae354f

python -m fpga_bench run \
    --config benchmark_configs/deepset_indexed_check.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz" \
    --set training.enabled=false \
    --set evaluation.split=validation
```

Read the automatically generated report: expect `Broadcast conversion: indexed`,
four indexed broadcasts, zero replica auxiliaries, one transpose, and numerical
acceptance. No Xilinx executable is required for that software check. If quality
fails on your checkpoint, preserve the report and investigate before synthesis.

Then request one synthesis candidate (fresh run; no stage resumption):

```bash
source ~/Xilinx/Vitis_HLS/2024.1/settings64.sh
conda activate deepset-bench

python -m fpga_bench run \
    --config benchmark_configs/deepset_indexed_check.yaml \
    --set run.name=deepset_indexed_hls \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz" \
    --set training.enabled=false \
    --set evaluation.split=validation \
    --set stages.synthesize=true \
    --set hls.timeout_seconds=7200
```

Check `swapon --show` before synthesis so the same swap remains active. Vivado
mapping, routing and RTL co-simulation remain disabled. The 7200-second limit is
a stop condition, not an expected runtime. Upload only report/review.zip; vendor
failure excerpts and sampled memory are included automatically now.

For an optional controlled compiled-C comparison of both implementations:

```bash
python -m fpga_bench study \
    --config benchmark_configs/experiments/deepset_broadcast_comparison.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz" \
    --set training.enabled=false
```

This repeats both C++ compilations, but uses the same saved weights and inputs.
It does not synthesize either candidate by default.

## Executed evidence and limits

The actual paired software study used a trained research-width encoder with
32 tracks, eight features/context inputs, phi256, attention16, rho16, two hidden
track layers, precision28,10 / accumulator36,12 and reuse128. Its checkpoint is
from the earlier reduced-sample verification (512 training / 128 validation),
not the user's separately trained checkpoint, whose weights were not uploaded.

Both children passed numerical acceptance and data/weight fingerprint checks.
The float-lowering maximum error was zero in both modes. All saved numeric arrays,
including compiled-C embeddings and frozen-decoder reconstructions, were exactly
equal between modes. Both had rho MAE 0.0001694711 against TensorFlow. This is an
observed equality for this checkpoint/precision/data, not a general bit-exactness
guarantee for every mixed-precision configuration.

No AMD synthesis executable is available in the implementation environment.
Removed auxiliary nodes are directly verified; improved HLS memory/runtime,
utilisation, achievable initiation interval and timing are not yet established.
Other native layer unrolling and the pooling transpose remain possible costs.

Final verification: **50 tests passed** in 89.367 seconds, including actual
compiled-C broadcast/full-encoder tests, calibration tracing and pdflatex PDF
compilation. An independent review found no critical/important kernel or graph
issues; its null-memory-statistic robustness finding was reproduced with a failing
test and corrected before the final suite. The research-width comparison report
also compiled successfully to PDF. No vendor synthesis success is claimed.

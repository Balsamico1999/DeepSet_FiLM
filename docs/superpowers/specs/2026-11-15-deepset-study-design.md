# Floating DeepSet encoder study

Approved scope: the user accepted the proposed floating-model study/reporting increment and supplied Architecture MAX_TRACKS=32, MAX_TRACK_PAIRS=16, PHI_DIM=256, ATTENTION_HIDDEN_DIM=16, RHO_DIM=16, LATENT_DIM=16, N_HIDDEN_LAYERS=2. Only the track encoder is synthesized; pair, GraphNet and QAT work is deferred.

Keep the qualified Python 3.10 / Keras 2 / hls4ml 1.3 environment. Original physics and model-builder scripts remain unchanged. Preserve small smoke configs and add a documented research profile with the supplied dimensions, eight synthetic track/context features matching the eight original track and event-plus-jet input dimensions. Pair capacity is recorded as provenance, not used as encoder input count. The VAE latent width is metadata when vae_output=False.

Fix layer-level reuse/strategy propagation. Profile calibration data. Record requested and effective hardware settings, input/weight fingerprints and explicit encoder-only boundaries. Keep vendor implementation disabled for the study. Correct clock-constraint ordering for later optional implementation, without claiming board-level timing signoff.

Add an optional bounded structured synthetic distribution, retaining iid fixtures. Train the floating encoder and host decoder; record initial/final loss, mean reconstruction baseline, weight changes, embedding/gate activity and validation-selected weights. Report frozen-host-decoder reconstruction on reference and HLS embeddings, preserving task quality separately from numerical acceptance.

A study trains/loads once and evaluates Cartesian precision/reuse configurations using the same checkpoint, normalization and dataset. It is sequential, bounded, and keeps failed points. No automatic full research-width HLS build in tests. Geometry sweeps require separately trained checkpoints.

Generate reproducible scientific figures (PNG/PDF), aggregate CSV/JSON, Markdown and LaTeX tables, and a compiled report PDF when pdflatex exists. Treat missing hardware fields as missing; keep HLS estimates separate from mapped/routed results. Escape labels and preserve build logs. Refuse accidental overwrite of explicit reports unless requested.

Use current primitives rather than a framework migration. Alternatives considered: retrain per hardware point (confounds effects); add QAT now (user explicitly deferred); concentrate on full physical runs (too expensive for exploratory work). Chosen: fixed-checkpoint float study, Vitis HLS estimates, selected later validation.

# Implementation ledger
Source baseline is the previously delivered environment archive; this is an isolated extracted checkout, not the user's SSH repository.
Ruling: use native execution and one final independent review; original source files remain unchanged. Run synthetic paths without ROOT.
Ruling: hls4ml 1.3 installed code uses vitis-run and does not expose build(pnr=True); implement vendor commands and Vivado implementation Tcl explicitly rather than assuming the latest documentation matches this release.

Completed: safe YAML composition and overrides, versioned NPZ fixtures, existing-model training/checkpoint adapters, faithful explicit broadcast replication, compiled-C comparison, frozen-calibration diagnostics, multi-input weight/activation profiling, vendor subprocess stages/report normalization, sweeps and documentation.

Verification: 16 unittest checks passed; complete DeepSet C++/profiling run and graph software run passed; four precision/reuse sweep runs completed with quality flags preserved. Independent review finding about RTL output checks fixed and regression-tested. See verification.md for actual evidence and limits.

Delivery: incremental patch targets baseline commit 1316482 (previous environment update). A full source archive and example evidence are included for clean installation. Original AutoEncoder.py, GraphNet.py, DataProcessor.py and legacy training/loss files were not edited.

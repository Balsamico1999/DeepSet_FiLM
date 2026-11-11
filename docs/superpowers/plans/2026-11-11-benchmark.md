# FPGA Benchmark Implementation Plan

Goal: ROOT-free synthetic training and DeepSet hardware comparison with reproducible YAML configurations and stage-specific evidence.
Architecture: small fpga_bench package, shared NPZ contract, adapters around existing model builders, subprocess vendor stages, isolated run directories.
Tech stack: Python 3.10, NumPy, PyYAML, TensorFlow 2.14, optional torch 2.5, hls4ml 1.3; unittest tests.
Spec: ../../benchmark-design.md. User authorized implementation in this session.

## Constraints
Preserve original model/training files. Synthetic fixtures are not physics samples. Default hardware boundary encoder-only. Unsupported graph HLS is explicit. Missing vendor reports must never be zero results. All paths stable across working directories. User commands must work without ROOT.

## Review focus
Include cycles/type errors; malformed masks/caches; NaN/zero metric denominators; missing/failed vendor tools and absent reports; checkpoint shape mismatch and stale run artifacts.

## Tasks
- [x] Config/data: tests first for ordered inheritance, overrides, cycle/type errors and deterministic disjoint fixtures; implement config.py and data.py; run unittest.
- [x] Metrics/reports: tests first for known errors and report absence/resource units; implement metrics.py and reports.py; run unittest.
- [x] Adapters/conversion: integration tests first for finite training and reload plus faithful lowered DeepSet equivalence; implement models.py and hardware.py; run compiled-C and graph training checks.
- [x] Runner/docs: test CLI failure/status semantics and sweeps; implement runner.py, __main__.py, YAML profiles, documentation; run all tests, real smoke runs, final independent review; package incremental patch.

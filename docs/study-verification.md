# Verification of the floating encoder increment — 2026-11-15

Source baseline: Balsamico1999/DeepSet_FiLM, feat/fpga-benchmark,
3f7f7808f3cb749608e4e72e15199f838939ff48. The original model builders and
physics trainers are unchanged. Tests used Python 3.10.22, TensorFlow/Keras
2.14, hls4ml 1.3 and the repository Linux lock; optional CPU torch 2.5.1 was
installed to exercise existing GraphNet software tests.

## Executed checks

- Full suite: **37 tests passed**, including actual small-model compiled-C
  conversion/inference, Resource reuse factor 4, calibration traces, report
  generation and real pdflatex compilation. Vendor reports in parser/plot
  unit tests are fixtures, not new FPGA measurements.
- Four-point small study: one checkpoint and semantic dataset fingerprint,
  24/20-bit precision (integer width 10), requested reuse 1/4. All execution
  stages passed. The 24-bit points passed numerical tolerances; the 20-bit
  points failed the configured embedding tolerances, while remaining visible
  as successful executions with failed quality checks.
- Small-study mean embedding absolute errors: 0.0002378663 (24-bit),
  0.003511215 (20-bit). Absolute maxima: 0.00311041 and 0.0392848.
- Separate calibration profile: eight samples, IDs 80–87, distinct from the
  validation IDs 64–79 used by that study. Numeric ranges and traces saved.
- Research architecture software training: 32 tracks, phi 256, attention 16,
  rho 16, two track hidden layers, eight features/context inputs. Encoder
  contains 85,457 trainable parameters (85,633 total); host decoder 134,728.
  This verification used reduced splits 512/128/128/256 and eight epochs,
  rather than the profile's full default sample counts. Validation Chamfer
  fell from 3.6031783 to 2.4547920; constant-mean baseline 3.5380683.
  Encoder weight change L2 was 1.1778121; 64 optimizer steps. One rho
  coordinate had zero validation variance: improvement is not evidence that
  every embedding coordinate is useful. Synthetic reconstruction quality
  does not establish physics/anomaly-detection performance.

## Review corrections

A final independent review identified five issues, all addressed with failing
regression tests followed by a passing full suite: an invalid sweep point
aborted later points; accumulator/override variants could be merged in
reports; missing summaries could inherit a misleading passed status; a failed
LaTeX compiler could leave a partial PDF; unavailable profile bounds could be
plotted as zero. Reports now retain those settings, failures and missing values.

## Limits

No AMD executable was available here. No new Vitis HLS estimate, RTL
co-simulation, Vivado mapping or routing result is claimed. The research-width
encoder was trained in software, but not converted/compiled by hls4ml in this
verification. Qualify that larger conversion and hardware cost on the user's
host. Reuse legality can differ by lowered layer; inspect emitted settings.
The broad broadcast-copy lowering remains a possible hardware overhead.

The example archive includes relative study indexes, saved evidence and
compiled reports. Absolute paths in historical manifests/resolved configs
record the original execution location; use the supplied YAML profiles for
new runs. Generated C++ projects and shared libraries are omitted.

# Readable reporting by default

Problem: a successful CLI run previously printed execution JSON and required a
separate report command. This left numerical acceptance and task behaviour hard
to assess. The revised CLI finishes run/study commands with a plain-language
summary and automatically builds report/report.html, report/report.pdf,
report/summary.md, LaTeX sources/tables, figures and report/review.zip.

Open report.html in any browser: it embeds its plots and needs no server or
Python. Share review.zip for interpretation: it contains that HTML, a PDF when
compilation succeeds, and compact evidence in review.json. It omits checkpoints,
input/prediction arrays and vendor logs. Those raw artifacts stay in the run.

## Contract and implementation

- fpga_bench/presentation.py explains saved records for console and document use.
  It reports execution separately from numerical acceptance, actual tolerance
  counts, embedding/reconstruction errors, host decisions, training diagnostics,
  measured hardware stages, stage durations and the limits of the evidence.
- fpga_bench/study_report.py puts those explanations ahead of the existing tables
  and figures; HTML and review ZIP come from the same records.
- fpga_bench/__main__.py invokes reporting after CLI run/study completion. A study
  creates one combined report, rather than reporting each child as it executes.
  The legacy sweep command creates a report for each independent run.
- Direct Python runner.run/run_study APIs continue to return saved evidence;
  automatic presentation belongs to the command-line workflow.
- Missing/failed comparisons must never be called accepted. Missing hardware
  remains unmeasured. Report errors remain separate from immutable run results,
  are recorded in report_status.json and give a nonzero CLI exit status.
- Existing nonempty report directories require --overwrite. Automatic reports
  normally target newly created run directories.

## Commands

Existing run and study commands need no extra flags. PDF compilation uses
pdflatex by default. If LaTeX is unavailable, --no-pdf still creates the readable
HTML, Markdown, TeX and shareable review ZIP. --no-report disables automatic
report creation; --json restores machine-readable completion output.

To report an already completed run without retraining or recompiling its model:

```bash
python -m fpga_bench report --run-dir benchmark_runs/YOUR_RUN
```

Add --overwrite to refresh an existing report, or --no-pdf to skip PDF compilation.
The old --compile-latex flag remains accepted for compatibility.

## Verification

Regression tests in tests/test_readable_workflow.py exercise automatic run and
study reporting, execution/quality separation, missing quality, portable HTML
and evidence ZIP, JSON opt-out and independent report failures. They failed
before implementation. Existing compiled-C/model and LaTeX tests remain in the
full suite. Attached 24/28-bit results are used only as saved evidence for example
reports; their training or hardware stages are not rerun or newly claimed.

Final qualification: **44 tests passed**, including real compiled-C and pdflatex checks. The attached-evidence comparison PDF compiled successfully and its first page was rendered and inspected.

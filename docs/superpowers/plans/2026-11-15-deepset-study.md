# Floating DeepSet Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Fixed-checkpoint DeepSet precision/reuse studies with reconstruction diagnostics, calibrated profiling and scientific reports.
**Architecture:** Extend existing adapters and runner, add focused diagnostics/study/plotting modules, preserve existing APIs and legacy source. Train one model; freeze input and weight identities across hardware points.
**Tech Stack:** Python 3.10, TensorFlow/Keras 2.14, NumPy, hls4ml 1.3, Matplotlib, pandas, pdflatex.
**Spec:** ../specs/2026-11-15-deepset-study-design.md

## Global Constraints
Encoder-only FPGA boundary. Research architecture 32/256/16/16 with two track layers; LATENT_DIM=16 is inactive with vae_output=False; MAX_TRACK_PAIRS=16 belongs to a deferred variant. Eight track and context inputs in synthetic research profile. Keep small smoke tests. No QAT or GraphNet feature changes. No AMD runs available here; test compiled C on small models and compile real LaTeX.

## Review Focus
- Name-level configuration must override defaults correctly and preserve explicit per-layer overrides.
- Studies must never retrain between hardware points or use test data for precision selection.
- Empty/nonfinite outputs, collapsed embeddings and constant baselines must remain visible.
- Missing/failed hardware and LaTeX stages must not become zero/success records.
- Paths, escaped labels, checkpoint provenance and stale report output must remain unambiguous.

## Tasks
- [x] Task 1: configuration/hardware correctness. Write RED tests for layer reuse/strategy precedence and clock constraints. Implement configuration helper in hardware.py, validate profiling options in config.py; verify small compiled C and core tests; commit.
- [x] Task 2: learning/data/diagnostics. RED tests for seeded structured fixtures, masked Chamfer and training/checkpoint/host-decoder diagnostics. Extend data.py/models.py and add diagnostics.py; preserve legacy shapes/API; verify full tests; commit.
- [x] Task 3: fixed checkpoint study/orchestration. RED tests for identical checkpoint/data IDs across points, failure preservation and frozen task comparison. Extend runner.py and add study.py; CLI study command, research/training/study profiles; verify integration; commit.
- [x] Task 4: reports and plots. RED tests for absent hardware, Markdown/LaTeX escaping, aggregate figures, compile status and actual compiled PDF. Add plotting.py/study_report.py, extend CLI report, document commands and limits. Generate actual small-run report; full tests, fresh independent review, patch verification; commit and package.

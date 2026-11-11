"""Stage orchestration. Every run retains evidence, including partial failures."""

from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import re
import time
import traceback
import uuid
import numpy as np
import yaml
from .config import load_config, resolve_path
from .data import generate, load_dataset, save_dataset
from .metrics import differences, score_comparison
from .models import create_adapter
from .reports import collect_reports, normalize_reports, parse_utilization


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_for_config(cfg):
    m = cfg["model"]
    d = cfg["data"]
    return generate(
        seed=cfg["run"]["seed"],
        max_tracks=m["max_tracks"],
        n_features=m["n_features"],
        n_context=m["n_context"],
        **{
            k: d[k] for k in ("min_tracks", "samples", "distribution", "scale", "shift")
        },
    )


def run(cfg):
    name = re.sub("[^a-zA-Z0-9_.-]+", "_", cfg["run"]["name"]).strip(".") or "run"
    path = resolve_path(cfg, cfg["run"]["output_dir"]) / (
        name
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    path.mkdir(parents=True, exist_ok=False)
    resolved = {k: v for k, v in cfg.items() if k != "_meta"}
    resolved["project_root"] = cfg["_meta"]["project_root"]
    (path / "resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    summary = {
        "schema_version": 1,
        "run_dir": str(path.resolve()),
        "status": "running",
        "model_kind": cfg["model"]["kind"],
        "hardware_boundary": "DeepSet encoder only; host anomaly scoring excluded. GraphNet reference includes decoder.",
        "synthetic_warning": "Random bounded inputs; optional shifted labels are not physics signals.",
        "stages": {},
        "hardware": normalize_reports({}, cfg["hls"]["clock_period_ns"]),
    }
    for name_ in (
        "data",
        "model",
        "training",
        "reference",
        "convert",
        "compile",
        "compare",
        "profile",
        "synthesize",
        "cosim",
        "vivado_synthesis",
        "place_and_route",
    ):
        summary["stages"][name_] = {"status": "not_requested"}
    versions = {}
    for package in (
        "numpy",
        "tensorflow",
        "keras",
        "qkeras",
        "torch",
        "hls4ml",
        "quantizers",
        "PyYAML",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).parent.parent / n
        for n in ("AutoEncoder.py", "GraphNet.py", "VICReg_utility.py")
    ]
    write_json(
        path / "manifest.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "configuration_sources": cfg["_meta"],
            "source_sha256": {str(p.name): file_hash(p) for p in source_paths},
            "vendor_tools_executed_only_when_requested": True,
            "optimizer_resume_supported": False,
        },
    )

    def stage(name, fn, dependencies=(), requested=True):
        if not requested:
            return None
        missing = [
            dep for dep in dependencies if summary["stages"][dep]["status"] != "passed"
        ]
        if missing:
            summary["stages"][name] = {
                "status": "blocked",
                "reason": "Prerequisites failed: " + ", ".join(missing),
            }
            return None
        start = time.perf_counter()
        record = {"status": "running"}
        summary["stages"][name] = record
        try:
            with (
                (path / (name + ".log")).open("w") as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                result = fn()
            record["status"] = "passed"
            return result
        except NotImplementedError as exc:
            record.update(status="unsupported", reason=str(exc))
            return None
        except Exception as exc:
            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            with (path / (name + ".log")).open("a") as log:
                traceback.print_exc(file=log)
            return None
        finally:
            record["host_wall_seconds"] = time.perf_counter() - start
            write_json(path / "summary.json", summary)

    dataset = stage(
        "data",
        lambda: (
            load_dataset(resolve_path(cfg, cfg["data"]["path"]))
            if cfg["data"]["path"]
            else generate_for_config(cfg)
        ),
    )
    if dataset is not None:
        save_dataset(path / "data.npz", dataset)
        summary["data_sha256"] = file_hash(path / "data.npz")
    adapter = stage("model", lambda: create_adapter(cfg, dataset), ("data",))
    if adapter is not None:
        summary["effective_model"] = adapter.cfg["model"]
        stage(
            "training",
            lambda: adapter.fit(dataset, cfg["training"]),
            ("model",),
            cfg["training"]["enabled"],
        )
        try:
            adapter.save(path / "checkpoint")
            write_json(path / "history.json", adapter.history)
        except Exception as exc:
            summary["stages"]["model"].update(
                status="failed", reason=f"Checkpoint save failed: {exc}"
            )

    def reference():
        outputs = adapter.predict(dataset["test"])
        cal = adapter.predict(dataset["calibration"])
        for name_, value in {
            **outputs,
            **{"calibration_" + k: v for k, v in cal.items()},
        }.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite reference output: {name_}")
        np.savez_compressed(path / "reference_predictions.npz", **outputs)
        np.savez_compressed(path / "calibration_predictions.npz", **cal)
        if adapter.kind == "graphnet":
            from sklearn.metrics import roc_auc_score

            labels = dataset["test"]["labels"]
            summary["reference_toy_auc"] = (
                float(roc_auc_score(labels, outputs["score"]))
                if len(np.unique(labels)) == 2
                else None
            )
        return outputs, cal

    refs = stage(
        "reference",
        reference,
        ("model", "training") if cfg["training"]["enabled"] else ("model",),
    )
    s = cfg["stages"]
    vendor = any(
        s[k] for k in ("synthesize", "cosim", "vivado_synthesis", "place_and_route")
    )
    need_hls = s["compare"] or s["profile"] or vendor
    from .hardware import (
        convert_adapter,
        run_hls_stage,
        run_vivado_stage,
        profile_model,
    )

    converted = stage(
        "convert",
        lambda: convert_adapter(adapter, cfg, path / "hls", dataset["test"]),
        ("reference",),
        need_hls,
    )
    hls = lowered = None
    if converted is not None:
        hls, lowered, _ = converted
    stage("compile", lambda: hls.compile(), ("convert",), need_hls)

    def compare():
        actual = np.asarray(hls.predict(adapter.inputs(dataset["test"]))).reshape(
            len(dataset["test"]["tracks"]), -1
        )
        ref = refs[0]["rho"]
        kwargs = {k: cfg["metrics"][k] for k in ("atol", "rtol", "relative_floor")}
        metrics = {"rho": differences(ref, actual, **kwargs)}
        arrays = {
            "reference_rho": ref,
            "hls_rho": actual,
            "sample_id": dataset["test"]["ids"],
            "label": dataset["test"]["labels"],
        }
        if cfg["metrics"]["score"]:
            scores, score_arrays = score_comparison(
                refs[1]["rho"],
                ref,
                actual,
                labels=dataset["test"]["labels"],
                shrinkage=cfg["metrics"]["shrinkage"],
                background_quantile=cfg["metrics"]["background_quantile"],
                **kwargs,
            )
            metrics["host_score"] = scores
            arrays.update(score_arrays)
        write_json(path / "metrics.json", metrics)
        np.savez_compressed(path / "predictions.npz", **arrays)
        summary["metrics"] = metrics
        summary["quality_within_tolerance"] = metrics["rho"]["all_within_tolerance"]
        # A completed comparison can reveal an unacceptable precision. Do not hide that result.

    stage("compare", compare, ("compile",), s["compare"])

    def golden():
        n = min(cfg["hls"]["cosim_samples"], len(dataset["test"]["tracks"]))
        x = [v[:n] for v in adapter.inputs(dataset["test"])]
        np.save(
            path / "hls/testbench_outputs.npy",
            np.asarray(hls.predict(x)).reshape(n, -1),
        )
        hls.config.config["OutputPredictions"] = str(
            (path / "hls/testbench_outputs.npy").resolve()
        )
        hls.write()

    if vendor and summary["stages"]["compile"]["status"] == "passed":
        try:
            golden()
        except Exception as exc:
            summary["stages"]["compile"].update(
                status="failed", reason=f"Testbench creation failed: {exc}"
            )

    def profile():
        summary["profiling"] = profile_model(
            adapter, lowered, hls, dataset["test"], path / "profiles"
        )

    stage("profile", profile, ("compile",), s["profile"])
    stage(
        "synthesize",
        lambda: run_hls_stage(path / "hls", cfg, "synthesize"),
        ("compile",),
        vendor,
    )
    stage(
        "cosim",
        lambda: run_hls_stage(path / "hls", cfg, "cosim"),
        ("synthesize",),
        s["cosim"],
    )
    stage(
        "vivado_synthesis",
        lambda: run_vivado_stage(path / "hls", cfg, False),
        ("synthesize",),
        s["vivado_synthesis"] or s["place_and_route"],
    )
    stage(
        "place_and_route",
        lambda: run_vivado_stage(path / "hls", cfg, True),
        ("vivado_synthesis",),
        s["place_and_route"],
    )
    if (path / "hls").exists():
        try:
            report, raw = collect_reports(path / "hls", cfg["hls"]["clock_period_ns"])
            summary["hardware"] = report
            if (path / "hls/vivado_synth.rpt").exists():
                summary["hardware"]["vivado_synthesis"] = parse_utilization(
                    path / "hls/vivado_synth.rpt"
                )
            write_json(path / "vendor_reports_raw.json", raw)
        except Exception as exc:
            summary["report_error"] = str(exc)
    summary["status"] = (
        "failed"
        if summary.get("report_error")
        or any(
            v["status"] in ("failed", "blocked", "unsupported")
            for v in summary["stages"].values()
        )
        else "passed"
    )
    write_json(path / "summary.json", summary)
    write_summary_csv(path / "summary.csv", summary)
    return summary


def write_summary_csv(path, summary):
    row = {
        "run_dir": summary["run_dir"],
        "status": summary["status"],
        "model": summary["model_kind"],
        "quality_within_tolerance": summary.get("quality_within_tolerance"),
        "rho_mae": summary.get("metrics", {}).get("rho", {}).get("mae"),
        "rho_max_abs": summary.get("metrics", {}).get("rho", {}).get("max_abs"),
    }
    estimate = summary["hardware"]["hls_estimate"]
    row.update(
        hls_latency_cycles=estimate["latency_cycles_max"],
        hls_ii_cycles=estimate["ii_cycles_max"],
        routed_wns_ns=summary["hardware"]["implementation"]["wns_ns"],
    )
    for key, value in estimate["resources"].items():
        row["hls_estimated_" + key] = value["used"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def sweep(config_path, overrides=(), max_runs=16):
    cfg = load_config(config_path, overrides)
    axes = cfg["sweep"]
    keys = list(axes)
    if not keys:
        raise ValueError("sweep must define at least one dot-path axis")
    count = int(np.prod([len(axes[k]) for k in keys]))
    if count > max_runs:
        raise ValueError(f"Sweep has {count} runs; increase --max-runs explicitly")
    results = []
    for i, values in enumerate(itertools.product(*(axes[k] for k in keys))):
        extra = [k + "=" + json.dumps(v) for k, v in zip(keys, values)]
        child = load_config(config_path, [*overrides, *extra])
        child["run"]["name"] += f"_sweep{i:03d}"
        result = run(child)
        result["sweep_values"] = dict(zip(keys, values))
        results.append(result)
    out = resolve_path(cfg, cfg["run"]["output_dir"]) / (
        "sweep_" + uuid.uuid4().hex[:8] + ".json"
    )
    write_json(out, results)
    csv_path = out.with_suffix(".csv")
    rows = []
    for result in results:
        with (Path(result["run_dir"]) / "summary.csv").open() as f:
            row = next(csv.DictReader(f))
        row.update(result["sweep_values"])
        rows.append(row)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results, str(out)

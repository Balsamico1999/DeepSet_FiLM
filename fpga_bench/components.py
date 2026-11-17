"""Saved-encoder component diagnostics; each hardware boundary is isolated."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import yaml
from .config import resolve_path
from .data import load_dataset, dataset_fingerprint
from .diagnostics import weights_fingerprint
from .hardware import convert_model, run_hls_stage, prepare_testbench
from .metrics import differences
from .models import create_adapter
from .reports import collect_reports, normalize_reports
from .runner import write_json


def component_specs(model):
    """Discover the linear track stem and expose only explicit supported cuts."""
    import tensorflow as tf

    current = model.inputs[0]
    path = []
    while True:
        candidates = [
            l
            for l in model.layers
            if isinstance(l, (tf.keras.layers.Dense, tf.keras.layers.ReLU))
            and l.input is current
        ]
        if len(candidates) != 1:
            break
        layer = candidates[0]
        path.append(layer)
        current = layer.output
    dense = [l for l in path if isinstance(l, tf.keras.layers.Dense) and l.trainable]
    relu = [l for l in path if isinstance(l, tf.keras.layers.ReLU)]
    if not dense or len(dense) != len(relu):
        raise ValueError("Unsupported encoder track-path layout")
    specs = {"track_phi": path}
    for i, (d, r) in enumerate(zip(dense, relu)):
        specs[f"track_dense_{i}"] = [d]
        specs[f"track_relu_{i}"] = [r]
        specs[f"track_block_{i}"] = [d, r]
    for name in ("gamma", "beta"):
        specs["context_" + name] = [model.get_layer(name)]
    return specs


def array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in arrays.items():
        a = np.ascontiguousarray(value)
        digest.update(json.dumps([name, a.dtype.str, a.shape]).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def extract_component(adapter, split, name, max_samples):
    import tensorflow as tf

    specs = component_specs(adapter.model)
    if name not in specs:
        raise ValueError(f"Unknown component {name}; available: " + ", ".join(specs))
    layers = specs[name]
    n = min(max_samples, len(split["tracks"]))
    if n < 1:
        raise ValueError("Component extraction requires nonempty inputs")
    probe = tf.keras.Model(adapter.model.inputs, [layers[0].input, layers[-1].output])
    captured, expected = probe([x[:n] for x in adapter.inputs(split)], training=False)
    captured, expected = captured.numpy(), expected.numpy()
    if not np.isfinite(captured).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite component activation")
    inp = tf.keras.Input(tuple(captured.shape[1:]), name="component_input")
    output = inp
    for layer in layers:
        cloned = layer.__class__.from_config(layer.get_config())
        output = cloned(output)
        cloned.set_weights(layer.get_weights())
    component = tf.keras.Model(inp, output, name="component_" + name)
    np.testing.assert_allclose(
        component(captured, training=False).numpy(),
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Component extraction changed the floating model",
    )
    arrays = {
        "input": captured,
        "reference_output": expected,
        "sample_id": split["ids"][:n],
    }
    info = {
        "name": name,
        "source_layers": [l.name for l in layers],
        "input_shapes": [list(captured.shape[1:])],
        "output_shape": list(expected.shape[1:]),
        "parameter_count": component.count_params(),
        "samples": n,
        "sample_ids": arrays["sample_id"].tolist(),
        "input_sha256": array_fingerprint(
            {"input": captured, "sample_id": arrays["sample_id"]}
        ),
        "data_sha256": array_fingerprint(arrays),
        "weights_sha256": array_fingerprint(
            {f"weight_{i}": v for i, v in enumerate(component.get_weights())}
        ),
    }
    return component, [captured], expected, info


def run_components(config, progress=False):
    """Load once, capture independently, and continue after individual failures."""
    cfg = deepcopy(config)
    if cfg["model"]["kind"] != "deepset_film":
        raise ValueError("Component diagnostics support the DeepSet encoder only")
    if cfg["training"]["enabled"]:
        raise ValueError("Component tests require training.enabled=false")
    if not cfg["model"]["checkpoint"]:
        raise ValueError("Component tests require a saved model.checkpoint")
    if not cfg["data"]["path"]:
        raise ValueError("Component tests require the saved data.path")
    if cfg["hls"]["backend"] != "Vitis" or cfg["hls"]["io_type"] != "io_parallel":
        raise ValueError("Component diagnostics currently require Vitis io_parallel")
    if not cfg["stages"]["compare"] or any(
        cfg["stages"][s]
        for s in ("profile", "cosim", "vivado_synthesis", "place_and_route")
    ):
        raise ValueError(
            "Component tests require compare=true; profile, cosim and physical stages must be false"
        )
    dataset = load_dataset(resolve_path(cfg, cfg["data"]["path"]))
    adapter = create_adapter(cfg, dataset)
    available = component_specs(adapter.model)
    unknown = set(cfg["components"]["names"]) - set(available)
    if unknown:
        raise ValueError(
            "Unknown component: "
            + ", ".join(sorted(unknown))
            + "; available: "
            + ", ".join(available)
        )
    parent_weights = weights_fingerprint(adapter)
    parent_data = dataset_fingerprint(dataset)
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    root = resolve_path(cfg, cfg["run"]["output_dir"]) / ("components_" + stamp)
    root.mkdir(parents=True, exist_ok=False)
    resolved = {k: v for k, v in cfg.items() if k != "_meta"}
    resolved["project_root"] = cfg["_meta"]["project_root"]
    (root / "resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    versions = {}
    for package in ("numpy", "tensorflow", "keras", "hls4ml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(
        root / "manifest.json",
        {
            "packages": versions,
            "configuration_sources": cfg["_meta"],
            "checkpoint": str(resolve_path(cfg, cfg["model"]["checkpoint"])),
            "dataset_sha256": parent_data,
            "weights_sha256": parent_weights,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path(__file__).parent / "hardware.py"]
            },
        },
    )
    index = {
        "schema_version": 1,
        "study_dir": str(root.resolve()),
        "status": "running",
        "kind": "components",
        "weights_sha256": parent_weights,
        "dataset_sha256": parent_data,
        "runs": [],
    }
    write_json(root / "study.json", index)
    for i, name in enumerate(cfg["components"]["names"]):
        if progress:
            print(
                f"\nComponent {i + 1}/{len(cfg['components']['names'])}: {name}",
                flush=True,
            )
        point_cfg = deepcopy(cfg)
        model = inputs = expected = info = None
        path = root / "points" / f"{i:02d}_{name}"
        path.mkdir(parents=True, exist_ok=False)
        summary = {
            "schema_version": 1,
            "status": "running",
            "run_dir": str(path.resolve()),
            "model_kind": "deepset_component",
            "evaluation_split": cfg["evaluation"]["split"],
            "requested_hls": point_cfg["hls"],
            "weights_sha256": parent_weights,
            "dataset_sha256": parent_data,
            "component": {"name": name},
            "hardware_boundary": "Isolated encoder component; not full-encoder performance",
            "hardware": normalize_reports({}, cfg["hls"]["clock_period_ns"]),
            "stages": {
                s: {"status": "not_requested"}
                for s in (
                    "extract",
                    "reference",
                    "convert",
                    "compile",
                    "compare",
                    "testbench",
                    "synthesize",
                )
            },
        }

        def stage(label, fn, deps=(), requested=True):
            if not requested:
                return None
            if any(summary["stages"][d]["status"] != "passed" for d in deps):
                summary["stages"][label] = {
                    "status": "blocked",
                    "reason": "Component prerequisite failed",
                }
                write_json(path / "summary.json", summary)
                return None
            if progress:
                suffix = (
                    f" (timeout {cfg['hls']['timeout_seconds']}s)"
                    if label == "synthesize"
                    else ""
                )
                print(f"  {name}: {label}{suffix}", flush=True)
            start = time.perf_counter()
            entry = {"status": "running"}
            summary["stages"][label] = entry
            try:
                with (
                    (path / (label + ".log")).open("w") as log,
                    redirect_stdout(log),
                    redirect_stderr(log),
                ):
                    value = fn()
                entry["status"] = "passed"
                return value
            except Exception as exc:
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                with (path / (label + ".log")).open("a") as log:
                    traceback.print_exc(file=log)
                return None
            finally:
                entry["host_wall_seconds"] = time.perf_counter() - start
                if progress:
                    print(
                        f"  {name}: {label} {entry['status']} ({entry['host_wall_seconds']:.1f}s)"
                        + (": " + entry["reason"] if "reason" in entry else ""),
                        flush=True,
                    )
                write_json(path / "summary.json", summary)

        captured = stage(
            "extract",
            lambda: extract_component(
                adapter,
                dataset[cfg["evaluation"]["split"]],
                name,
                cfg["components"]["samples"],
            ),
        )
        if captured is not None:
            model, inputs, expected, info = captured
            selected_names = {l.name for l in model.layers}
            overrides = point_cfg["hls"]["layer_overrides"]
            info["ignored_layer_overrides"] = sorted(set(overrides) - selected_names)
            point_cfg["hls"]["layer_overrides"] = {
                k: v for k, v in overrides.items() if k in selected_names
            }
            point_resolved = {k: v for k, v in point_cfg.items() if k != "_meta"}
            point_resolved["project_root"] = cfg["_meta"]["project_root"]
            (path / "resolved.yaml").write_text(
                yaml.safe_dump(point_resolved, sort_keys=False)
            )
            info["source_checkpoint"] = str(
                resolve_path(cfg, cfg["model"]["checkpoint"])
            )
            info["source_encoder_model"] = adapter.cfg["model"]
            summary["component"] = info
            summary["component"]["capture_role"] = cfg["evaluation"]["split"]
            np.savez_compressed(
                path / "component_data.npz",
                input=inputs[0],
                reference_output=expected,
                sample_id=np.asarray(info["sample_ids"]),
            )
            write_json(path / "component.json", info)
        stage("reference", lambda: True, ("extract",))
        converted = stage(
            "convert",
            lambda: convert_model(
                model,
                point_cfg,
                path / "hls",
                inputs,
                expected,
                project_name="component_" + name,
                boundary="isolated component " + name,
            ),
            ("reference",),
        )
        hls = converted[0] if converted is not None else None
        stage("compile", lambda: hls.compile(), ("convert",))

        def compare():
            actual = np.asarray(hls.predict(inputs[0])).reshape(expected.shape)
            metric = differences(
                expected,
                actual,
                **{k: cfg["metrics"][k] for k in ("atol", "rtol", "relative_floor")},
            )
            summary["metrics"] = {"component_output": metric}
            summary["quality_within_tolerance"] = metric["all_within_tolerance"]
            write_json(path / "metrics.json", summary["metrics"])
            np.savez_compressed(
                path / "predictions.npz",
                reference_output=expected,
                hls_output=actual,
                sample_id=np.asarray(info["sample_ids"]),
            )

        stage("compare", compare, ("compile",))
        if (
            cfg["stages"]["synthesize"]
            and summary.get("quality_within_tolerance") is True
        ):
            stage(
                "testbench",
                lambda: prepare_testbench(
                    hls, path / "hls", inputs, cfg["hls"]["cosim_samples"]
                ),
                ("compare",),
            )
        if (
            cfg["stages"]["synthesize"]
            and summary.get("quality_within_tolerance") is False
        ):
            summary["stages"]["synthesize"] = {
                "status": "blocked",
                "reason": "Isolated component output tolerance failed; synthesis skipped",
            }
        else:
            stage(
                "synthesize",
                lambda: run_hls_stage(path / "hls", point_cfg, "synthesize"),
                ("compare", "testbench"),
                cfg["stages"]["synthesize"],
            )
        if (path / "hls").exists():
            try:
                summary["hardware"], raw = collect_reports(
                    path / "hls", cfg["hls"]["clock_period_ns"]
                )
                write_json(path / "vendor_reports_raw.json", raw)
            except Exception as exc:
                summary["report_error"] = f"{type(exc).__name__}: {exc}"
        verified = weights_fingerprint(adapter) == parent_weights
        if not verified:
            summary["identity_error"] = (
                "Source encoder/decoder weights changed during component extraction"
            )
        summary["status"] = (
            "passed"
            if verified
            and not summary.get("report_error")
            and all(
                v["status"] in ("passed", "not_requested")
                for v in summary["stages"].values()
            )
            else "failed"
        )
        write_json(path / "summary.json", summary)
        index["runs"].append(
            {
                "path": str(path.relative_to(root)),
                "component": name,
                "status": summary["status"],
                "identity_verified": verified,
            }
        )
        write_json(root / "study.json", index)
    index["status"] = (
        "passed" if all(r["status"] == "passed" for r in index["runs"]) else "failed"
    )
    write_json(root / "study.json", index)
    return index

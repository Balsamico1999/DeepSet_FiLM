"""Controlled hardware studies: one floating checkpoint and fixed inputs."""

from copy import deepcopy
from datetime import datetime, timezone
import itertools
import math
import json
from pathlib import Path
import uuid
from .config import load_config, resolve_path
from .runner import run, write_json

HARDWARE_AXES = {
    "hls.precision",
    "hls.accum_precision",
    "hls.reuse_factor",
    "hls.strategy",
    "hls.layer_overrides",
    "hls.broadcast_implementation",
}


def run_study(config_path, overrides=(), max_runs=16):
    cfg = load_config(config_path, overrides)
    axes = cfg["sweep"]
    if not axes or set(axes) - HARDWARE_AXES:
        raise ValueError(
            "Study axes must be hardware precision/reuse/strategy settings; geometry requires a new trained model"
        )
    count = math.prod(len(values) for values in axes.values())
    if max_runs < 1 or count > max_runs:
        raise ValueError(f"Study has {count} points; maximum is {max_runs}")
    combinations = itertools.product(*axes.values())
    if cfg["model"]["kind"] != "deepset_film":
        raise ValueError(
            "This controlled study supports the floating DeepSet encoder only"
        )
    root = resolve_path(cfg, cfg["run"]["output_dir"]) / (
        "study_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    index = {
        "schema_version": 1,
        "study_dir": str(root.resolve()),
        "status": "running",
        "evaluation_split": "validation",
        "hardware_boundary": "DeepSet encoder only; reconstruction uses a frozen host decoder",
        "axes": axes,
        "runs": [],
    }

    def save():
        write_json(root / "study.json", index)

    save()
    parent = deepcopy(cfg)
    parent["run"].update(name="training", output_dir=str(root / "training"))
    parent["evaluation"]["split"] = "validation"
    parent["stages"] = {key: False for key in cfg["stages"]}
    result = run(parent)
    index["training_run"] = str(Path(result["run_dir"]).relative_to(root))
    if result["status"] != "passed":
        index.update(status="failed", reason="Training/reference parent failed")
        save()
        return index
    index["weights_sha256"] = result["weights_sha256"]
    index["dataset_sha256"] = result["dataset_sha256"]
    checkpoint = Path(result["run_dir"]) / "checkpoint"
    data = Path(result["run_dir"]) / "data.npz"
    for i, values in enumerate(combinations):
        parameters = dict(zip(axes, values))
        try:
            extras = [
                f"{key}=" + json.dumps(value) for key, value in parameters.items()
            ]
            child = load_config(config_path, [*overrides, *extras])
            child["model"]["checkpoint"] = str(checkpoint)
            child["data"]["path"] = str(data)
            child["training"]["enabled"] = False
            child["evaluation"]["split"] = "validation"
            child["run"].update(name=f"point_{i:03d}", output_dir=str(root / "points"))
            point = run(child)
        except Exception as exc:
            from .reports import normalize_reports

            failed = root / "points" / f"point_{i:03d}_configuration_failed"
            failed.mkdir(parents=True, exist_ok=False)
            requested = deepcopy(cfg["hls"])
            for key, value in parameters.items():
                requested[key.split(".", 1)[1]] = value
            point = {
                "schema_version": 1,
                "run_dir": str(failed),
                "status": "failed",
                "evaluation_split": "validation",
                "model_kind": "deepset_film",
                "requested_hls": requested,
                "hardware": normalize_reports({}, cfg["hls"]["clock_period_ns"]),
                "stages": {
                    "configuration": {
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                },
            }
            write_json(failed / "summary.json", point)
            (failed / "configuration.log").write_text(f"{type(exc).__name__}: {exc}\n")
        record = {
            "path": str(Path(point["run_dir"]).relative_to(root)),
            "parameters": parameters,
            "status": point["status"],
        }
        verified = all(
            point.get(key) == index[key] for key in ("weights_sha256", "dataset_sha256")
        )
        record["identity_verified"] = verified
        if not verified and point["status"] == "passed":
            record.update(
                status="failed",
                identity_error="Weights or dataset identity changed or was not verified",
            )
        index["runs"].append(record)
        save()
    index["status"] = (
        "failed" if any(p["status"] != "passed" for p in index["runs"]) else "passed"
    )
    save()
    return index

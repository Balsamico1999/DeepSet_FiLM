"""Safe, ordered YAML composition with typed dot-path overrides."""

from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "run": {"name": "smoke", "seed": 42, "output_dir": "benchmark_runs"},
    "data": {
        "path": None,
        "samples": {"train": 64, "validation": 16, "calibration": 32, "test": 32},
        "min_tracks": 2,
        "distribution": "uniform",
        "scale": 0.5,
        "shift": 0.7,
    },
    "model": {
        "kind": "deepset_film",
        "max_tracks": 8,
        "n_features": 8,
        "n_context": 8,
        "phi_dim": 16,
        "attention_hidden_dim": 4,
        "rho_dim": 8,
        "track_layers": 2,
        "hidden": 16,
        "latent_dim": 8,
        "encoder_blocks": 1,
        "decoder_blocks": 1,
        "mlp_layers": 2,
        "checkpoint": None,
    },
    "training": {
        "enabled": True,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "decoder_hidden": 16,
    },
    "hls": {
        "backend": "Vitis",
        "part": "xcu250-figd2104-2L-e",
        "clock_period_ns": 5.0,
        "io_type": "io_parallel",
        "precision": "fixed<16,6>",
        "accum_precision": "fixed<32,12>",
        "reuse_factor": 1,
        "strategy": "Latency",
        "layer_overrides": {},
        "timeout_seconds": 3600,
        "cosim_samples": 4,
    },
    "stages": {
        "compare": True,
        "profile": True,
        "synthesize": False,
        "cosim": False,
        "vivado_synthesis": False,
        "place_and_route": False,
    },
    "metrics": {
        "atol": 0.01,
        "rtol": 0.01,
        "relative_floor": 1e-6,
        "score": True,
        "shrinkage": 0.01,
        "background_quantile": 0.95,
    },
    "sweep": {},
}


def merge(base, child):
    out = deepcopy(base)
    for key, value in child.items():
        out[key] = (
            merge(out[key], value)
            if isinstance(value, dict) and isinstance(out.get(key), dict)
            else deepcopy(value)
        )
    return out


def load_config(path, overrides=()):
    sources = []

    def read(p, stack):
        p = Path(p).resolve()
        if p in stack:
            raise ValueError(
                "YAML include cycle: " + " -> ".join(map(str, [*stack, p]))
            )
        raw = yaml.safe_load(p.read_text())
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{p}: expected a YAML mapping")
        bases = raw.pop("_base_", [])
        if isinstance(bases, str):
            bases = [bases]
        if not isinstance(bases, list) or not all(isinstance(x, str) for x in bases):
            raise ValueError("_base_ must be a path or list of paths")
        out = {}
        for name in bases:
            out = merge(out, read(p.parent / name, [*stack, p]))
        if "project_root" in raw:
            raw["project_root"] = str((p.parent / str(raw["project_root"])).resolve())
        sources.append(str(p))
        return merge(out, raw)

    path = Path(path).resolve()
    config = merge(DEFAULTS, read(path, []))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected key=value override: {item}")
        key, raw = item.split("=", 1)
        keys = key.split(".")
        target = config
        for k in keys[:-1]:
            if not isinstance(target.get(k), dict):
                raise ValueError(f"Unknown override path: {key}")
            target = target[k]
        if keys[0] != "sweep" and keys[-1] not in target:
            raise ValueError(f"Unknown override: {key}")
        target[keys[-1]] = yaml.safe_load(raw)
    validate(config)
    project = config.get("project_root")
    if not project:
        project = next(
            (p for p in path.parents if (p / "AutoEncoder.py").exists()), path.parent
        )
    config["_meta"] = {
        "project_root": str(Path(project).resolve()),
        "sources": sources,
        "overrides": list(overrides),
    }
    return config


def validate(c):
    unknown = set(c) - set(DEFAULTS) - {"project_root"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for section, defaults in DEFAULTS.items():
        if not isinstance(c[section], dict):
            raise ValueError(f"{section} must be a mapping")
        if section == "sweep":
            continue
        extra = set(c[section]) - set(defaults)
        if extra:
            raise ValueError(f"Unknown {section} keys: {sorted(extra)}")
        for key, default in defaults.items():
            v = c[section][key]
            if default is None:
                if v is not None and (not isinstance(v, str) or not v):
                    raise ValueError(
                        f"{section}.{key}: expected a nonempty path or null"
                    )
                continue
            valid = (
                isinstance(v, type(default))
                if not isinstance(default, float)
                else isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            if isinstance(default, int) and not isinstance(default, bool):
                valid = isinstance(v, int) and not isinstance(v, bool)
            if not valid:
                raise ValueError(f"{section}.{key}: expected {type(default).__name__}")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{section}.{key}: must be finite")
    if c["model"]["kind"] not in ("deepset_film", "graphnet"):
        raise ValueError("Unknown model kind")
    for key in (
        "max_tracks",
        "n_features",
        "n_context",
        "phi_dim",
        "attention_hidden_dim",
        "rho_dim",
        "track_layers",
        "hidden",
        "latent_dim",
        "encoder_blocks",
        "decoder_blocks",
        "mlp_layers",
    ):
        if c["model"][key] < 1:
            raise ValueError(f"model.{key} must be positive")
    t = c["training"]
    h = c["hls"]
    d = c["data"]
    m = c["metrics"]
    if c["run"]["seed"] < 0:
        raise ValueError("run.seed must be nonnegative")
    if t["decoder_hidden"] < 1:
        raise ValueError("training.decoder_hidden must be positive")
    if t["epochs"] < 1 or t["batch_size"] < 2 or t["learning_rate"] <= 0:
        raise ValueError("Training epochs/lr must be positive and batch_size >= 2")
    if h["backend"] not in ("Vitis", "Vivado"):
        raise ValueError("hls.backend must be Vitis or Vivado")
    if h["io_type"] != "io_parallel":
        raise ValueError(
            "Version 0.1 supports io_parallel only; streaming requires separate validation"
        )
    if (
        h["reuse_factor"] < 1
        or h["clock_period_ns"] <= 0
        or h["timeout_seconds"] <= 0
        or h["cosim_samples"] < 2
    ):
        raise ValueError("Invalid HLS scheduling/clock/timeout/cosim settings")
    if not 0 <= d["min_tracks"] <= c["model"]["max_tracks"]:
        raise ValueError("Invalid track count range")
    if set(d["samples"]) != {"train", "validation", "calibration", "test"} or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 2
        for v in d["samples"].values()
    ):
        raise ValueError("Each split needs an integer sample count >= 2")
    if d["distribution"] not in ("uniform", "normal") or d["scale"] <= 0:
        raise ValueError("Invalid synthetic distribution or scale")
    if (
        m["atol"] < 0
        or m["rtol"] < 0
        or m["relative_floor"] <= 0
        or not 0 < m["background_quantile"] < 1
        or not 0 < m["shrinkage"] <= 1
    ):
        raise ValueError("Invalid metric tolerances or calibration settings")
    if not isinstance(h["layer_overrides"], dict):
        raise ValueError("layer_overrides must be a mapping")
    for key, values in c["sweep"].items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key} must be a nonempty list")


def resolve_path(config, value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(config["_meta"]["project_root"]) / p

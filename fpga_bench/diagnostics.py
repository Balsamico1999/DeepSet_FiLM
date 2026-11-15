"""Finite, masked diagnostics shared by training, comparison and reports."""

import hashlib
import json
import numpy as np


def array_statistics(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Statistics need finite nonempty arrays")
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "zero_fraction": float(np.mean(a == 0)),
        "abs_percentiles": {
            str(q): float(np.percentile(np.abs(a), q)) for q in (50, 95, 99, 99.9, 100)
        },
    }


def chamfer_per_example(targets, predictions, mask, batch_size=128):
    """Match the existing TF symmetric squared Chamfer and slot-count mask."""
    x = np.asarray(targets, np.float64)
    p = np.asarray(predictions, np.float64)
    m = np.asarray(mask)[..., 0] > 0
    if (
        x.shape != p.shape
        or x.ndim != 3
        or m.shape != x.shape[:2]
        or not np.isfinite(x).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Invalid Chamfer shapes/values")
    result = []
    for start in range(0, len(x), batch_size):
        a = x[start : start + batch_size]
        b = p[start : start + batch_size]
        valid = m[start : start + batch_size]
        count = valid.sum(1)
        prediction_mask = np.arange(p.shape[1])[None, :] < count[:, None]
        distances = np.sum((b[:, :, None, :] - a[:, None, :, :]) ** 2, axis=-1)
        forward = np.min(np.where(valid[:, None, :], distances, np.inf), axis=2)
        backward = np.min(
            np.where(prediction_mask[:, :, None], distances, np.inf), axis=1
        )
        loss = (
            np.where(prediction_mask, forward, 0).sum(1)
            + np.where(valid, backward, 0).sum(1)
        ) / np.maximum(count, 1)
        result.append(loss)
    return np.concatenate(result)


def weights_fingerprint(adapter):
    digest = hashlib.sha256()
    if adapter.kind == "deepset_film":
        groups = [
            ("encoder", adapter.model.get_weights()),
            ("host_decoder", adapter.decoder.get_weights()),
        ]
    else:
        groups = [
            (
                "graph",
                [v.detach().cpu().numpy() for v in adapter.model.state_dict().values()],
            )
        ]
    for group, values in groups:
        for i, value in enumerate(values):
            a = np.ascontiguousarray(value)
            digest.update(json.dumps([group, i, a.dtype.str, a.shape]).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def normalized_targets(adapter, split):
    return (split["tracks"] - adapter.norm["track_shift"]) / adapter.norm["track_scale"]


def reconstruction_metrics(adapter, split, reconstruction):
    loss = chamfer_per_example(
        normalized_targets(adapter, split), reconstruction, split["mask"]
    )
    return {
        "mean_chamfer": float(loss.mean()),
        "median_chamfer": float(np.median(loss)),
        "p95_chamfer": float(np.percentile(loss, 95)),
        "boundary": "encoder plus frozen host decoder; decoder excluded from FPGA cost",
    }, loss

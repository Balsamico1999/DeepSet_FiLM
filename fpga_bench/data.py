"""Versioned NumPy fixtures. Random shifted samples are software tests, not LLP simulation."""

from pathlib import Path
import numpy as np

SPLITS = ("train", "validation", "calibration", "test")


def generate(
    seed=42,
    max_tracks=8,
    n_features=8,
    n_context=8,
    min_tracks=2,
    samples=None,
    distribution="uniform",
    scale=0.5,
    shift=0.7,
):
    if not 0 <= min_tracks <= max_tracks or min(max_tracks, n_features, n_context) < 1:
        raise ValueError("Invalid feature dimensions or counts")
    if (
        distribution not in ("uniform", "normal")
        or not np.isfinite([scale, shift]).all()
        or scale <= 0
    ):
        raise ValueError("Invalid distribution/scale")
    samples = samples or dict(train=64, validation=16, calibration=32, test=32)
    if set(samples) != set(SPLITS) or any(
        not isinstance(n, int) or n < 2 for n in samples.values()
    ):
        raise ValueError("Four splits with >=2 samples each are required")
    rng = np.random.default_rng(seed)
    out = {}
    offset = 0
    for split in SPLITS:
        n = samples[split]
        counts = rng.integers(min_tracks, max_tracks + 1, size=n)
        mask = (np.arange(max_tracks)[None, :] < counts[:, None])[..., None].astype(
            np.float32
        )

        def draw(shape):
            x = (
                rng.uniform(-scale, scale, shape)
                if distribution == "uniform"
                else np.clip(rng.normal(0, scale, shape), -3 * scale, 3 * scale)
            )
            return x.astype(np.float32)

        tracks = draw((n, max_tracks, n_features))
        context = draw((n, n_context))
        labels = np.zeros(n, np.int8)
        if split == "test":
            labels[n // 2 :] = 1
            tracks[n // 2 :, :, 0] += shift
        out[split] = {
            "tracks": tracks * mask,
            "mask": mask,
            "context": context,
            "counts": counts.astype(np.int32),
            "ids": np.arange(offset, offset + n, dtype=np.int64),
            "labels": labels,
        }
        offset += n
    validate_dataset(out)
    return out


def validate_dataset(data):
    if set(data) != set(SPLITS):
        raise ValueError("Dataset must contain all four split roles")
    ids = []
    dimensions = None
    for name, s in data.items():
        x = np.asarray(s["tracks"])
        mask = np.asarray(s["mask"])
        context = np.asarray(s["context"])
        n = len(x)
        if (
            n < 2
            or x.ndim != 3
            or min(x.shape[1:], default=0) < 1
            or mask.shape != (*x.shape[:2], 1)
            or context.ndim != 2
            or context.shape[1] < 1
            or len(context) != n
        ):
            raise ValueError(f"{name}: inconsistent shapes")
        shape = (x.shape[1:], context.shape[1:])
        if dimensions is not None and dimensions != shape:
            raise ValueError("Split feature dimensions differ")
        dimensions = shape
        if any(not np.isfinite(s[k]).all() for k in ("tracks", "mask", "context")):
            raise ValueError(f"{name}: nonfinite input")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must be binary")
        for k in ("counts", "ids", "labels"):
            if np.asarray(s[k]).shape != (n,):
                raise ValueError(f"{name}: invalid {k} shape")
        if not np.issubdtype(
            np.asarray(s["ids"]).dtype, np.integer
        ) or not np.issubdtype(np.asarray(s["counts"]).dtype, np.integer):
            raise ValueError("IDs/counts must be integer")
        if not np.array_equal(s["counts"], mask.sum((1, 2))):
            raise ValueError("Counts and masks disagree")
        if not np.all(x[mask[..., 0] == 0] == 0):
            raise ValueError("Padded rows must be zero")
        if not np.isin(s["labels"], [0, 1]).all():
            raise ValueError("Labels must be binary")
        ids.extend(s["ids"].tolist())
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs overlap across splits")


def save_dataset(path, data):
    validate_dataset(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{split}_{key}": v for split, s in data.items() for key, v in s.items()}
    np.savez_compressed(path, schema_version=np.array(1), **arrays)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        if int(z["schema_version"]) != 1:
            raise ValueError("Unsupported NPZ schema")
        out = {
            name: {
                k: z[f"{name}_{k}"]
                for k in ("tracks", "mask", "context", "counts", "ids", "labels")
            }
            for name in SPLITS
        }
    validate_dataset(out)
    return out


def fit_normalization(train):
    physical = train["tracks"][train["mask"][..., 0] > 0]
    if not len(physical):
        raise ValueError("Training needs at least one real track")

    def stats(x):
        low = x.min(0)
        high = x.max(0)
        return (low + high) / 2, np.where(high > low, (high - low) / 2, 1).astype(
            np.float32
        )

    shift, scale = stats(physical)
    cshift, cscale = stats(train["context"])
    return {
        "track_shift": shift,
        "track_scale": scale,
        "context_shift": cshift,
        "context_scale": cscale,
    }

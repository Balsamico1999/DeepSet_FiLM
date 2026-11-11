"""Output errors and frozen-calibration synthetic anomaly diagnostics."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def differences(reference, actual, atol=0.01, rtol=0.01, relative_floor=1e-6):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError("Output shapes must match and be nonempty")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite output cannot pass numerical comparison")
    err = np.abs(a - b)
    relative = err / np.maximum(np.abs(a), relative_floor)
    out = {
        "shape": list(a.shape),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(err.max()),
        "relative_mae": float(relative.mean()),
        "absolute_error_percentiles": {
            str(q): float(np.percentile(err, q)) for q in (50, 90, 95, 99, 100)
        },
        "within_tolerance_fraction": float(np.mean(err <= atol + rtol * np.abs(a))),
        "all_within_tolerance": bool(np.all(err <= atol + rtol * np.abs(a))),
        "atol": atol,
        "rtol": rtol,
        "relative_floor": relative_floor,
        "nonfinite_count": 0,
    }
    if a.ndim == 2:
        out["mae_per_coordinate"] = err.mean(0).tolist()
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        dot = np.sum(a * b, axis=1)
        cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        cos[(np.linalg.norm(a, axis=1) == 0) & (np.linalg.norm(b, axis=1) == 0)] = 1
        out["mean_cosine_similarity"] = float(np.clip(cos, -1, 1).mean())
    return out


def score_comparison(
    calibration,
    reference,
    actual,
    labels=None,
    shrinkage=0.01,
    background_quantile=0.95,
    **kwargs,
):
    cal = np.asarray(calibration, np.float64)
    a = np.asarray(reference, np.float64)
    b = np.asarray(actual, np.float64)
    if (
        cal.ndim != 2
        or len(cal) < 2
        or a.ndim != 2
        or a.shape != b.shape
        or cal.shape[1] != a.shape[1]
    ):
        raise ValueError("Invalid score embedding shapes")
    if not all(np.isfinite(x).all() for x in (cal, a, b)):
        raise ValueError("Nonfinite score input")
    mean = cal.mean(0)
    cov = np.atleast_2d(np.cov(cal, rowvar=False))
    rank = int(np.linalg.matrix_rank(cov))
    scale = float(np.trace(cov) / len(mean))
    scale = scale if scale > 1e-12 else 1.0
    regularized = (1 - shrinkage) * cov + shrinkage * scale * np.eye(len(mean))
    inv = np.linalg.inv(regularized)

    def score(x):
        d = x - mean
        return np.einsum("ni,ij,nj->n", d, inv, d)

    ca = score(cal)
    sa = score(a)
    sb = score(b)
    threshold = float(np.quantile(ca, background_quantile))
    correlation = (
        float(spearmanr(sa, sb).statistic)
        if np.ptp(sa) > 0 and np.ptp(sb) > 0
        else None
    )
    result = {
        "calibration": "frozen_reference_mean_and_regularized_covariance",
        "covariance_rank": rank,
        "shrinkage": shrinkage,
        "threshold": threshold,
        "background_quantile": background_quantile,
        "score_error": differences(sa, sb, **kwargs),
        "spearman": correlation,
        "decision_disagreement_fraction": float(
            np.mean((sa > threshold) != (sb > threshold))
        ),
        "task_interpretation": "Artificial shifted random inputs only; not physics sensitivity.",
    }
    if labels is not None and len(np.unique(labels)) == 2:
        labels = np.asarray(labels)
        result["toy_auc_reference"] = float(roc_auc_score(labels, sa))
        result["toy_auc_hls"] = float(roc_auc_score(labels, sb))
        for name, s in [("reference", sa), ("hls", sb)]:
            result[f"toy_fpr_{name}"] = float(np.mean(s[labels == 0] > threshold))
            result[f"toy_tpr_{name}"] = float(np.mean(s[labels == 1] > threshold))
        result["background_count"] = int(np.sum(labels == 0))
        result["empirical_fpr_step"] = 1.0 / result["background_count"]
    return result, {
        "reference_score": sa,
        "hls_score": sb,
        "calibration_score": ca,
        "score_mean": mean,
        "score_covariance_inverse": inv,
    }

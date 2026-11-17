"""Scientific figures from saved evidence; missing measurements are never zeros."""

from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def variant_key(row):
    """Axes held constant when plotting precision/reuse; never merge variants."""
    return (
        row.get("strategy"),
        row.get("accum_precision"),
        row.get("layer_overrides", "{}"),
        row.get("broadcast_implementation", "not recorded"),
        row.get("component_name"),
        row.get("relu_implementation", "not recorded"),
    )


def bounds_for_layers(stats, names):
    positions = []
    magnitudes = []
    for i, name in enumerate(names):
        low = stats[name].get("representable_min")
        high = stats[name].get("representable_max")
        if low is not None and high is not None:
            positions.append(i)
            magnitudes.append(max(abs(low), abs(high)))
    return positions, magnitudes


def make_plots(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)

    colors = plt.get_cmap("tab10")

    def table_plot(columns, title, name, ylabel):
        available = [
            col
            for col in columns
            if any(r["row"].get(col) is not None for r in records)
        ]
        if not available:
            return
        fig, ax = plt.subplots(figsize=(max(6, len(records) * 0.55), 4))
        for i, col in enumerate(available):
            x = [j for j, r in enumerate(records) if r["row"].get(col) is not None]
            ax.plot(
                x,
                [records[j]["row"][col] for j in x],
                marker="o",
                label=col,
                color=colors(i),
            )
        ax.set_xticks(
            range(len(records)),
            [r["row"]["id"] for r in records],
            rotation=45,
            ha="right",
        )
        ax.set(title=title, xlabel="Configuration", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        save(fig, name)

    table_plot(
        ["component_mae", "component_max_abs"],
        "Isolated component output conversion error",
        "component_errors",
        "Absolute error",
    )
    table_plot(
        ["rho_mae", "rho_max_abs"],
        "Encoder output conversion error",
        "output_errors",
        "Absolute error",
    )
    table_plot(
        ["reference_chamfer", "hls_chamfer"],
        "Reconstruction through the same frozen host decoder",
        "reconstruction",
        "Masked squared Chamfer",
    )
    table_plot(
        ["hls_latency_cycles", "hls_ii_cycles"],
        "HLS estimates; target-clock assumptions",
        "hls_cycles",
        "Cycles",
    )
    table_plot(
        ["hls_DSP", "hls_LUT", "hls_FF"],
        "HLS resource estimates",
        "hls_resources",
        "Estimated resource count",
    )
    table_plot(
        ["synthesis_seconds", "compile_seconds"],
        "Host tool execution time",
        "build_time",
        "Seconds",
    )
    table_plot(
        ["mapped_LUT", "routed_LUT"],
        "Vivado mapped/routed LUT evidence",
        "implemented_resources",
        "LUT count",
    )
    for metric, ylabel in [
        ("hls_DSP", "Estimated DSPs"),
        ("hls_LUT", "Estimated LUTs"),
        ("hls_ii_cycles", "Estimated initiation interval (cycles)"),
    ]:
        groups = {}
        for record in records:
            row = record["row"]
            if row.get(metric) is None or row.get("reuse_requested") is None:
                continue
            groups.setdefault((row.get("precision"), *variant_key(row)), []).append(row)
        if groups:
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, rows in groups.items():
                rows = sorted(rows, key=lambda r: r["reuse_requested"])
                ax.plot(
                    [r["reuse_requested"] for r in rows],
                    [r[metric] for r in rows],
                    "-o",
                    label=" / ".join(map(str, key[:3]))
                    + " / overrides="
                    + (
                        "none"
                        if key[3] == "{}"
                        else hashlib.sha256(str(key[3]).encode()).hexdigest()[:8]
                    )
                    + " / broadcast="
                    + str(key[4])
                    + (" / " + str(key[5]) if key[5] else ""),
                )
            ax.set(
                xlabel="Requested global reuse (effective values are layer-specific)",
                ylabel=ylabel,
                title="Precision/reuse trade-off: HLS estimates",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
            save(fig, "reuse_" + metric)
    # Heatmaps preserve missing/failed measurements as masked cells.
    for variant in sorted({variant_key(r["row"]) for r in records}, key=str):
        subset = [r["row"] for r in records if variant_key(r["row"]) == variant]
        strategy = str(variant[0])
        variant_id = hashlib.sha256(json.dumps(variant).encode()).hexdigest()[:8]
        precisions = sorted({str(r["precision"]) for r in subset})
        reuse = sorted(
            {r["reuse_requested"] for r in subset if r["reuse_requested"] is not None}
        )
        if len(precisions) < 2 or len(reuse) < 2:
            continue
        for metric in (
            "rho_mae",
            "relative_chamfer_change",
            "hls_DSP",
            "hls_ii_cycles",
        ):
            matrix = np.full((len(precisions), len(reuse)), np.nan)
            for row in subset:
                if row.get(metric) is not None and row["reuse_requested"] is not None:
                    matrix[
                        precisions.index(str(row["precision"])),
                        reuse.index(row["reuse_requested"]),
                    ] = row[metric]
            if not np.isfinite(matrix).any():
                continue
            fig, ax = plt.subplots(
                figsize=(max(6, len(reuse) * 1.2), max(3, len(precisions) * 0.7))
            )
            image = ax.imshow(
                np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis"
            )
            for y in range(len(precisions)):
                for x in range(len(reuse)):
                    value = matrix[y, x]
                    ax.text(
                        x,
                        y,
                        f"{value:.3g}" if np.isfinite(value) else "--",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            ax.set_xticks(range(len(reuse)), reuse)
            ax.set_yticks(range(len(precisions)), precisions)
            ax.set(
                xlabel="Requested global reuse",
                ylabel="Precision",
                title=f"{metric} / {strategy} / acc {variant[1]} / {variant[3]} / variant {variant_id}",
            )
            fig.colorbar(image, ax=ax, label=metric)
            import re

            save(
                fig,
                "heatmap_"
                + metric
                + "_"
                + re.sub("[^A-Za-z0-9]+", "_", strategy)
                + "_"
                + variant_id,
            )
    points = [
        r["row"]
        for r in records
        if r["row"].get("hls_DSP") is not None
        and r["row"].get("relative_chamfer_change") is not None
    ]
    if points:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, row in enumerate(points):
            ax.scatter(
                row["hls_DSP"], 100 * row["relative_chamfer_change"], color=colors(i)
            )
            ax.annotate(
                row["id"],
                (row["hls_DSP"], 100 * row["relative_chamfer_change"]),
                fontsize=7,
            )
        ax.set(
            xlabel="Estimated DSPs",
            ylabel="Reconstruction loss change (%)",
            title="Resource versus task-quality trade-off",
        )
        ax.grid(alpha=0.2)
        save(fig, "resource_quality")
    # Histories and checkpoint diagnostics are frozen across study points.
    if records:
        path = records[0]["path"]
        history_path = path / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history:
                fig, ax = plt.subplots(figsize=(7, 4))
                for key in ("train_loss", "validation_loss"):
                    ax.plot(
                        [h["epoch"] for h in history],
                        [h[key] for h in history],
                        "-o",
                        label=key,
                    )
                diag = records[0]["summary"].get("training_diagnostics", {})
                if diag.get("constant_validation_chamfer") is not None:
                    ax.axhline(
                        diag["constant_validation_chamfer"],
                        ls="--",
                        label="Constant-mean validation baseline",
                    )
                ax.set(
                    xlabel="Epoch",
                    ylabel="Masked squared Chamfer",
                    title="Floating-model training",
                )
                ax.legend()
                ax.grid(alpha=0.2)
                save(fig, "training")
        arrays = path / "checkpoint/diagnostics.npz"
        if arrays.exists():
            with np.load(arrays, allow_pickle=False) as z:
                if "encoder_weights" in z:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(z["encoder_weights"], bins=70)
                    ax.set(
                        xlabel="Learned encoder weight",
                        ylabel="Count",
                        title="Selected checkpoint: trainable weights only",
                    )
                    save(fig, "weights")
                if "rho_validation" in z:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.bar(
                        np.arange(z["rho_validation"].shape[1]),
                        np.std(z["rho_validation"], axis=0),
                    )
                    ax.set(
                        xlabel="Embedding coordinate",
                        ylabel="Validation standard deviation",
                        title="Embedding activity / collapse diagnostic",
                    )
                    save(fig, "embedding_activity")
    for record in records:
        path = record["path"]
        prediction_path = path / "predictions.npz"
        if prediction_path.exists():
            with np.load(prediction_path, allow_pickle=False) as z:
                component = record["summary"].get("model_kind") == "deepset_component"
                a = z["reference_output" if component else "reference_rho"].ravel()
                b = z["hls_output" if component else "hls_rho"].ravel()
                indices = np.linspace(0, len(a) - 1, min(len(a), 20000), dtype=int)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].scatter(a[indices], b[indices], s=5, alpha=0.4)
                low = min(a.min(), b.min())
                high = max(a.max(), b.max())
                axes[0].plot([low, high], [low, high], "k--")
                axes[0].set(
                    xlabel="Reference component output"
                    if component
                    else "Reference rho",
                    ylabel="HLS component output" if component else "HLS rho",
                    title=record["row"]["id"],
                )
                error = np.sort(np.abs(a - b))
                axes[1].plot(error, np.arange(1, len(error) + 1) / len(error))
                axes[1].set(
                    xlabel="Absolute component error"
                    if component
                    else "Absolute embedding error",
                    ylabel="Empirical cumulative fraction",
                    title="Conversion-error CDF",
                )
                save(fig, "predictions_" + record["row"]["id"])
                if "reference_score" in z and len(np.unique(z["label"])) == 2:
                    from sklearn.metrics import roc_curve, roc_auc_score

                    fig, ax = plt.subplots(figsize=(6, 4))
                    for key in ("reference_score", "hls_score"):
                        fpr, tpr, _ = roc_curve(z["label"], z[key])
                        auc = roc_auc_score(z["label"], z[key])
                        ax.plot(fpr, tpr, label=f"{key}: AUC={auc:.4f}")
                    ax.set(
                        xlabel="Artificial-label FPR",
                        ylabel="Artificial-label TPR",
                        title="Synthetic diagnostic ROC",
                    )
                    ax.legend()
                    save(fig, "roc_" + record["row"]["id"])
        ranges = path / "profiles/ranges.json"
        if ranges.exists():
            stats = json.loads(ranges.read_text())
            if stats:
                names = list(stats)[:30]
                fig, ax = plt.subplots(figsize=(10, 7))
                measured = [
                    stats[n]["reference"]["abs_percentiles"]["100"] for n in names
                ]
                ax.barh(
                    np.arange(len(names)), measured, label="Reference absolute maximum"
                )
                positions, bounds = bounds_for_layers(stats, names)
                if bounds:
                    ax.scatter(
                        bounds,
                        positions,
                        marker="|",
                        s=100,
                        label="Format absolute bound",
                    )
                ax.set_yticks(np.arange(len(names)), names, fontsize=6)
                ax.set_xscale("symlog", linthresh=1e-5)
                ax.set(
                    xlabel="Magnitude; calibration subset only",
                    title="Activation ranges and representable bounds (first 30 matched layers)",
                )
                ax.legend(fontsize=8)
                save(fig, "activation_ranges_" + record["row"]["id"])
        errors = path / "profiles/layer_errors.json"
        if errors.exists():
            stats = json.loads(errors.read_text())
            pairs = sorted(
                [(n, s["mae"]) for n, s in stats.items() if "mae" in s],
                key=lambda p: p[1],
                reverse=True,
            )[:20]
            if pairs:
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(np.arange(len(pairs)), [p[1] for p in pairs])
                ax.set_yticks(np.arange(len(pairs)), [p[0] for p in pairs], fontsize=7)
                ax.set(xlabel="MAE", title="Largest matched-layer conversion errors")
                save(fig, "layer_errors_" + record["row"]["id"])
        traces = (
            list((path / "hls").glob("*.resources.json"))
            if (path / "hls").exists()
            else []
        )
        for trace in traces:
            samples = json.loads(trace.read_text()).get("samples", [])
            if not samples:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                [s["seconds"] for s in samples],
                [s["group_pss_kib"] / 1024 for s in samples],
                label="Sampled process-group PSS",
            )
            ax.set(xlabel="Host elapsed seconds", ylabel="MiB", title=trace.stem)
            ax.legend()
            save(fig, "memory_" + record["row"]["id"] + "_" + trace.name.split(".")[0])
    plt.close("all")
    return figures

"""One evidence source for CSV, Markdown, LaTeX and compiled scientific reports."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from .plotting import make_plots


def latex_escape(value):
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(escapes.get(c, c) for c in str(value))


def display(value):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def collect_records(source):
    source = Path(source).resolve()
    file = (
        source
        if source.is_file()
        else source
        / ("study.json" if (source / "study.json").exists() else "summary.json")
    )
    index = json.loads(file.read_text())
    base = file.parent
    points = (
        index.get("runs")
        if "runs" in index
        else [{"path": ".", "status": index.get("status")}]
    )
    records = []
    for i, point in enumerate(points):
        path = (base / point["path"]).resolve()
        summary_path = path / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = {"status": "missing", "hardware": {}, "stages": {}}
        hls = summary.get("requested_hls", {})
        resolved = path / "resolved.yaml"
        if not hls and resolved.exists():
            import yaml

            hls = yaml.safe_load(resolved.read_text()).get("hls", {})
        metrics = summary.get("metrics", {})
        rho = metrics.get("rho", {})
        recon = metrics.get("host_reconstruction", {})
        hardware = summary.get("hardware", {})
        estimate = hardware.get("hls_estimate", {})
        resources = estimate.get("resources", {})

        def used(group, key):
            item = group.get(key, {})
            return item.get("used") if isinstance(item, dict) else item

        effective = path / "hls/emitted_precisions.json"
        reuse = None
        if effective.exists():
            layers = json.loads(effective.read_text())
            reuse = ",".join(
                map(
                    str,
                    sorted(
                        {
                            l["reuse_factor"]
                            for l in layers
                            if isinstance(l.get("reuse_factor"), int)
                        }
                    ),
                )
            )
        stages = summary.get("stages", {})
        evidence_status = summary.get("status", "missing")
        status = (
            evidence_status
            if evidence_status != "passed"
            else point.get("status", evidence_status)
        )
        override_json = json.dumps(hls.get("layer_overrides", {}), sort_keys=True)
        row = {
            "id": f"p{i:03d}",
            "run_dir": str(path),
            "status": status,
            "study_identity_error": point.get("identity_error"),
            "evaluation_split": summary.get("evaluation_split", "test"),
            "precision": hls.get("precision"),
            "accum_precision": hls.get("accum_precision"),
            "layer_overrides": override_json,
            "layer_override_id": "none"
            if override_json == "{}"
            else hashlib.sha256(override_json.encode()).hexdigest()[:8],
            "reuse_requested": hls.get("reuse_factor"),
            "reuse_effective_set": reuse,
            "strategy": hls.get("strategy"),
            "weights_sha256": summary.get("weights_sha256"),
            "dataset_sha256": summary.get("dataset_sha256"),
            "quality_within_tolerance": summary.get("quality_within_tolerance"),
            "rho_mae": rho.get("mae"),
            "rho_rmse": rho.get("rmse"),
            "rho_max_abs": rho.get("max_abs"),
            "reference_chamfer": recon.get(
                "reference", summary.get("reference_reconstruction", {})
            ).get("mean_chamfer"),
            "hls_chamfer": recon.get("hls", {}).get("mean_chamfer"),
            "relative_chamfer_change": recon.get("relative_mean_chamfer_change"),
            "score_disagreement": metrics.get("host_score", {}).get(
                "decision_disagreement_fraction"
            ),
            "hls_latency_cycles": estimate.get("latency_cycles_max"),
            "hls_ii_cycles": estimate.get("ii_cycles_max"),
            "hls_target_latency_ns": estimate.get("latency_ns_at_target"),
            "synthesis_seconds": stages.get("synthesize", {}).get("host_wall_seconds"),
            "compile_seconds": stages.get("compile", {}).get("host_wall_seconds"),
            "routed_wns_ns": hardware.get("implementation", {}).get("wns_ns"),
            "mapped_LUT": used(hardware.get("vivado_synthesis", {}), "LUT"),
            "routed_LUT": used(
                hardware.get("implementation", {}).get("resources", {}), "LUT"
            ),
        }
        for name in ("DSP", "LUT", "FF", "BRAM_18K", "URAM"):
            row["hls_" + name] = used(resources, name)
        records.append({"path": path, "summary": summary, "row": row})
    return records


HEADINGS = {
    "id": "ID",
    "precision": "Precision",
    "accum_precision": "Accumulator",
    "layer_override_id": "Overrides ID",
    "reuse_requested": "RF request",
    "reuse_effective_set": "RF emitted",
    "strategy": "Strategy",
    "status": "Run status",
    "quality_within_tolerance": "Numerical pass",
    "rho_mae": "Rho MAE",
    "rho_max_abs": "Max error",
    "reference_chamfer": "Ref. Chamfer",
    "hls_chamfer": "HLS Chamfer",
    "hls_latency_cycles": "Latency cycles",
    "hls_ii_cycles": "II cycles",
    "hls_DSP": "DSP est.",
    "hls_LUT": "LUT est.",
    "hls_FF": "FF est.",
    "mapped_LUT": "Mapped LUT",
    "routed_LUT": "Routed LUT",
    "routed_wns_ns": "WNS ns",
}


def table(rows, columns, latex=False):
    if latex:
        lines = [
            r"\begin{tabular}{" + "l" * len(columns) + "}",
            r"\toprule",
            " & ".join(latex_escape(HEADINGS.get(c, c)) for c in columns) + r" \\",
            r"\midrule",
        ]
        lines += [
            " & ".join(latex_escape(display(row.get(c))) for c in columns) + r" \\"
            for row in rows
        ]
        return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])
    clean = lambda value: display(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join(
        [
            "| " + " | ".join(HEADINGS.get(c, c) for c in columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| " + " | ".join(clean(row.get(c)) for c in columns) + " |"
                for row in rows
            ],
        ]
    )


def build_report(
    source,
    output,
    compile_pdf=False,
    overwrite=False,
    title="Floating DeepSet encoder study",
):
    records = collect_records(source)
    rows = [r["row"] for r in records]
    if not rows:
        raise ValueError("No run records to report")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty; use --overwrite deliberately")
    output.mkdir(parents=True, exist_ok=True)
    # An explicit overwrite must not leave an old PDF posing as the new report.
    if (output / "report.pdf").exists():
        (output / "report.pdf").unlink()
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = make_plots(records, output / "figures")
    tables = {
        "configurations": [
            "id",
            "precision",
            "reuse_requested",
            "reuse_effective_set",
            "strategy",
            "status",
        ],
        "quantization_details": ["id", "accum_precision", "layer_override_id"],
        "quality": [
            "id",
            "quality_within_tolerance",
            "rho_mae",
            "rho_max_abs",
            "reference_chamfer",
            "hls_chamfer",
        ],
        "hls_estimates": [
            "id",
            "hls_latency_cycles",
            "hls_ii_cycles",
            "hls_DSP",
            "hls_LUT",
            "hls_FF",
        ],
        "implementation": ["id", "mapped_LUT", "routed_LUT", "routed_wns_ns"],
    }
    (output / "tables").mkdir(exist_ok=True)
    notes = "Encoder-only FPGA boundary. Reconstruction uses a frozen host decoder. HLS resources/cycles are estimates. Target-clock throughput is conditional on timing closure. Random/structured synthetic inputs are engineering diagnostics, not physics performance. Missing values are shown as --. Individual run status and numerical acceptance are separate."
    markdown = [f"# {title}", notes]
    tex = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=18mm]{geometry}",
        r"\usepackage{booktabs,graphicx}",
        r"\begin{document}",
        r"\newsavebox{\studytab}",
        r"\section*{" + latex_escape(title) + "}",
        latex_escape(notes),
    ]
    for name, columns in tables.items():
        content = table(rows, columns, latex=True)
        (output / "tables" / f"{name}.tex").write_text(content)
        markdown += ["\n## " + name.replace("_", " ").title(), table(rows, columns)]
        tex += [
            r"\subsection*{" + latex_escape(name.replace("_", " ").title()) + "}",
            r"\begingroup\small\sbox{\studytab}{\input{tables/"
            + name
            + r".tex}}"
            + r"\noindent\ifdim\wd\studytab>\linewidth\resizebox{\linewidth}{!}{\usebox{\studytab}}"
            + r"\else\usebox{\studytab}\fi\par\endgroup",
        ]
    warnings = []
    for record in records:
        summary = record["summary"]
        if record["row"]["status"] == "missing":
            warnings.append(record["row"]["id"] + ": missing summary.json evidence")
        if record["row"].get("study_identity_error"):
            warnings.append(
                record["row"]["id"] + ": " + record["row"]["study_identity_error"]
            )
        for warning in summary.get("training_diagnostics", {}).get("warnings", []):
            warnings.append(record["row"]["id"] + ": " + warning)
        for stage, detail in summary.get("stages", {}).items():
            if detail.get("status") in ("failed", "blocked", "unsupported"):
                warnings.append(
                    record["row"]["id"] + " " + stage + ": " + detail.get("reason", "")
                )
        warnings += [
            record["row"]["id"] + ": " + str(w)
            for w in summary.get("hardware", {}).get("warnings", [])
        ]
    overrides = {
        r["layer_override_id"]: r["layer_overrides"]
        for r in rows
        if r["layer_override_id"] != "none"
    }
    for label, content in overrides.items():
        markdown += ["\nLayer overrides " + label + ": `" + content + "`"]
        tex += [
            r"\paragraph{Layer overrides "
            + latex_escape(label)
            + "} "
            + latex_escape(content)
        ]
    markdown += [
        "\n## Diagnostics and failures",
        *(
            ["- " + w for w in warnings]
            or [
                "No recorded warnings. This does not establish physics validity or complete timing signoff."
            ]
        ),
    ]
    tex += [
        r"\subsection*{Diagnostics and failures}",
        *([latex_escape(w) + r"\par" for w in warnings] or ["No recorded warnings."]),
    ]
    for figure in figures:
        markdown += [f"\n![{figure}](figures/{figure}.png)"]
        tex += [
            r"\clearpage",
            r"\begin{center}\includegraphics[width=.98\textwidth,height=.83\textheight,keepaspectratio]{figures/"
            + figure
            + r".pdf}\end{center}",
        ]
    tex.append(r"\end{document}")
    (output / "summary.md").write_text("\n\n".join(markdown))
    (output / "report.tex").write_text("\n".join(tex))
    latex = {"status": "not_requested"}
    if compile_pdf:
        compiler = shutil.which("pdflatex")
        if compiler is None:
            latex = {
                "status": "failed",
                "reason": "pdflatex not found; Markdown/TeX and figures retained",
            }
        else:
            try:
                with (output / "latex-build.log").open("w") as log:
                    for _ in range(2):
                        subprocess.run(
                            [
                                compiler,
                                "-no-shell-escape",
                                "-interaction=nonstopmode",
                                "-halt-on-error",
                                "report.tex",
                            ],
                            cwd=output,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=120,
                        )
                if not (output / "report.pdf").exists():
                    raise RuntimeError("pdflatex produced no PDF")
                latex = {"status": "passed", "pdf": "report.pdf"}
            except (subprocess.SubprocessError, RuntimeError) as exc:
                if (output / "report.pdf").exists():
                    (output / "report.pdf").unlink()
                latex = {
                    "status": "failed",
                    "reason": str(exc),
                    "log": "latex-build.log",
                }
    result = {
        "schema_version": 1,
        "status": "failed" if latex["status"] == "failed" else "passed",
        "source": str(Path(source).resolve()),
        "data_status": "incomplete"
        if any(r["status"] == "missing" for r in rows)
        else (
            "contains_failed_runs"
            if any(r["status"] != "passed" for r in rows)
            else "complete"
        ),
        "rows": rows,
        "figures": figures,
        "latex": latex,
        "warnings": warnings,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

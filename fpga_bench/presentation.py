"""Evidence-derived explanations shared by console, PDF, HTML and review bundle."""

import base64
import html
import json
import math
from pathlib import Path
import zipfile


def number(value):
    return "not measured" if value is None else f"{value:.6g}"


def explain(record):
    s, row = record["summary"], record["row"]
    metrics = s.get("metrics", {})
    rho = metrics.get("rho", {})
    quality = s.get("quality_within_tolerance")
    verdict = (
        "PASS" if quality is True else ("FAIL" if quality is False else "not measured")
    )
    lines = [
        f"Execution: {row['status']}. Output tolerance: {verdict}.",
        f"Evaluation: {row['evaluation_split']}. Precision: {row.get('precision') or 'not applicable'}; accumulator: {row.get('accum_precision') or 'not recorded'}.",
    ]
    if row.get("reuse_requested") is not None:
        lines.append(
            f"Reuse requested: {row['reuse_requested']}; emitted layer values: {row.get('reuse_effective_set') or 'not recorded'}."
        )
    if rho:
        lines.append(
            f"Embedding error: mean {number(rho.get('mae'))}; maximum {number(rho.get('max_abs'))}."
        )
        n = math.prod(rho.get("shape", [])) if rho.get("shape") else None
        fraction = rho.get("within_tolerance_fraction")
        if n is not None and fraction is not None:
            accepted = round(n * fraction)
            lines.append(
                f"Values within tolerance: {accepted} / {n} ({100 * fraction:.5g}%); outside: {n - accepted}."
            )
        if "atol" in rho and "rtol" in rho:
            lines.append(
                f"Tolerance rule: absolute error <= {rho['atol']:g} + {rho['rtol']:g} * abs(reference)."
            )
    recon = metrics.get("host_reconstruction", {})
    if recon:
        change = recon.get("relative_mean_chamfer_change")
        lines.append(
            f"Frozen-host-decoder reconstruction: {number(recon.get('reference', {}).get('mean_chamfer'))} -> {number(recon.get('hls', {}).get('mean_chamfer'))} Chamfer (lower is better)."
        )
        if change is not None:
            lines.append(
                f"Mean reconstruction change: {100 * change:+.5g}%; small signed changes are numerical perturbations, not proof of an improvement."
            )
    score = metrics.get("host_score", {})
    if score.get("decision_disagreement_fraction") is not None:
        lines.append(
            f"Host score threshold decisions changed: {100 * score['decision_disagreement_fraction']:.5g}% on these {row['evaluation_split']} inputs."
        )
    training = s.get("training_diagnostics", {})
    if training:
        lines.append(
            f"Floating training validation Chamfer: {number(training.get('initial_validation_chamfer'))} -> {number(training.get('best_validation_chamfer'))}; constant-output baseline: {number(training.get('constant_validation_chamfer'))}."
        )
        inactive = sum(v == 0 for v in training.get("rho_std_per_coordinate", []))
        if inactive:
            lines.append(
                f"Embedding activity: {inactive} coordinate(s) have zero validation variance; inspect alongside reconstruction quality."
            )
    stages = s.get("stages", {})
    hardware = s.get("hardware", {})
    estimate = hardware.get("hls_estimate", {})
    if stages.get("synthesize", {}).get("status") == "passed":
        lines.append(
            f"HLS estimate: latency {number(estimate.get('latency_cycles_max'))} cycles; initiation interval {number(estimate.get('ii_cycles_max'))} cycles. These are synthesis estimates."
        )
        resources = estimate.get("resources", {})
        lines.append(
            "HLS estimated resources: "
            + ", ".join(
                f"{key} {number(resources.get(key, {}).get('used'))}"
                for key in ("DSP", "LUT", "FF", "BRAM_18K")
            )
            + "."
        )
    else:
        state = stages.get("synthesize", {}).get("status", "not recorded")
        lines.append(
            "FPGA synthesis not run; no FPGA latency or utilization measured."
            if state == "not_requested"
            else f"HLS synthesis: {state}; no successful estimate established."
        )
    implementation = hardware.get("implementation", {})
    wns = implementation.get("wns_ns")
    if wns is not None:
        lines.append(
            f"Routed setup slack: {wns:g} ns. "
            + (
                "Target setup timing missed."
                if wns < 0
                else "Nonnegative setup slack; inspect complete timing constraints/checks."
            )
        )
    else:
        lines.append("Routed timing: not established.")
    timed = [
        (name, d["host_wall_seconds"])
        for name, d in stages.items()
        if "host_wall_seconds" in d
    ]
    if timed:
        lines.append(
            "Host stage times (not FPGA latency): "
            + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timed)
            + "."
        )
    for name, detail in stages.items():
        if detail.get("status") in ("failed", "blocked", "unsupported"):
            lines.append(
                f"Attention: {name} {detail['status']}: {detail.get('reason', 'no reason recorded')}"
            )
    for warning in training.get("warnings", []):
        lines.append("Attention: " + str(warning))
    if row.get("study_identity_error"):
        lines.append("Attention: " + row["study_identity_error"])
    if row["status"] != "passed":
        lines.append(
            "Next: resolve failed or missing evidence before interpreting this as a qualified result."
        )
    elif quality is False:
        lines.append(
            "Next: investigate precision/layer errors before accepting numerical equivalence; keep the acceptance tolerances fixed."
        )
    elif quality is True:
        lines.append(
            "Numerical baseline accepted on these inputs. Hardware cost and timing require the separately requested vendor stages."
        )
    else:
        lines.append(
            "Next: compare the saved floating checkpoint with compiled fixed-point inference; conversion quality is not measured yet."
        )
    return lines


def overview(records):
    lines = [
        f"Runs: {len(records)}; execution passed: {sum(r['row']['status'] == 'passed' for r in records)}; output tolerance passed: {sum(r['summary'].get('quality_within_tolerance') is True for r in records)}; failed: {sum(r['summary'].get('quality_within_tolerance') is False for r in records)}; not measured: {sum(r['summary'].get('quality_within_tolerance') is None for r in records)}."
    ]
    for record in records:
        lines += [f"{record['row']['id']} — {record['path'].name}", *explain(record)]
    return lines


def portable_review(records, output, figures, tables, title, result):
    """Self-contained HTML and a small ZIP: no checkpoints, arrays or vendor logs."""
    output = Path(output)
    esc = lambda value: html.escape(str(value), quote=True)
    blocks = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>" + esc(title) + "</title>",
        "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#18202b}h1,h2{line-height:1.2}section{margin:28px 0;padding:18px;background:#f3f6fa;border-radius:8px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd3dd}div.table{overflow:auto}img{max-width:100%;height:auto}small{color:#526070}@media print{section{break-inside:avoid}}</style>",
        "<h1>" + esc(title) + "</h1><h2>Results at a glance</h2>",
        "<p>" + esc(overview(records)[0]) + "</p>",
    ]
    compact = {
        "schema_version": 1,
        "report_status": result["status"],
        "latex": result["latex"],
        "warnings": result["warnings"],
        "runs": [],
    }
    for r in records:
        lines = explain(r)
        blocks += [
            "<section><h2>" + esc(r["row"]["id"] + " — " + r["path"].name) + "</h2>",
            *["<p>" + esc(line) + "</p>" for line in lines],
            "</section>",
        ]
        s = r["summary"]
        compact["runs"].append(
            {
                "configuration": r["row"],
                "interpretation": lines,
                "metrics": s.get("metrics", {}),
                "stages": s.get("stages", {}),
                "training_diagnostics": s.get("training_diagnostics", {}),
                "hardware": s.get("hardware", {}),
            }
        )
    blocks += [
        "<p><strong>Scope:</strong> Encoder-only hardware; decoder/scoring remain on the host. Synthetic diagnostics do not establish physics performance. Calibration coverage is empirical, not an overflow proof.</p>",
        "<p>PDF compilation: "
        + esc(result["latex"]["status"])
        + ". "
        + esc(result["latex"].get("reason", ""))
        + "</p>",
    ]
    for name, (headers, rows) in tables.items():
        blocks += [
            "<h2>"
            + esc(name.replace("_", " ").title())
            + '</h2><div class="table"><table><thead><tr>',
            *["<th>" + esc(h) + "</th>" for h in headers],
            "</tr></thead><tbody>",
        ]
        for row in rows:
            blocks += ["<tr>", *["<td>" + esc(v) + "</td>" for v in row], "</tr>"]
        blocks += ["</tbody></table></div>"]
    for warning in result["warnings"]:
        blocks += ["<p><strong>Attention:</strong> " + esc(warning) + "</p>"]
    for name in figures:
        encoded = base64.b64encode(
            (output / "figures" / f"{name}.png").read_bytes()
        ).decode("ascii")
        blocks += [
            "<h2>"
            + esc(name.replace("_", " ").title())
            + '</h2><img alt="'
            + esc(name)
            + '" src="data:image/png;base64,'
            + encoded
            + '">'
        ]
    blocks += ["</html>"]
    (output / "report.html").write_text("\n".join(blocks))
    (output / "review.json").write_text(json.dumps(compact, indent=2, allow_nan=False))
    with zipfile.ZipFile(output / "review.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.html", "review.json", "report.pdf"):
            if (output / name).exists():
                archive.write(output / name, name)
        archive.writestr(
            "README.txt",
            "Open report.html in any browser, or report.pdf if included. All HTML figures are embedded. review.json contains compact evidence for automated review. No checkpoint, input arrays or vendor logs are required to read this report.\n",
        )

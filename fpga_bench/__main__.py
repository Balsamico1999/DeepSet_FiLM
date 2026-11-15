"""Human-readable CLI with automatic reports; raw JSON remains opt-in."""

import argparse
import json
from pathlib import Path
import sys
from .config import load_config, resolve_path
from .data import save_dataset
from .runner import generate_for_config, run, sweep


def finish(source, result, args, enabled=True):
    from .study_report import build_report, collect_records
    from .presentation import overview

    source = Path(source)
    report = None
    if enabled:
        output = (
            Path(args.output) if getattr(args, "output", None) else source / "report"
        )
        try:
            report = build_report(
                source,
                output,
                compile_pdf=not args.no_pdf,
                overwrite=getattr(args, "overwrite", False),
                title=getattr(args, "title", "Floating DeepSet encoder study"),
            )
            report = {**report, "output_dir": str(output.resolve())}
        except Exception as exc:
            report = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        (source / "report_status.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    if args.json:
        payload = {"status": result["status"], "report": report}
        if "runs" in result:
            payload.update(study_dir=str(source), points=len(result["runs"]))
        else:
            payload.update(run_dir=str(source), stages=result.get("stages", {}))
        # Keep raw summaries on disk; stdout JSON is bounded as before.
        if report:
            payload["report"] = {
                k: report[k]
                for k in ("status", "output_dir", "reason", "latex")
                if k in report
            }
        print(json.dumps(payload, indent=2))
    else:
        print("\nResults at a glance")
        for line in overview(collect_records(source)):
            print("  " + line)
        print("\nResults directory: " + str(source.resolve()))
        if report:
            if report["status"] == "failed":
                print(
                    "Report generation failed: "
                    + report.get(
                        "reason",
                        report.get("latex", {}).get(
                            "reason", "inspect report_status.json"
                        ),
                    )
                )
                print(
                    "Completed run evidence is preserved; report failure does not change numerical results."
                )
            if "output_dir" in report:
                output = Path(report["output_dir"])
                print("Open in browser: " + str(output / "report.html"))
                if (output / "report.pdf").exists():
                    print("PDF report: " + str(output / "report.pdf"))
                print("Share this one file for review: " + str(output / "review.zip"))
        else:
            print("Automatic report disabled (--no-report).")
    return int(
        result["status"] != "passed" or bool(report and report["status"] != "passed")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Synthetic training and encoder FPGA benchmarking with automatic reports"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "sweep", "study"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], dest="overrides")
        if name == "generate":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
        else:
            p.add_argument(
                "--no-report",
                action="store_true",
                help="Skip automatic human-readable reports",
            )
            p.add_argument(
                "--no-pdf",
                action="store_true",
                help="Keep HTML/Markdown/TeX reports without running pdflatex",
            )
            if name != "sweep":
                p.add_argument(
                    "--json",
                    action="store_true",
                    help="Print machine-readable completion instead of the readable summary",
                )
            else:
                p.set_defaults(json=False)
        if name in ("sweep", "study"):
            p.add_argument("--max-runs", type=int, default=16)
    p = sub.add_parser("report")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir")
    selection.add_argument("--study-dir")
    p.add_argument("--output")
    p.add_argument(
        "--compile-latex",
        action="store_true",
        help="Compatibility flag: PDF compilation is now the default",
    )
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--title", default="Floating DeepSet encoder study")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            source = Path(args.study_dir or args.run_dir).resolve()
            filename = "study.json" if args.study_dir else "summary.json"
            result = json.loads((source / filename).read_text())
            return finish(source, result, args)
        cfg = load_config(args.config, args.overrides)
        if args.command == "generate":
            path = resolve_path(cfg, args.output)
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; use --overwrite deliberately")
            save_dataset(path, generate_for_config(cfg))
            print(path)
            return 0
        if args.command == "run":
            result = run(cfg)
            return finish(result["run_dir"], result, args, not args.no_report)
        if args.command == "study":
            from .study import run_study

            result = run_study(args.config, args.overrides, args.max_runs)
            return finish(result["study_dir"], result, args, not args.no_report)
        results, path = sweep(args.config, args.overrides, args.max_runs)
        print("Sweep index: " + path)
        status = 0
        for result in results:
            status = max(
                status, finish(result["run_dir"], result, args, not args.no_report)
            )
        return status
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run from the project root: python -m fpga_bench --help."""

import argparse
import json
from pathlib import Path
import sys
from .config import load_config, resolve_path
from .data import save_dataset
from .runner import generate_for_config, run, sweep, write_summary_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ROOT-free synthetic training and FPGA evidence pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "sweep", "study"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], dest="overrides")
        if name == "generate":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
        if name in ("sweep", "study"):
            p.add_argument("--max-runs", type=int, default=16)
    p = sub.add_parser("report")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir")
    selection.add_argument("--study-dir")
    p.add_argument("--output")
    p.add_argument("--compile-latex", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--title", default="Floating DeepSet encoder study")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            if args.study_dir or args.output or args.compile_latex:
                from .study_report import build_report

                source = Path(args.study_dir or args.run_dir)
                output = Path(args.output) if args.output else source / "report"
                result = build_report(
                    source,
                    output,
                    compile_pdf=args.compile_latex,
                    overwrite=args.overwrite,
                    title=args.title,
                )
                print(
                    json.dumps(
                        {
                            "status": result["status"],
                            "output_dir": str(output),
                            "latex": result["latex"],
                        },
                        indent=2,
                    )
                )
                return 0 if result["status"] == "passed" else 1
            p = Path(args.run_dir)
            result = json.loads((p / "summary.json").read_text())
            write_summary_csv(p / "summary.csv", result)
            print(json.dumps(result, indent=2))
            return 0
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
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "run_dir": result["run_dir"],
                        "stages": result["stages"],
                    },
                    indent=2,
                )
            )
            return 0 if result["status"] == "passed" else 1
        if args.command == "study":
            from .study import run_study

            result = run_study(args.config, args.overrides, args.max_runs)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "study_dir": result["study_dir"],
                        "points": len(result["runs"]),
                    },
                    indent=2,
                )
            )
            return int(result["status"] != "passed")
        results, path = sweep(args.config, args.overrides, args.max_runs)
        print(path)
        return int(any(r["status"] != "passed" for r in results))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

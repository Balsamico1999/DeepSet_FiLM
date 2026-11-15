"""User-facing contract: one command leaves a readable, portable review."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fpga_bench import __main__ as cli
from fpga_bench import study_report


class ReadableWorkflowTests(unittest.TestCase):
    def fixture(self, root, accepted=True):
        run = Path(root) / "run"
        run.mkdir()
        summary = {
            "status": "passed",
            "run_dir": str(run),
            "model_kind": "deepset_film",
            "evaluation_split": "validation",
            "hardware": {},
            "stages": {
                "compare": {"status": "passed"},
                "synthesize": {"status": "not_requested"},
            },
            "quality_within_tolerance": accepted,
            "requested_hls": {"precision": "fixed<28,10>", "reuse_factor": 128},
            "metrics": {
                "rho": {
                    "shape": [512, 16],
                    "mae": 0.000439,
                    "max_abs": 0.005,
                    "within_tolerance_fraction": 1.0 if accepted else 8191 / 8192,
                    "all_within_tolerance": accepted,
                    "atol": 0.01,
                    "rtol": 0.01,
                },
                "host_reconstruction": {
                    "reference": {"mean_chamfer": 0.352},
                    "hls": {"mean_chamfer": 0.3519},
                    "relative_mean_chamfer_change": -0.00004,
                },
                "host_score": {"decision_disagreement_fraction": 0.0},
            },
        }
        (run / "summary.json").write_text(json.dumps(summary))
        return run, summary

    def test_report_starts_with_interpretation_and_saves_one_shareable_review(self):
        with tempfile.TemporaryDirectory() as d:
            run, _ = self.fixture(d)
            result = study_report.build_report(run, run / "report")
            text = (run / "report/summary.md").read_text()
            self.assertLess(
                text.index("Results at a glance"), text.index("Configurations")
            )
            self.assertIn("8192 / 8192", text)
            self.assertIn("FPGA synthesis not run", text)
            html = (run / "report/report.html").read_text()
            self.assertIn("0.000439", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn("<script", html)
            with zipfile.ZipFile(run / "report/review.zip") as z:
                self.assertIn("report.html", z.namelist())
                compact = json.loads(z.read("review.json"))
                self.assertEqual(compact["runs"][0]["metrics"]["rho"]["mae"], 0.000439)
            self.assertEqual(result["review_file"], "review.zip")

    def test_failed_quality_is_explicit_and_unmeasured_quality_is_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            run, s = self.fixture(d, False)
            study_report.build_report(run, run / "report")
            self.assertIn(
                "Output tolerance: FAIL", (run / "report/summary.md").read_text()
            )
            s.pop("quality_within_tolerance")
            s.pop("metrics")
            (run / "summary.json").write_text(json.dumps(s))
            study_report.build_report(run, run / "report", overwrite=True)
            self.assertIn(
                "Output tolerance: not measured",
                (run / "report/summary.md").read_text(),
            )

    def test_plain_run_generates_report_and_prints_metrics_instead_of_json(self):
        with tempfile.TemporaryDirectory() as d:
            run, s = self.fixture(d)
            stream = io.StringIO()
            with (
                patch.object(cli, "run", return_value=s),
                contextlib.redirect_stdout(stream),
            ):
                status = cli.main(
                    [
                        "run",
                        "--config",
                        "benchmark_configs/deepset_smoke.yaml",
                        "--no-pdf",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((run / "report/report.html").exists())
            self.assertIn("Output tolerance: PASS", stream.getvalue())
            self.assertIn("review.zip", stream.getvalue())
            self.assertFalse(stream.getvalue().lstrip().startswith("{"))

    def test_study_generates_one_combined_report(self):
        with tempfile.TemporaryDirectory() as d:
            run, _ = self.fixture(d)
            index = {
                "status": "passed",
                "study_dir": d,
                "runs": [{"path": "run", "status": "passed"}],
            }
            (Path(d) / "study.json").write_text(json.dumps(index))
            with (
                patch("fpga_bench.study.run_study", return_value=index),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = cli.main(
                    [
                        "study",
                        "--config",
                        "benchmark_configs/experiments/deepset_study_smoke.yaml",
                        "--no-pdf",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((Path(d) / "report/review.zip").exists())
            self.assertFalse((run / "report").exists())

    def test_plain_report_command_builds_readable_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            run, _ = self.fixture(d)
            with contextlib.redirect_stdout(io.StringIO()):
                status = cli.main(["report", "--run-dir", str(run), "--no-pdf"])
            self.assertEqual(status, 0)
            self.assertTrue((run / "report/review.zip").exists())

    def test_opt_out_keeps_json_available_without_reporting(self):
        with tempfile.TemporaryDirectory() as d:
            run, s = self.fixture(d)
            stream = io.StringIO()
            with (
                patch.object(cli, "run", return_value=s),
                contextlib.redirect_stdout(stream),
            ):
                status = cli.main(
                    [
                        "run",
                        "--config",
                        "benchmark_configs/deepset_smoke.yaml",
                        "--no-report",
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stream.getvalue())["status"], "passed")
            self.assertFalse((run / "report").exists())

    def test_report_failure_preserves_run_and_prints_clear_failure(self):
        with tempfile.TemporaryDirectory() as d:
            run, s = self.fixture(d)
            stream = io.StringIO()
            with (
                patch.object(cli, "run", return_value=s),
                patch.object(
                    study_report,
                    "build_report",
                    side_effect=RuntimeError("report disk error"),
                ),
                contextlib.redirect_stdout(stream),
            ):
                status = cli.main(
                    [
                        "run",
                        "--config",
                        "benchmark_configs/deepset_smoke.yaml",
                        "--no-pdf",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("Report generation failed", stream.getvalue())
            self.assertEqual(
                json.loads((run / "summary.json").read_text())["status"], "passed"
            )
            self.assertTrue((run / "report_status.json").exists())

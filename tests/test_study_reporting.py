import json
import tempfile
import unittest
import shutil
from unittest.mock import patch
from pathlib import Path
from fpga_bench import study_report
from fpga_bench.reports import normalize_reports


class ReportingTests(unittest.TestCase):
    def fixture(self, root):
        run = Path(root) / "run"
        run.mkdir()
        summary = {
            "schema_version": 1,
            "run_dir": str(run),
            "status": "passed",
            "model_kind": "deepset_film",
            "stages": {},
            "evaluation_split": "validation",
            "requested_hls": {
                "precision": "fixed<16,6>",
                "reuse_factor": 4,
                "strategy": "Resource",
            },
            "hardware": normalize_reports({}),
            "quality_within_tolerance": False,
            "metrics": {
                "rho": {"mae": 0.1, "max_abs": 0.5},
                "host_reconstruction": {"relative_mean_chamfer_change": 0.02},
            },
        }
        (run / "summary.json").write_text(json.dumps(summary))
        (run / "history.json").write_text(
            '[{"epoch":1,"train_loss":1.0,"validation_loss":1.1}]'
        )
        return run

    def test_missing_utilization_stays_missing_and_tables_escape_labels(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            result = study_report.build_report(
                run, Path(d) / "report", compile_pdf=False, title="Study_1 & 50%"
            )
            self.assertEqual(result["status"], "passed")
            self.assertIsNone(result["rows"][0]["hls_DSP"])
            tex = (Path(d) / "report/report.tex").read_text()
            self.assertIn(r"Study\_1 \& 50\%", tex)
            self.assertTrue((Path(d) / "report/summary.md").exists())
            with self.assertRaises(FileExistsError):
                study_report.build_report(run, Path(d) / "report", compile_pdf=False)

    @unittest.skipUnless(shutil.which("pdflatex"), "Optional pdflatex compiler")
    def test_real_latex_compiles_and_failed_stage_is_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            s = json.loads((run / "summary.json").read_text())
            s["status"] = "failed"
            s["stages"] = {"synthesize": {"status": "failed", "reason": "missing tool"}}
            (run / "summary.json").write_text(json.dumps(s))
            result = study_report.build_report(
                run, Path(d) / "report", compile_pdf=True
            )
            self.assertEqual(result["latex"]["status"], "passed")
            pdf = Path(d) / "report/report.pdf"
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertEqual(result["rows"][0]["status"], "failed")
            self.assertIsNone(result["rows"][0]["hls_DSP"])

    def test_missing_compiler_preserves_sources_and_marks_report_failed(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            with patch("fpga_bench.study_report.shutil.which", return_value=None):
                result = study_report.build_report(
                    run, Path(d) / "report", compile_pdf=True
                )
            self.assertEqual(result["status"], "failed")
            self.assertTrue((Path(d) / "report/report.tex").exists())
            self.assertFalse((Path(d) / "report/report.pdf").exists())

    def test_hardware_figures_use_estimates_and_failed_points_remain_missing(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            s = json.loads((run / "summary.json").read_text())
            s["hardware"] = normalize_reports(
                {
                    "CSynthesisReport": {
                        "WorstLatency": "12",
                        "IntervalMax": "4",
                        "DSP": "8",
                        "LUT": "100",
                        "FF": "200",
                    }
                }
            )
            (run / "summary.json").write_text(json.dumps(s))
            result = study_report.build_report(
                run, Path(d) / "report", compile_pdf=False
            )
            self.assertEqual(result["rows"][0]["hls_DSP"], 8)
            self.assertIn("hls_resources", result["figures"])
            self.assertIn("reuse_hls_DSP", result["figures"])

    def test_absent_summary_overrides_passed_index_and_identity_errors_are_visible(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "study.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "path": "missing",
                                "status": "passed",
                                "identity_error": "weights changed",
                            }
                        ]
                    }
                )
            )
            result = study_report.build_report(root, root / "report", compile_pdf=False)
            self.assertEqual(result["rows"][0]["status"], "missing")
            self.assertEqual(result.get("data_status"), "incomplete")
            self.assertTrue(any("weights changed" in w for w in result["warnings"]))
            self.assertTrue(any("missing" in w.lower() for w in result["warnings"]))

    def test_accumulator_and_layer_overrides_survive_provenance_and_plot_grouping(self):
        from fpga_bench import plotting

        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            s = json.loads((run / "summary.json").read_text())
            s["requested_hls"].update(
                accum_precision="fixed<32,12>",
                layer_overrides={"dense": {"ReuseFactor": 2}},
            )
            (run / "summary.json").write_text(json.dumps(s))
            row = study_report.collect_records(run)[0]["row"]
            self.assertIn("accum_precision", row)
            self.assertEqual(row["accum_precision"], "fixed<32,12>")
            self.assertIn("dense", row["layer_overrides"])
            self.assertTrue(hasattr(plotting, "variant_key"))
            other = dict(row, accum_precision="fixed<24,10>")
            self.assertNotEqual(plotting.variant_key(row), plotting.variant_key(other))
            other = dict(row, layer_overrides="{}")
            self.assertNotEqual(plotting.variant_key(row), plotting.variant_key(other))

    def test_failed_latex_removes_partial_pdf(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            run = self.fixture(d)
            out = Path(d) / "report"

            def fail(command, **kwargs):
                (Path(kwargs["cwd"]) / "report.pdf").write_bytes(b"%PDF-partial")
                raise subprocess.CalledProcessError(1, command)

            with (
                patch(
                    "fpga_bench.study_report.shutil.which",
                    return_value="/usr/bin/pdflatex",
                ),
                patch("fpga_bench.study_report.subprocess.run", side_effect=fail),
            ):
                result = study_report.build_report(run, out, compile_pdf=True)
            self.assertEqual(result["latex"]["status"], "failed")
            self.assertFalse((out / "report.pdf").exists())

    def test_unavailable_range_bounds_are_not_replaced_by_zero(self):
        from fpga_bench import plotting

        self.assertTrue(hasattr(plotting, "bounds_for_layers"))
        stats = {
            "missing": {},
            "zero": {"representable_min": 0, "representable_max": 0},
            "valid": {"representable_min": -4, "representable_max": 3.5},
        }
        self.assertEqual(
            plotting.bounds_for_layers(stats, list(stats)), ([1, 2], [0, 4])
        )

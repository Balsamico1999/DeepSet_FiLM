import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from fpga_bench.study_report import build_report


class VendorReviewTests(unittest.TestCase):
    def test_review_includes_bounded_failure_log_and_compact_memory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "hls").mkdir()
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "stages": {
                            "synthesize": {"status": "failed", "reason": "timeout"}
                        },
                        "hardware": {},
                    }
                )
            )
            (root / "hls/synthesize.log").write_text(
                "prefix\n" * 20000 + "ERROR: compiler stopped\n"
            )
            (root / "hls/synthesize.resources.json").write_text(
                json.dumps(
                    {
                        "sampled_peak_group_pss_kib": 5000000,
                        "samples": [
                            {
                                "group_pss_kib": 5000000,
                                "system_available_kib": 40000,
                                "system_swap_used_kib": 20000000,
                                "seconds": 20,
                            }
                        ],
                    }
                )
            )
            build_report(root, root / "report")
            with zipfile.ZipFile(root / "report/review.zip") as z:
                names = z.namelist()
                log = next(n for n in names if n.endswith("synthesize.log.tail.txt"))
                self.assertIn(b"ERROR: compiler stopped", z.read(log))
                self.assertLessEqual(len(z.read(log)), 65536)
                compact = json.loads(z.read("review.json"))["runs"][0][
                    "vendor_diagnostics"
                ]
                self.assertEqual(
                    compact["synthesize"]["memory"]["peak_system_swap_used_kib"],
                    20000000,
                )
                self.assertIn(
                    "ERROR: compiler stopped", (root / "report/report.html").read_text()
                )
                self.assertFalse(any(n.endswith(".resources.json") for n in names))

    def test_missing_proc_memory_values_do_not_break_diagnostics(self):
        from fpga_bench.vendor_diagnostics import collect_vendor_diagnostics

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "hls").mkdir()
            samples = [
                {"system_available_kib": None, "system_swap_used_kib": None},
                {"system_available_kib": 40000, "system_swap_used_kib": 20000000},
            ]
            (root / "hls/synthesize.resources.json").write_text(
                json.dumps({"samples": samples})
            )
            data, _ = collect_vendor_diagnostics(root)
            self.assertEqual(
                data["synthesize"]["memory"]["minimum_system_available_kib"], 40000
            )
            self.assertEqual(
                data["synthesize"]["memory"]["peak_system_swap_used_kib"], 20000000
            )

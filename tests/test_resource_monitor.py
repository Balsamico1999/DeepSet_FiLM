import json
from pathlib import Path
import sys
import tempfile
import unittest
from fpga_bench.hardware import run_command


class ResourceMonitorTests(unittest.TestCase):
    def test_command_retains_sampled_memory_with_explicit_units(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tool.log"
            run_command(
                [
                    sys.executable,
                    "-c",
                    "import time; a=bytearray(8*1024*1024); time.sleep(.25)",
                ],
                d,
                p,
                5,
            )
            stats = p.with_suffix(".resources.json")
            self.assertTrue(
                stats.exists(), "Sampled process-group resource record is missing"
            )
            data = json.loads(stats.read_text())
            self.assertIn("sampled", data["measurement"])
            self.assertTrue(data["samples"])
            if sys.platform.startswith("linux"):
                self.assertGreater(data["sampled_peak_group_pss_kib"], 0)

    def test_failed_main_process_does_not_leave_live_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = Path(d) / "child.pid"
            program = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); Path('child.pid').write_text(str(p.pid)); raise SystemExit(3)"
            with self.assertRaises(RuntimeError):
                run_command(
                    [sys.executable, "-c", program], d, Path(d) / "failed.log", 5
                )
            pid = int(pidfile.read_text())
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().rsplit(")", 1)[1].split()[0], "Z")

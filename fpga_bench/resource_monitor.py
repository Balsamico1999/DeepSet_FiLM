"""Sample Linux process-group PSS and system swap without extra dependencies."""

import json
from pathlib import Path
import sys
import threading
import time


def sample_group(pgid):
    pss = 0
    rss = 0
    count = 0
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            lines = (directory / "smaps_rollup").read_text().splitlines()
            values = {
                line.split(":", 1)[0]: int(line.split()[1])
                for line in lines
                if line.startswith(("Pss:", "Rss:"))
            }
            pss += values.get("Pss", 0)
            rss += values.get("Rss", 0)
            count += 1
        except (OSError, ValueError, IndexError):
            continue
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    return {
        "group_pss_kib": pss,
        "group_rss_kib": rss,
        "processes": count,
        "system_available_kib": info.get("MemAvailable"),
        "system_swap_used_kib": info.get("SwapTotal", 0) - info.get("SwapFree", 0),
    }


class ResourceMonitor:
    def __init__(self, pgid, path, interval=0.2):
        self.pgid = pgid
        self.path = Path(path)
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        if sys.platform.startswith("linux"):
            self.thread.start()

    def collect(self):
        while not self.stop_event.is_set():
            sample = sample_group(self.pgid)
            sample["seconds"] = time.monotonic() - self.start
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def finish(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        result = {
            "measurement": "sampled Linux process-group PSS; not an exact whole-job peak",
            "units": "KiB",
            "interval_seconds": self.interval,
            "coverage": "same process group; RSS may double-count shared pages; system swap includes other processes",
            "sampled_peak_group_pss_kib": max(
                (s["group_pss_kib"] for s in self.samples), default=None
            ),
            "sampled_peak_group_rss_kib": max(
                (s["group_rss_kib"] for s in self.samples), default=None
            ),
            "samples": self.samples,
        }
        self.path.write_text(json.dumps(result, indent=2, allow_nan=False))

"""Bounded vendor evidence for a single portable review; omit large sample arrays."""

import json
from pathlib import Path


def collect_vendor_diagnostics(run):
    root = Path(run) / "hls"
    metadata, files = {}, {}
    for stage in ("synthesize", "cosim", "vivado_synthesis", "place_and_route"):
        entry = {}
        log = root / f"{stage}.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 65536))
                tail = f.read(65536).decode("utf-8", errors="ignore")
            files[f"{stage}.log.tail.txt"] = tail.encode()
            errors = [
                s for s in tail.splitlines() if "ERROR:" in s or "failed" in s.lower()
            ]
            entry["messages"] = (errors or tail.splitlines()[-5:])[-12:]
            entry["log_tail_truncated"] = size > 65536
        resource = root / f"{stage}.resources.json"
        if resource.exists():
            try:
                raw = json.loads(resource.read_text())
                samples = raw.get("samples", [])
                entry["memory"] = {k: v for k, v in raw.items() if k != "samples"}
                entry["memory"].update(
                    sample_count=len(samples),
                    minimum_system_available_kib=min(
                        (
                            s["system_available_kib"]
                            for s in samples
                            if isinstance(s.get("system_available_kib"), (int, float))
                        ),
                        default=None,
                    ),
                    peak_system_swap_used_kib=max(
                        (
                            s["system_swap_used_kib"]
                            for s in samples
                            if isinstance(s.get("system_swap_used_kib"), (int, float))
                        ),
                        default=None,
                    ),
                )
            except (ValueError, OSError) as exc:
                entry["memory_read_error"] = str(exc)
        if entry:
            metadata[stage] = entry
    for report in root.glob("*_prj/solution*/syn/report/csynth_design_size.rpt"):
        files["csynth_design_size.rpt"] = report.read_bytes()[:131072]
        break
    return metadata, files

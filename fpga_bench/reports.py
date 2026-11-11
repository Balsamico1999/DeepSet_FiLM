"""Normalize vendor evidence without inventing values for unavailable stages."""

from pathlib import Path
import json


def number(value):
    try:
        x = float(value)
        return int(x) if x.is_integer() else x
    except (ValueError, TypeError):
        return None


def resources(raw, aliases):
    out = {}
    for label, (key, capacity) in aliases.items():
        used = number(raw.get(key))
        available = number(raw.get(capacity))
        out[label] = {
            "used": used,
            "available": available,
            "percent": 100.0 * used / available
            if used is not None and available
            else None,
        }
    return out


def parse_utilization(path):
    aliases = {
        "CLB LUTs": "LUT",
        "Slice LUTs": "LUT",
        "CLB Registers": "FF",
        "Slice Registers": "FF",
        "DSPs": "DSP",
        "DSP48E1": "DSP",
        "DSP48E2": "DSP",
        "Block RAM Tile": "BRAM_tiles",
        "URAM": "URAM",
    }
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="replace").splitlines():
        cells = [x.strip() for x in line.split("|")][1:-1]
        if len(cells) < 3:
            continue
        label = cells[0].rstrip("*").strip()
        if label in aliases:
            values = [number(x.replace(",", "")) for x in cells[1:]]
            # Vivado table: Used, Fixed, Prohibited, Available, Util%.
            used = values[0]
            available = values[-2] if len(values) >= 5 else None
            out[aliases[label]] = {
                "used": used,
                "available": available,
                "percent": 100.0 * used / available
                if used is not None and available
                else None,
            }
    return out


def normalize_reports(raw, clock_ns=5.0, implementation=None):
    syn = raw.get("CSynthesisReport", {})
    cos = raw.get("CosimReport", {})
    aliases = {
        k: (k, "Available" + k) for k in ("DSP", "LUT", "FF", "BRAM_18K", "URAM")
    }
    latency = number(syn.get("WorstLatency"))
    ii = number(syn.get("IntervalMax"))
    estimate = {
        "source": "HLS synthesis estimate",
        "latency_cycles_min": number(syn.get("BestLatency")),
        "latency_cycles_max": latency,
        "ii_cycles_max": ii,
        "estimated_clock_period_ns": number(syn.get("EstimatedClockPeriod")),
        "target_clock_period_ns": clock_ns,
        "latency_ns_at_target": latency * clock_ns if latency is not None else None,
        "ideal_jets_per_second_at_target": 1e9 / (ii * clock_ns)
        if ii and ii > 0
        else None,
        "resources": resources(syn, aliases),
    }
    impl = {
        "source": "Vivado implementation",
        "timing_met": None,
        "wns_ns": None,
        "target_clock_period_ns": clock_ns,
        "resources": {},
    }
    impl.update(implementation or {})
    if impl.get("wns_ns") is not None:
        impl["timing_met"] = impl["wns_ns"] >= 0
    return {
        "hls_estimate": estimate,
        "rtl_cosimulation": {
            "source": "RTL transaction simulation",
            "status": cos.get("Status"),
            "latency_cycles_min": number(cos.get("LatencyMin")),
            "latency_cycles_max": number(cos.get("LatencyMax")),
            "ii_cycles_min": number(cos.get("IntervalMin")),
            "ii_cycles_max": number(cos.get("IntervalMax")),
        },
        "vivado_synthesis": raw.get("VivadoSynthReport", {}),
        "implementation": impl,
    }


def collect_reports(hls_dir, clock_ns=5.0):
    hls_dir = Path(hls_dir)
    raw = {}
    warnings = []
    if any(hls_dir.glob("*_prj/solution*/syn/report/*_csynth.xml")):
        try:
            from hls4ml.report import parse_vivado_report

            raw = parse_vivado_report(str(hls_dir)) or {}
        except Exception as exc:
            warnings.append(f"Vendor report parser: {exc}")
    impl = {}
    metadata = hls_dir / "implementation_timing.json"
    if metadata.exists():
        try:
            impl = json.loads(metadata.read_text())
        except (ValueError, OSError) as exc:
            warnings.append(f"Implementation timing: {exc}")
    impl["resources"] = parse_utilization(hls_dir / "implementation_utilization.rpt")
    result = normalize_reports(raw, clock_ns, impl)
    result["warnings"] = warnings
    result["raw_report_files"] = [
        str(p.relative_to(hls_dir))
        for p in hls_dir.rglob("*")
        if p.suffix in (".rpt", ".xml", ".json") and p.is_file()
    ]
    return result, raw

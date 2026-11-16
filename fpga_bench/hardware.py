"""Faithful lowering, compiled-C comparison, bounded vendor builds and profiling."""

import json
from copy import deepcopy
import re
import os
from pathlib import Path
import shutil
import signal
import subprocess
import numpy as np
import yaml
from .config import merge
from .models import lower_broadcasts


def configure_hls(generated, cfg):
    """Apply defaults at name granularity, then explicit layer overrides."""
    config = deepcopy(generated)
    h = cfg["hls"]
    config["Model"].update(
        Precision=h["precision"], ReuseFactor=h["reuse_factor"], Strategy=h["strategy"]
    )
    for settings in config["LayerName"].values():
        settings.update(
            Precision={
                "result": h["precision"],
                "weight": h["precision"],
                "bias": h["precision"],
                "accum": h["accum_precision"],
            },
            ReuseFactor=h["reuse_factor"],
            Strategy=h["strategy"],
            Trace=bool(cfg["stages"]["profile"]),
        )
    for name, override in h["layer_overrides"].items():
        if name not in config["LayerName"]:
            raise ValueError(f"Unknown HLS layer override: {name}")
        config["LayerName"][name] = merge(config["LayerName"][name], override)
    return config


def convert_adapter(adapter, cfg, path, evaluation):
    if adapter.kind != "deepset_film":
        raise NotImplementedError(
            "Existing GraphNet requires a fixed-shape/custom graph hardware adapter; software training remains supported"
        )
    return convert_model(
        adapter.model,
        cfg,
        path,
        adapter.inputs(evaluation),
        adapter.predict(evaluation)["rho"],
    )


def convert_model(
    model,
    cfg,
    path,
    inputs,
    expected,
    project_name="deepset_bench",
    boundary="encoder only; host scoring excluded",
):
    """Convert a fixed-shape Keras model with independent floating reference outputs."""
    import hls4ml

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    h = cfg["hls"]
    implementation = h["broadcast_implementation"]
    if implementation == "indexed":
        from .indexed_broadcast import register_hls

        register_hls(h["backend"])
    lowered, rewrites = lower_broadcasts(model, implementation=implementation)
    actual = lowered(inputs, training=False).numpy()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Float lowering changed model semantics",
    )
    config = hls4ml.utils.config_from_keras_model(
        lowered,
        granularity="name",
        backend=h["backend"],
        default_precision=h["precision"],
        default_reuse_factor=h["reuse_factor"],
    )
    config = configure_hls(config, cfg)
    n = min(h["cosim_samples"], len(expected))
    flat = np.concatenate([x[:n].reshape(n, -1) for x in inputs], axis=1)
    np.save(path / "testbench_inputs.npy", flat)
    (path / "requested_hls_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (path / "lowering.json").write_text(
        json.dumps(
            {
                "implementation": implementation,
                "structure": {
                    "original_layers": len(model.layers),
                    "converted_layers": len(lowered.layers),
                    "replica_layers": sum(
                        "_replica_" in l.name for l in lowered.layers
                    ),
                    "indexed_broadcasts": sum(
                        type(l).__name__ == "IndexedBroadcast" for l in lowered.layers
                    ),
                    "transpose_layers": sum(
                        type(l).__name__ == "Permute" for l in lowered.layers
                    ),
                },
                "rewrites": rewrites,
                "max_float_lowering_error": float(np.max(np.abs(actual - expected))),
                "hardware_boundary": boundary,
            },
            indent=2,
        )
    )
    backend = hls4ml.backends.get_backend(h["backend"])
    adjustments = []
    for layer in lowered.layers:
        weights = layer.get_weights()
        if layer.__class__.__name__ == "Dense" and weights:
            n_in, n_out = weights[0].shape
            settings = config["LayerName"][layer.name]
            requested = settings["ReuseFactor"]
            valid = backend.get_valid_reuse_factors(n_in, n_out)
            # hls4ml 1.3 lowers rank-3 Dense to pointwise Conv1D. Its
            # io_parallel Resource kernel asserts RF <= input channels.
            if (
                len(layer.input_shape) > 2
                and settings["Strategy"].lower() == "resource"
            ):
                valid = [rf for rf in valid if rf <= n_in]
            resolved = backend.get_closest_reuse_factor(valid, requested)
            settings["ReuseFactor"] = resolved
            adjustments.append(
                {"layer": layer.name, "requested": requested, "resolved": resolved}
            )
    (path / "reuse_adjustments.json").write_text(json.dumps(adjustments, indent=2))
    (path / "resolved_hls_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    hls = hls4ml.converters.convert_from_keras_model(
        lowered,
        hls_config=config,
        backend=h["backend"],
        part=h["part"],
        clock_period=h["clock_period_ns"],
        io_type=h["io_type"],
        output_dir=str(path),
        project_name=project_name,
        input_data_tb=str((path / "testbench_inputs.npy").resolve()),
    )
    emitted = [
        {
            "name": layer.name,
            "class": type(layer).__name__,
            "reuse_factor": layer.get_attr("reuse_factor"),
            "strategy": layer.get_attr("strategy"),
            "broadcast_parallel_lanes": layer.get_attr("broadcast_parallel_lanes"),
            "variables": {v.name: str(v.type.precision) for v in layer.get_variables()},
        }
        for layer in hls.get_layers()
    ]
    (path / "emitted_precisions.json").write_text(json.dumps(emitted, indent=2))
    return hls, lowered, rewrites


def run_command(command, cwd, log_path, timeout):
    with Path(log_path).open("w") as log:
        log.write("COMMAND: " + repr(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        from .resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(
            proc.pid, Path(log_path).with_suffix(".resources.json")
        )
        try:
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Command exceeded {timeout}s; see {log_path}")
            if code:
                raise RuntimeError(f"Command exited {code}; see {log_path}")
        except BaseException:
            # A killed main Vivado process can leave memory-hungry helpers.
            import time

            try:
                os.killpg(proc.pid, signal.SIGTERM)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        finally:
            monitor.finish()


def validate_cosim_outputs(path):
    """Check actual RTL transactions, including count, against compiled-C goldens.

    Text output uses the vendor testbench's default six significant digits.
    This tolerance covers serialization; C-to-RTL log equality is additionally
    enforced by build_prj.tcl validation=1.
    """
    path = Path(path)
    golden = np.load(path / "testbench_outputs.npy", allow_pickle=False)
    actual = np.loadtxt(path / "tb_data/rtl_cosim_results.log", ndmin=2)
    if actual.shape != golden.shape or not np.isfinite(actual).all():
        raise ValueError(
            f"Invalid RTL result shape/values: {actual.shape}, expected {golden.shape}"
        )
    np.testing.assert_allclose(
        actual,
        golden,
        rtol=1e-5,
        atol=1e-6,
        err_msg="RTL outputs differ from compiled-C goldens",
    )
    (path / "cosim_validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "transactions": len(golden),
                "max_abs": float(np.max(np.abs(actual - golden))),
                "rtol": 1e-5,
                "atol": 1e-6,
                "comparison": "RTL text outputs vs saved compiled-C outputs; Tcl also compares C/RTL logs",
            },
            indent=2,
        )
    )


def run_hls_stage(path, cfg, stage):
    path = Path(path)
    h = cfg["hls"]
    options = dict(
        reset=0,
        csim=int(stage == "synthesize"),
        synth=int(stage == "synthesize"),
        cosim=int(stage == "cosim"),
        validation=int(stage == "cosim"),
        export=0,
        vsynth=0,
        fifo_opt=0,
    )
    if h["backend"] == "Vitis":
        (path / "build_opt.tcl").write_text(
            "array set opt {\n"
            + "".join(f"{key} {value}\n" for key, value in options.items())
            + "}\n"
        )
        if shutil.which("vitis-run"):
            command = ["vitis-run", "--tcl", "build_prj.tcl", "--mode", "hls"]
        elif shutil.which("vitis_hls"):
            command = ["vitis_hls", "-f", "build_prj.tcl"]
        else:
            raise FileNotFoundError(
                "Vitis HLS launcher missing: source AMD settings64.sh (vitis-run or vitis_hls required)"
            )
    else:
        if not shutil.which("vivado_hls"):
            raise FileNotFoundError(
                "Legacy Vivado backend requires vivado_hls; Vivado alone is insufficient"
            )
        command = [
            "vivado_hls",
            "-f",
            "build_prj.tcl",
            *[f"{key}={value}" for key, value in options.items()],
        ]
    run_command(command, path, path / f"{stage}.log", h["timeout_seconds"])
    if stage == "synthesize" and not any(
        path.glob("*_prj/solution*/syn/report/*_csynth.xml")
    ):
        raise RuntimeError("HLS command exited without a synthesis XML report")
    if stage == "cosim":
        from hls4ml.report import parse_vivado_report

        cos = (parse_vivado_report(str(path)) or {}).get("CosimReport", {})
        if str(cos.get("Status", "")).lower() != "pass":
            raise RuntimeError(f"RTL co-simulation did not report Pass: {cos}")
        validate_cosim_outputs(path)


def implementation_tcl(route, clock_ns, threads=1, clock_source=None):
    if not isinstance(threads, int) or not 1 <= threads <= 8:
        raise ValueError("Vivado threads must be 1..8")
    if clock_source is not None and not re.fullmatch(
        r"[A-Za-z0-9_./:-]+", clock_source
    ):
        raise ValueError("Invalid clock source site")
    tail = (
        """
place_design
phys_opt_design
route_design
report_utilization -file implementation_utilization.rpt
report_timing_summary -file implementation_timing.rpt
set paths [get_timing_paths -max_paths 1]
set wns null
if {[llength $paths] > 0} {set wns [get_property SLACK [lindex $paths 0]]}
set fh [open implementation_timing.json w]
puts $fh [format {{"wns_ns": %s}} $wns]
close $fh
write_checkpoint -force routed.dcp
"""
        if route
        else ""
    )
    constraint = (
        f"create_clock -name ap_clk -period {float(clock_ns)} [get_ports ap_clk]"
    )
    if clock_source:
        constraint += f"\nset_property HD.CLK_SRC {clock_source} [get_ports ap_clk]"
    return (
        f"set_param general.maxThreads {threads}\n"
        + """source project.tcl
set rtl [glob -nocomplain ${project_name}_prj/solution1/syn/verilog/*.v]
if {[llength $rtl] == 0} {error "No synthesized HLS Verilog found"}
read_verilog $rtl
set fh [open benchmark_clock.xdc w]
"""
        + "\n".join("puts $fh {" + line + "}" for line in constraint.splitlines())
        + """
close $fh
read_xdc benchmark_clock.xdc
synth_design -top $project_name -part $part -mode out_of_context
opt_design
report_utilization -file vivado_synth.rpt
report_timing_summary -file vivado_synth_timing.rpt
"""
        + tail
    )


def run_vivado_stage(path, cfg, route=False):
    path = Path(path)
    if not shutil.which("vivado"):
        raise FileNotFoundError("vivado missing: source the Vivado settings64.sh")
    script = path / ("benchmark_route.tcl" if route else "benchmark_synth.tcl")
    script.write_text(
        implementation_tcl(
            route,
            cfg["hls"]["clock_period_ns"],
            cfg["hls"]["vivado_threads"],
            cfg["hls"]["clock_source"],
        )
    )
    run_command(
        ["vivado", "-mode", "batch", "-source", script.name],
        path,
        path / ("place_and_route.log" if route else "vivado_synthesis.log"),
        cfg["hls"]["timeout_seconds"],
    )
    required = path / ("implementation_timing.json" if route else "vivado_synth.rpt")
    if not required.exists():
        raise RuntimeError(f"Vivado exited without required report {required.name}")


def profile_model(
    adapter, lowered, hls, split, path, max_samples=128, split_name="unspecified"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from hls4ml.model.profiling import (
        numerical,
        array_to_summary,
        plots,
        types_plots,
        activation_types_hlsmodel,
    )
    from .metrics import differences

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    x = [v[: min(max_samples, len(v))] for v in adapter.inputs(split)]
    figures = list(numerical(model=lowered, hls_model=hls))
    reference_figures = list(numerical(model=lowered, X=x))
    figures.extend(reference_figures)
    _, traces = hls.trace(x)
    from .diagnostics import array_statistics

    range_stats = {}
    activation_data = []
    for name, values in traces.items():
        values = np.abs(np.asarray(values).ravel())
        values = values[values != 0]
        if len(values):
            item = array_to_summary(values, fmt="boxplot")
            item["weight"] = name
            activation_data.append(item)
    if activation_data:
        figure = plots["boxplot"](activation_data, fmt="summary")
        types_plots["boxplot"](activation_types_hlsmodel(hls), fmt="summary")
        plt.title("HLS activations after optimization: multi-input trace")
        figures.append(figure)
    for i, figure in enumerate(figures):
        if figure is not None:
            figure.savefig(path / f"ranges_{i}.png", dpi=130)
            plt.close(figure)
    layers = [layer for layer in lowered.layers if layer.name in traces]
    results = {}
    arrays = {}
    if layers:
        values = tf.keras.Model(lowered.inputs, [layer.output for layer in layers])(
            x, training=False
        )
        if not isinstance(values, (list, tuple)):
            values = [values]
        for layer, value in zip(layers, values):
            a = value.numpy()
            b = np.asarray(traces[layer.name])
            range_stats[layer.name] = {
                "reference": array_statistics(a),
                "hls": array_statistics(b),
            }
            node = next(
                (node for node in hls.get_layers() if node.name == layer.name), None
            )
            if node is not None:
                precision = next(iter(node.get_variables())).type.precision
                if hasattr(precision, "width"):
                    width = precision.width
                    integer = getattr(precision, "integer", width)
                    signed = bool(getattr(precision, "signed", True))
                    step = 2.0 ** (-(width - integer))
                    low = -(2.0 ** (integer - 1)) if signed else 0.0
                    high = 2.0 ** (integer - int(signed)) - step
                    range_stats[layer.name].update(
                        precision=str(precision),
                        representable_min=low,
                        representable_max=high,
                        reference_outside_fraction=float(
                            np.mean((a < low) | (a > high))
                        ),
                    )

            arrays["reference__" + layer.name] = a
            arrays["hls__" + layer.name] = b
            if a.size == b.size and a.shape[0] == b.shape[0]:
                results[layer.name] = differences(
                    a.reshape(len(a), -1), b.reshape(len(b), -1)
                )
            else:
                results[layer.name] = {
                    "status": "shape_not_aligned",
                    "reference_shape": list(a.shape),
                    "hls_shape": list(b.shape),
                }
    np.savez_compressed(path / "layer_traces.npz", **arrays)
    (path / "ranges.json").write_text(
        json.dumps(range_stats, indent=2, allow_nan=False)
    )
    (path / "profile_info.json").write_text(
        json.dumps(
            {
                "split": split_name,
                "sample_ids": split["ids"][: len(x[0])].tolist(),
                "count": len(x[0]),
                "max_samples": max_samples,
            },
            indent=2,
        )
    )
    (path / "layer_errors.json").write_text(
        json.dumps(results, indent=2, allow_nan=False)
    )
    plt.close("all")
    return {
        "range_figures": sum(p is not None for p in figures),
        "matched_layer_names": list(results),
        "alignment": "matched names in lowered graph; unmatched optimized layers not paired",
    }

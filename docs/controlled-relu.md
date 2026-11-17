# Controlled ReLU and real vendor testbench inputs

This increment responds to the isolated 32x256 native ReLU timeout. That run
expanded 185 initial compiler instructions to114,695 after unrolling and73,733
after simplification. These are compiler counts, not FPGA instructions/resources.
The native kernel pipelines its entire function, which can cause the inner loop
to be fully unrolled. Its recorded ReuseFactor does not explicitly limit lanes.

## Implementation and scope

`hls.relu_implementation` selects `native` (unchanged default) or `controlled`.
The converter clones standard Keras ReLU layers to a serializable ControlledReLU
with the same name and floating equation max(x,0). Learned weights, dimensions,
normalization, activation boundaries and pooling are unchanged. Nonzero threshold,
negative slope and bounded/max-value ReLU variants are rejected in controlled
mode, rather than silently approximated. The extension supports AMD Vitis/Vivado
io_parallel only; it explicitly rejects streaming.

`controlled_relu.py` registers the Keras-v2 parser, IR layer, config/call templates
and C++ header through hls4ml's extension mechanism. The header is copied into
firmware/nnet_utils/nnet_controlled_relu.h in each generated project. It uses:

- lanes=max(1,ceil(number_of_elements / requested_reuse));
- steps=ceil(number_of_elements / lanes);
- an unrolled lane loop inside an outer loop with requested pipeline II=1;
- no function-level pipeline, with INLINE off preserving the kernel boundary;
- a guard for partial last groups.

For8192 elements and reuse128, this requests64 lanes over128 groups. Effective
steps and activation_parallel_lanes are retained in emitted_precisions.json.
Requested reuse greater than the element count resolves to one lane. This can
trade latency/throughput for compilation cost and area. It does not promise II=1,
clock closure or that all array-partitioning overhead disappears. The existing
fully partitioned io_parallel interface remains a potential cost.

The original native component YAML remains a valid baseline. A new
benchmark_configs/deepset_components_controlled_relu.yaml selects only the first
wide ReLU, controlled implementation, reuse128 and the same600-second vendor
subprocess limit. Full-encoder runs may select controlled ReLU later, but the next
test is isolated; no full-network synthesis is requested by this profile.

## Vendor testbench correction

The old component runner supplied captured inputs but not the prediction file
expected by hls4ml's generated testbench. The testbench required both files and
otherwise executed zero inputs. This did not invalidate the separate Python
compiled-C comparison, but it made the vendor C-simulation pass uninformative.

`hardware.prepare_testbench` now generates compiled-C reference outputs on the
selected captured samples, writes paired .npy and vendor .dat files, verifies
shapes/finite values, and removes the generated zero-input fallback. Missing
required files cause a nonzero testbench exit. A manifest records transaction
count and reference source. The component runner has an explicit testbench stage;
if it fails, HLS synthesis is blocked. Full-encoder preparation uses the same
helper after profiling, so tracing cannot overwrite the hardened testbench.

After a successful synthesis command, actual vendor C-simulation text outputs
must match the saved compiled-C outputs, including transaction count. If synthesis
fails or times out, any available C-simulation result is checked and recorded
without hiding the original build failure. The report/review.zip includes the
validation status and manifest. A missing result is not reported as a pass.
RTL co-simulation remains a separate existing stage, not implicitly performed.

## First command on the user's host

Apply the patch after the component-test update. Confirm /swap-vivado remains
active, source Vitis HLS and activate the existing Conda environment:

```bash
cd ~/DeepSet_FiLM
swapon --show
# Only if /swap-vivado is absent:
sudo swapon /swap-vivado
source ~/Xilinx/Vitis_HLS/2024.1/settings64.sh
conda activate deepset-bench
TRAIN_RUN=benchmark_runs/deepset_research_train_20261115T133211_11ae354f

python -m fpga_bench components \
    --config benchmark_configs/deepset_components_controlled_relu.yaml \
    --set model.checkpoint="$TRAIN_RUN/checkpoint" \
    --set data.path="$TRAIN_RUN/data.npz"
```

This checks captured-input numerical accuracy and then requests ONE controlled
ReLU HLS build. Preparation/compilation/reporting add host time outside the
600-second vendor cap. No training, full-encoder synthesis, physical mapping,
routing or RTL co-simulation is requested. Upload only report/review.zip.

Add --set stages.synthesize=false for software-only execution. To repeat the
native control with corrected vendor inputs, use the same command with
--set hls.relu_implementation=native. The shapes, checkpoint, precision, reuse
request and validation sample selection stay fixed. Native and controlled are
separate report/plot variants; do not compare old zero-input vendor simulations
as if they were captured-input numerical validation.

No new Python package is required. This stays on the qualified Python3.10 /
TensorFlow-Keras2.14 / hls4ml1.3 stack. No AMD executable is available in the
implementation workspace, so only software and generated-testbench checks can
be executed here; synthesis gains must be measured on the user's machine.

## Executed software evidence

Native and controlled compiled-C outputs were exactly equal on a trained
research-sized ReLU cut with shape32x256 and16 validation examples. This used the
earlier reduced-sample verification checkpoint, not the user's unuploaded weights.
The emitted controlled schedule recorded64 lanes and128 steps.

The actual generated testbench C++ was separately compiled with host g++ and run
on four captured examples with paired references. Its text outputs matched the
compiled-C goldens (maximum text-rounding difference approximately5e-7). Removing
the reference file caused exit code1. This is a host compilation of the generated
vendor testbench source, not a Vitis C-simulation or FPGA synthesis claim.

An independent review found no critical/important kernel or baseline regression.
It identified a stale-Csim-result edge case and a missing explicit streaming test.
The stale success was reproduced by a failing regression; new synthesis attempts
now delete prior Csim outputs and validation metadata before launch. Missing or
failed new outputs cannot inherit a prior pass. Streaming rejection has a direct
regression test. No synthesis memory/runtime or achievable-II improvement has yet
been established.

Final full suite: **61 tests passed** in460.372 seconds, including actual
compiled-C inference, original full-encoder/component checks, fail-closed paired
testbench generation, stale-result rejection, streaming rejection and pdflatex.

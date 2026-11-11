# Environment setup for DeepSet–FiLM and hls4ml

Assessed from source imports, actual API use, published package metadata, and isolated execution on 11 November 2026. Main target: Linux x86_64, CPython 3.10, TensorFlow/Keras 2.14, QKeras 0.9, hls4ml 1.3, AMD Vitis HLS. This setup deliberately preserves the current training code. Keras 3 migration is discussed separately below.

## 1. Pick the right profile

| File | Install in | Purpose |
|---|---|---|
| `requirements.txt` | `deepset-hls` | Default: TensorFlow DeepSet training, QKeras, ONNX export/runtime, analysis and hls4ml profiling/conversion |
| `requirements-common.txt` | Included by profiles | Shared scientific, plotting, configuration and compatibility dependencies |
| `requirements-graph.txt` | Same `deepset-hls` environment | Everything above plus PyTorch for GraphNet and `study_model_comparison.py` |
| `requirements-keras3-experimental.txt` | Separate Python 3.11 environment | Modern DeepSet development after optimizer/export migration; newer pinned hls4ml converter |
| `requirements-brevitas.txt` | Separate `deepset-brevitas` environment | Future PyTorch QAT/QONNX development; not used by current source |
| `requirements-lock-linux-py310.txt` | Same as default profile | Fully pinned resolved pip dependencies for the default profile, excluding ROOT, OS tools and AMD software |

All profile files use relative `-r` includes. Keep them together. The standalone `requirements.txt` cannot be moved away from `requirements-common.txt` without adjusting its include.

The optional GraphNet profile is needed to run the full current repository: `GraphNet.py`, `train_GraphNet.py`, and the comparison script import torch. No torchvision, torchaudio, torch_geometric, CUDA or Brevitas is needed for their existing CPU implementation.

## 2. Why these versions and packages

| Package/group | Actual reason in this repository |
|---|---|
| TensorFlow 2.14.1 + Keras 2.14.0 | Functional models, GradientTape, tf.data, and `tf.keras.optimizers.legacy.Adam`. Matches the hls4ml 1.3 QKeras extra's declared TensorFlow upper bound |
| QKeras 0.9.0 | Imported unconditionally by AutoEncoder.py, even when building a floating-point model |
| tensorflow-model-optimization 0.7.5 | QKeras dependency; also the upper bound in hls4ml's QKeras extra |
| keras-tuner 1.4.7 | QKeras transitive dependency; explicit selection prevents unrelated tuner-version drift |
| hls4ml 1.3.0 | Stable released conversion framework; replaces the original open-ended `>=1.0.0` |
| quantizers 1.2.2 | Runtime-tested compatibility pin: hls4ml 1.3 imports `get_fixed_quantizer_np`, which is absent from quantizers 2.0.0. Resolver success alone does not catch this import failure |
| NumPy 1.26.4 | Stable pre-NumPy-2 baseline for TensorFlow 2.14, ONNX Runtime 1.18, Numba and optional Brevitas 0.12.1 |
| SciPy | Resolution/LLP parameter fits, distributions and Gaussian smoothing |
| PyYAML | `import yaml` in trainers, data checks and studies; missing from original requirements |
| tqdm | Imported by DataProcessor at module import time |
| ONNX 1.17.0 | Graph checking/metadata and graph opset-18 export; also matches optional Brevitas export metadata |
| tf2onnx 1.16.1 | Explicitly called at the end of all three TensorFlow trainers |
| ONNX Runtime 1.18.1 | Required inference/parity checking, including GraphNet's mandatory post-training export verification |
| protobuf 3.20.3 | Declared constraint of tf2onnx 1.16.1 and QONNX 0.4; upgrading independently breaks dependency compatibility |
| matplotlib 3.8.4 | Plotting and `legend.legend_handles` in the latent-space study |
| scikit-learn | ROC/AUC in latent study and dependencies of QKeras/UMAP |
| umap-learn + Numba + pynndescent | Comparison UMAP plots; source raises if UMAP is unavailable. Pins avoid unrestricted JIT/scientific-stack drift |
| pandas + seaborn | Not directly imported by project modules, but directly imported by `hls4ml.model.profiling`, used in test.py |
| h5py | Model serialization and declared hls4ml/TensorFlow dependency |
| pydot | Both Keras and hls4ml diagram functions; requires native Graphviz `dot` too |
| setuptools 69.5.1 | Keeps legacy `pkg_resources`; also meets optional Brevitas 0.12.1's `<70` bound |
| torch 2.5.1, optional current-code profile | Provides scatter_reduce, weights_only checkpoint loading and legacy ONNX export APIs used by GraphNet |
| brevitas[export] 0.12.1 + QONNX 0.4 + onnxoptimizer + onnxscript, optional future profile | Quantized PyTorch export tooling; this does not automatically convert the existing ragged GraphNet |

Packages such as joblib, llvmlite, pydigitalwavetools, networkx, ml-dtypes and TensorBoard are still installed through dependency resolution. Their exact versions are included in the default lock. They do not all need independent hand-maintained top-level requirements.

A combined legacy/Brevitas solve is possible by backtracking to compatible transitive packages (for example older onnx-ir). Separate environments are a maintenance recommendation, not a claim that the frameworks can never coexist. The separate Brevitas profile can use its own modern export dependencies without constraining the TensorFlow baseline.

## 3. Native tools

Use a Linux distribution supported by your exact AMD tool release. Native Linux is the recommended synthesis host; do not assume a native Windows/macOS installation is equivalent. These example OS commands are for Ubuntu/Debian; adapt the package names for another supported distribution.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git graphviz
```

`g++` and `make` support hls4ml's compiled C++ simulation. Graphviz supplies `dot`; installing pydot alone is insufficient for diagrams. AMD installers may require additional distribution-specific libraries; use AMD's prerequisites for the installed release rather than treating this short list as its complete OS specification.

Two scripts (`fit_LLP_parameters.py` and `study_latent_space.py`) explicitly set `text.usetex=True`. To use them unchanged:

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended cm-super dvipng ghostscript
```

Alternatively set `text.usetex=False` in those scripts. TeX is not needed for neural-network inference, conversion, or synthesis. For a headless machine, `export MPLBACKEND=Agg` handles display access but does not disable the scripts' explicit LaTeX setting.

## 4. Create the default environment, including PyROOT

Install Miniforge/Conda first if not already available. The project uses `ROOT.TFile`, `ROOT.RDataFrame`, `Range`, `Define`, `Filter`, and `AsNumpy`; uproot alone is not a drop-in substitute. Even cached graph training currently imports ROOT through DataProcessor.

Create a clean environment from conda-forge, using strict channel priority. ROOT's compiled Python bindings must match the environment's Python version.

```bash
conda create --name deepset-hls --override-channels --channel conda-forge \
    --strict-channel-priority python=3.10 numpy=1.26.4 root_base pip
conda activate deepset-hls
```

`root_base` provides ROOT/PyROOT without the larger `root` metapackage's optional notebook stack. The ROOT build is left to the solver under the explicit Python/NumPy constraints; record the exact resolved build below. Do not claim this bootstrap command is already a fully frozen conda environment.

From the extracted project directory (the directory containing config.yaml):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-linux-py310.txt
python -m pip check
python check_environment.py
python check_environment.py --hls
```

The first check imports PyROOT and exercises a three-row RDataFrame, checks the model builders, and verifies a tiny Keras-to-ONNX numerical comparison. `--hls` additionally compiles and runs a tiny fixed-scale QKeras model through the Vitis backend's C++ simulation. It does not run Vitis HLS synthesis and does not establish that the full FiLM/mask/pooling graph converts.

Use `python -m pip install -r requirements.txt` instead of the lock if you deliberately want to resolve transitive dependencies afresh. Use `--skip-root` on the checker for a conversion-only environment; existing ROOT-dependent project scripts still need PyROOT.

Do not simultaneously source a different system ROOT's `thisroot.sh` inside this conda environment. If you already have an appropriately built system ROOT, a virtualenv is an alternative, but verify that its Python version/ABI matches and run the ROOT check. Never fix an ABI mismatch by merely adding a foreign site-packages directory to PYTHONPATH.

CPU execution is sufficient to get started. An NVIDIA GPU and CUDA are not requirements for hls4ml or FPGA synthesis. GPU training can be added later as a separately qualified environment.

## 5. Vivado is installed: check Vitis HLS separately

The hls4ml `Vitis` backend invokes the HLS compiler. Vivado's implementation/simulation tools alone do not replace it. Current hls4ml setup documentation gives Vitis HLS 2023.1+; 2024.1 is a conservative candidate among its documented tested versions. Match Vitis HLS and Vivado releases where practical, and ensure the intended FPGA device support/license is available.

Source the setup script from your actual AMD installation. For a conventional 2024.1 layout these paths may look like the following; substitute the real installation root and version:

```bash
source /opt/Xilinx/Vivado/2024.1/settings64.sh
source /opt/Xilinx/Vitis_HLS/2024.1/settings64.sh
command -v vivado
command -v vitis_hls
vivado -version
vitis_hls -version
```

Some unified installations expose a `Vitis/<version>/settings64.sh` layout and `vitis-run`; use the commands supported by that installation and confirm the generated hls4ml build scripts can find the required launcher. Do not create a fake vitis_hls symlink to vitis-run: their CLIs differ.

In conversion code explicitly select:

```python
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    backend="Vitis",
    part="xcu250-figd2104-2L-e",  # replace with the actual benchmark part
    clock_period=5,             # chosen constraint, not achieved performance
    output_dir="hls_runs/example",
)
hls_model.compile()  # C++ model, no synthesis
# Supply representative vectors and compare hls_model.predict(...) first.
hls_model.build(csim=True, synth=True, cosim=True)
# After successful simulation/synthesis, optional Vivado implementation:
# hls_model.build(vsynth=True, pnr=True)
```

This is workflow guidance, not a full working conversion of the existing model. Its broadcast/mask/precision issues remain. The existing test.py does not select the backend explicitly, uses a fresh untrained model, and has data/mask bugs; use check_environment.py first.

hls4ml documents problems with Dense + io_parallel + Resource synthesis in Vitis HLS 2025.1. If that is your installed version, qualify the exact strategy or use a documented alternative release rather than assuming an environment package change will fix it.

## 6. Optional current GraphNet/comparison support

Install CPU-only PyTorch explicitly first, avoiding an unintended download of CUDA runtime packages:

```bash
conda activate deepset-hls
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-graph.txt
python -m pip check
python check_environment.py --graph
```

The CPU wheel's local version suffix (for example +cpu) satisfies the `torch==2.5.1` requirement. For CUDA training choose the correct wheel from PyTorch's official installation instructions for the host/driver, then rerun checks. Do not install torchvision/torchaudio unless you add code that uses them.

GraphNet's input data file and code correctness issues are separate from dependency installation. In particular, cached mode still imports ROOT, and additional-cache scoring on CUDA has a known tensor-to-NumPy bug in this source snapshot.

## 7. Optional separate Brevitas environment

```bash
conda create --name deepset-brevitas --override-channels --channel conda-forge \
    --strict-channel-priority python=3.10 numpy=1.26.4 root_base pip
conda activate deepset-brevitas
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-brevitas.txt
python -m pip check
python -c "import torch, brevitas, qonnx, onnx, hls4ml; print(torch.__version__, brevitas.__version__)"
```

Omit ROOT at creation only if you will not import the repo's ROOT-coupled preprocessing modules. TensorFlow and original QKeras are deliberately absent from this profile. The checker is for the existing Keras-2 workflow and should not be run unchanged in this Brevitas-only environment.

Brevitas export dependencies are explicitly versioned rather than adding every optional Brevitas extra. LLM, vision, diffusion, notebook, FINN runtime and audio extras are unnecessary here. QONNX is the relevant interchange route for hls4ml; installing FINN, PYNQ or a board runtime is not required for this benchmark scope.

## 8. Save the environment that actually ran

After adding the selected profile and validating it:

```bash
mkdir -p environment_records
python --version > environment_records/python-version.txt
python -m pip freeze > environment_records/pip-freeze.txt
conda list --explicit > environment_records/conda-explicit-linux-64.txt
conda env export > environment_records/environment-resolved.yml
vivado -version > environment_records/vivado-version.txt
vitis_hls -version > environment_records/vitis-hls-version.txt
```

The explicit conda record captures the exact ROOT/Python binary builds on that platform; pip freeze captures the Python additions. Keep the project revision, input data identity, device/clock settings and tool reports beside them. The provided pip lock is not a replacement for recording ROOT/AMD/OS versions and has no artifact hashes.

## 9. Validation completed here

A temporary Linux x86_64 Python 3.10 environment was installed from the proposed default requirements. Imports including QKeras, TensorFlow Model Optimization, hls4ml profiling, UMAP, tf2onnx and ONNX Runtime succeeded; installed-package consistency passed. Both existing AutoEncoder model builders constructed and ran small forward passes. Tiny Keras -> ONNX -> ONNX Runtime parity passed. Tiny fixed-scale QKeras -> hls4ml Vitis-backend compiled-C parity passed. Explicit alpha=1 is used in that small quantizer fixture; this is not a test of every autoscaling quantizer setting.

All three legacy/GraphNet/Brevitas requirement profiles resolved against Linux/Python 3.10 metadata. GraphNet/Brevitas runtime, conda/PyROOT installation, full dataset training, full FiLM model conversion, RTL co-simulation and AMD synthesis/implementation were not performed here. ROOT and AMD tools are unavailable in this workspace. The package setup does not repair the previously identified data-contract, masking, normalization, or scientific-evaluation issues.

## 10. Keras 3 / QKeras-v3 migration

The DeepSet is a reasonable Keras 3 migration candidate. An isolated Python 3.11 environment with TensorFlow 2.21.0, Keras 3.14.1 and qkeras-v3 1.2.3 ran both existing AutoEncoder builders and forward passes unchanged. A TensorFlow GradientTape update using modern keras.optimizers.Adam also ran successfully. This is evidence for the model building/training core, not a complete migration of the scripts.

Required or likely changes:

| Area | Assessment |
|---|---|
| Install/import | Install qkeras-v3 instead of original qkeras. Both distribute the same import namespace `qkeras`; never install them together. Use `KERAS_BACKEND=tensorflow` before imports and leave `TF_USE_LEGACY_KERAS` unset. |
| Keras namespace | Prefer explicit `import keras` / `from keras import layers, Model`. With modern TensorFlow, tf.keras already refers to Keras 3; the unchanged builders worked in the probe. |
| Optimizers | Replace every tf.keras.optimizers.legacy.Adam with keras.optimizers.Adam. Legacy Adam raised ImportError in the probe. Match learning rate/epsilon and retrain or compare behavior; do not claim exact optimizer-state equivalence. |
| Custom losses and loops | GradientTape, tf.data and concrete TensorFlow loss/augmentation operations can remain with the TensorFlow backend. There is no need to rewrite every loss in keras.ops solely to adopt Keras 3. Other backends would require additional work. |
| Model topology | Dense, ReLU, Add, Multiply, masks and fixed pooling are reasonable to preserve. Migration does not solve full-model broadcast or precision constraints. |
| Export | Requalify the old tf2onnx.convert.from_keras route. The legacy tf2onnx/protobuf pins must not be carried over blindly to TensorFlow 2.21. Consider current Keras model.export(format="onnx") or a qualified TensorFlow-function export. Verify actual input/output names and numerical parity because comparison adapters depend on them. Direct Keras-to-hls4ml conversion does not require ONNX. |
| Serialization | Save native models with the .keras format and verify custom quantizers/layers round-trip. Save weight fixtures to compare float and quantized implementations. |
| Converter | Published hls4ml 1.3.0 failed the tiny QKeras-v3 QActivation conversion probe. Current development documentation advertises QKeras-v3 support, which must be distinguished from that released wheel. Use a qualified, pinned converter revision. |
| Unused quantizer features | QKeras-v3 documents AutoQKeras/pruning as currently non-working in 1.2.x. This repository does not use them; they do not block its manual Dense/ReLU QAT path. |
| GraphNet | The existing graph model remains PyTorch. A Keras-3 DeepSet migration does not automatically quantize or synthesize GraphNet. |

The hls4ml source revision `3c0bef2782846e93a41399fa3ec4d13633c39559` subsequently passed the same tiny QKeras-v3 compiled-C test with maximum absolute error 0.0 on three vectors. It is a development revision, not the released 1.3 wheel. The source archive was installed for this probe with an explicit development-version stamp; the commit above identifies the exact code. This demonstrates a working small-model route, not full FiLM synthesis.

To explore that route:

```bash
conda create --name deepset-keras3 --override-channels --channel conda-forge \
    --strict-channel-priority python=3.11 numpy=1.26.4 root_base pip
conda activate deepset-keras3
unset TF_USE_LEGACY_KERAS
export KERAS_BACKEND=tensorflow
python -m pip install --upgrade pip
python -m pip install -r requirements-keras3-experimental.txt
python -m pip check
python check_keras3_environment.py --hls
```

The modern profile intentionally does not include the old tf2onnx/protobuf pins. TensorFlow 2.21 requires protobuf>=6.31.1, which conflicts with legacy tf2onnx 1.16.1's protobuf 3.20 requirement. Direct Keras-to-hls4ml development can proceed while a modern ONNX export path is qualified. Do not run the existing training scripts unchanged and expect their legacy optimizer/export calls to work. The complete modern profile resolved against Linux/Python 3.11 metadata using the exact local source revision for hls4ml. Modern analysis runtime and complete dataset training were not exercised here; core model/gradient/C-conversion checks were.

Latest tested Keras 3.14.1 requires Python >=3.11; do not reuse the default Python 3.10 environment for these exact pins. Use a distinct environment and preserve the legacy baseline for comparisons. No production training script was migrated as part of this environment update.

## Sources

- hls4ml installation and frontend extras: https://fastmachinelearning.org/hls4ml/intro/setup.html
- hls4ml Vitis known issues and implementation: https://fastmachinelearning.org/hls4ml/backend/vitis.html
- ROOT installation and conda guidance: https://root.cern.ch/install/
- TensorFlow/Keras version relationship: https://keras.io/guides/migrating_to_keras_3/
- Matplotlib native LaTeX dependencies: https://matplotlib.org/3.9.2/users/explain/text/usetex.html
- QKeras-v3 installation/limitations: https://qkerasv3.readthedocs.io/en/latest/
- hls4ml current development configuration: https://github.com/fastmachinelearning/hls4ml/blob/main/pyproject.toml
- Exact released hls4ml/QKeras/Brevitas/QONNX wheel METADATA was also inspected during the dependency audit.

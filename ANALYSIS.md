# NeuralSAT — Code Analysis & CLI Usage Guide

This document analyzes the NeuralSAT codebase and explains, in detail, how to run
it from the command line.

## 1. What NeuralSAT is

NeuralSAT is a DPLL(T)-style verifier for deep neural networks. Given:
- a network (`--net`, in **ONNX** or PyTorch `.pth` format), and
- a specification (`--spec`, in **VNNLIB** format, describing input/output constraints),

it decides whether the property holds and prints one of:
- `unsat` — property holds (proved),
- `sat` — property violated (counterexample found),
- `unknown` / `timeout` — inconclusive within budget.

## 2. Repository layout relevant to running the tool

```
src/
  main.py            # primary CLI entry point (single-instance verification)
  main_dec.py         # CLI entry point using the "decompositional" verifier (for large nets, e.g. ResNet/VAE)
  wrapper.py           # neuralsat_verify()/neuralsat_falsify() — importable Python API, not a CLI
  test.py              # unittest-based smoke tests using the example networks/specs
  setting.py           # GlobalSettings singleton (`Settings`), parses --setting_file JSON, advanced settings
  configure/advanced.py# RestartSettings, MIPSettings, AbstractionSettings, DecompositionSettings
  verifier/            # core DPLL(T) verifier (verifier.py) and decompositional verifier (dec_verifier.py)
  abstractor/          # bound-propagation abstraction (auto_LiRPA-based)
  attacker/            # adversarial attack (PGD/random/MIP) used for fast SAT discovery
  heuristic/           # decision/restart/stabilize heuristics, SAT solver
  helper/
    network/           # read_onnx.py, read_pth.py — model loaders
    spec/              # objective.py — VNNLIB parser
    misc/               # logger, result codes (ReturnStatus), export helpers
    proof/              # proof/certificate export utilities
  example/             # sample ONNX models + VNNLIB specs used in docs/tests
vnncomp_scripts/       # run_instance.sh / prepare_instance.sh — VNN-COMP competition harness wrappers around main.py
web/                   # server.py — a Flask/HTTP wrapper exposing verification as a web API (not the CLI)
```

The **canonical CLI entry point is `src/main.py`**. `main_dec.py` is a variant for
large models that don't fit on one GPU (uses `DecompositionalVerifier`, requires
`--category`). Everything else (`wrapper.py`, `web/server.py`) calls into the same
`Verifier` / `parse_onnx` / `parse_vnnlib` machinery underneath.

## 3. Installation (prerequisite to running the CLI)

Full details in `doc/INSTALL.md`; summary:

- Python 3.10–3.11 recommended.
- `pip install neuralsat` (PyPI), or `pip install -r requirements.txt` in a venv,
  or conda-forge, or `./setup.sh` (uv-managed).
- Gurobi is used for MIP-based tightening/verification (`gurobipy`); a free
  academic license works. If no license is found, `Settings` auto-disables
  `use_mip_tightening`/MIP-dependent features (see `src/setting.py:5-11`).
- Optional: `pip install "third_party/haioc"` for custom CUDA/C++ ops.

All commands below are run **from the `src/` directory** (paths like
`example/onnx/...` are relative to `src/`).

### 3.1 Fixing `ModuleNotFoundError: No module named 'torch'`

This means dependencies were never installed into the interpreter you're
running `main.py` with — NeuralSAT does not vendor or auto-install `torch`;
you must run one of the installers below first. There's no single
"the" install script; pick one method and run it to completion:

| Script/command | What it does |
|---|---|
| `./setup.sh` (repo root) | See below — installs `uv` if missing, pins Python 3.12, and runs `uv sync` to install everything from `pyproject.toml` (incl. `torch`) into a project-local `.venv`. Run subsequent commands as `uv run src/main.py ...`. |
| `pip install -r requirements.txt` (inside a venv you created, e.g. `python3 -m venv neuralsat_env && source neuralsat_env/bin/activate`) | Installs pinned versions incl. `torch==2.1.2` directly via pip. |
| `pip install neuralsat` | Installs the packaged PyPI release (pulls in `torch` as a dependency per `pyproject.toml`). |
| conda-forge (see `doc/INSTALL.md` Option 3) | `conda create -n neuralsat python=3.10 -c conda-forge`, then `conda install -c conda-forge pytorch torchvision torchaudio ...`, then `pip install -r requirements.txt` for the rest. |

**Check which situation you're in before picking one:**
```bash
python3 --version         # NeuralSAT needs 3.10–3.13; 3.14+ is unsupported (pyproject: requires-python < 3.14)
python3 -c "import torch"  # ModuleNotFoundError here confirms nothing is installed yet
which pip pip3 uv          # on a minimal system, none of these may exist at all — see below
```

#### What we hit on this machine, and the fix

This environment (Fedora, system `python3` = 3.14.6) had **no `pip`, `pip3`,
or `uv` at all** — `pip install uv` (the original first line of `setup.sh`)
failed outright with `pip: command not found`. `pip` and `uv` were both
available as distro packages (`sudo dnf install python3-pip` / `sudo dnf
install uv`), but that requires root. Instead we used the official
no-sudo installer and made `setup.sh` self-healing so this never has to be
diagnosed manually again. Current `setup.sh`:

```bash
#!/bin/bash
set -e

if ! command -v uv &> /dev/null; then
    echo "[setup.sh] uv not found, installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python list
uv python pin 3.12
uv sync
```

Running `./setup.sh` now: installs `uv` to `~/.local/bin` if missing (no
sudo), downloads an isolated **CPython 3.12.13** via `uv` (bypassing the
unsupported system 3.14.6), creates `.venv` at the repo root, and installs
all 60 dependencies (`torch==2.13.0+cu130`, `onnx`, `onnxruntime`,
`gurobipy`, `onnxsim`, etc.) per `pyproject.toml`. Verified working:

```bash
$ uv run python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.13.0+cu130 False   # no GPU on this machine — CLI auto-falls back to --device cpu

$ cd src && uv run python3 main.py --net example/onnx/mnistfc-medium-net-554.onnx --spec example/vnnlib/test.vnnlib --device cpu --timeout 60
...
unsat,14.6449
```
matching the documented expected output for that example.

**Takeaway**: on any system without `pip`/`uv` preinstalled and no sudo
access, `./setup.sh` now bootstraps everything itself. On a system with an
unsupported default Python (3.14+) but `sudo` available, `sudo dnf install
uv` (or the conda-forge route) works equally well — `uv`/conda manage their
own Python version regardless of what `python3` resolves to system-wide.

## 4. Running from the CLI

### 4.1 Basic invocation

```bash
cd src
python3 main.py --net <path/to/model.onnx|.pth> --spec <path/to/spec.vnnlib>
```

With `uv`:
```bash
uv run src/main.py --net <...> --spec <...>
```

Minimal real example (from the repo's own examples):
```bash
cd src
python3 main.py --net example/onnx/mnistfc-medium-net-554.onnx --spec example/vnnlib/test.vnnlib
# -> prints "unsat,24.9284" (status, runtime in seconds)
```

A `sat` (property violated) example, which also includes a counterexample:
```bash
python3 main.py --net example/onnx/mnist-net_256x2.onnx --spec example/vnnlib/prop_1_0.05.vnnlib
# -> "sat,0.7526"
```

### 4.2 Full argument reference (`src/main.py`)

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--net` (required) | str | – | Path to network: `.onnx` or `.pth` |
| `--spec` (required) | str | – | Path to VNNLIB specification |
| `--input_shape` | ints | inferred | Override input shape, e.g. `--input_shape 1 3 32 32` |
| `--output_shape` | ints | inferred | Override output shape, e.g. `--output_shape 1 10` |
| `--batch` | int | 1000 | Max parallel branches verified per iteration |
| `--timeout` | float (sec) | 3600 | Overall wall-clock budget |
| `--device` | str | `cuda` | `cuda` or `cpu` (auto-falls back to `cpu` if no GPU) |
| `--verbosity` | 0/1/2 | 2 | 0=NOTSET, 1=INFO, 2=DEBUG |
| `--result_file` | str | – | Write `status,runtime` (and optional counterexample) to this file |
| `--export_cex` | flag | off | Include a counterexample string in `--result_file` when `sat` |
| `--disable_attack` | flag | — | (See note below: this is an `action='store_false'` flag, i.e. its presence turns the internal `use_attack` setting **off**) |
| `--disable_restart` | flag | — | Same pattern — disables the RESTART heuristic when passed |
| `--disable_stabilize` | flag | — | Same pattern — disables MIP-based STABILIZE tightening when passed |
| `--force_split` | `input`/`hidden` | auto | Force a specific branch-and-bound splitting strategy instead of the automatic heuristic |
| `--reasoning_output` | str | – | Path to export proof/reasoning steps (APTP format); requires `use_save_reasoning_step` to be enabled via `--setting_file` |
| `--setting_file` | str | – | JSON file overriding any `Settings` field (see §5) |
| `--test` | flag | off | Use special settings for small test runs |
| `--export_runtime` | flag | off | Include runtime alongside status in `--result_file` |

> **Note on the `--disable_*` flags**: they're implemented with
> `action='store_false'` (see `src/main.py:46-51`), which is unconventional —
> normally a `--disable_x` flag would use `store_true`. Because of
> `store_false`, the default value of `args.disable_attack` etc. is `True`,
> and *passing the flag sets it to `False`*, which `setting.py:62-67` then
> uses directly as `Settings.use_attack = args.disable_attack`. In practice
> this means passing e.g. `--disable_restart` on the command line does what
> its name says (disables restart), but only because the flag's boolean
> value is fed straight into the "use_X" setting — worth knowing if you read
> the source and find it inverted from what you expect.

Get this listing at any time with:
```bash
python3 main.py -h
```

### 4.3 Output

`main.py` always prints a final line:
```
<status>,<runtime_in_seconds>
```
where `<status>` is one of `unsat`, `sat`, `unknown`, `timeout` (see
`src/helper/misc/result.py`'s `ReturnStatus`). If `--result_file` is given,
the same (plus counterexample if `sat` and `--export_cex`) is written there
instead of/as well as stdout.

### 4.4 Tuning example commands (from `doc/INSTALL.md` / `src/example/cmd.txt`)

```bash
# CIFAR-10 example
python3 main.py --net example/onnx/cifar10_2_255_simplified.onnx \
                --spec example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib

# ACAS Xu example, disabling restart
python3 main.py --net example/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
                --spec example/vnnlib/prop_6.vnnlib --disable_restart

# Force CPU + input splitting
python3 main.py --net example/backup/motivation_example_159.onnx \
                --spec example/backup/motivation_example_159.vnnlib \
                --force_split hidden --device cpu

# Generate a proof/reasoning trace
python3 main.py --net example/proof/sample.onnx --spec example/proof/sample.vnnlib \
                --setting_file example/proof/proof_setting.json \
                --device cpu --reasoning_output example/aptp/ --force_split input

# Bound the total runtime and write result+counterexample to a file
python3 main.py --net example/onnx/pensieve_big_parallel.onnx \
                --spec example/vnnlib/pensieve_parallel_55.vnnlib \
                --timeout 60 --result_file out.txt --export_cex
```

### 4.5 Decompositional verifier (`src/main_dec.py`)

For very large networks (e.g. deep ResNets, VAEs) that don't fit in one shot,
use `main_dec.py`, which requires `--category` to select a pre-tuned settings
profile (see `configure/advanced.py:DecompositionSettings.setup_decompose`,
supported categories: `resnet6`, `resnet12`, `resnet18`, `resnet36`,
`vae_base`, `vae_wide`, `vae_deep`):

```bash
python3 main_dec.py --net path/to/resnet12.pth --spec path/to/spec.vnnlib --category resnet12
```

It first tries an adversarial attack, then `original_verify` (single-shot),
and falls back to `decompositional_verify` on CUDA OOM or if the category's
profile requests decomposition. Same `--batch`, `--timeout`, `--device`,
`--verbosity`, `--result_file` flags as `main.py` apply.

### 4.6 VNN-COMP competition harness

`vnncomp_scripts/run_instance.sh` and `prepare_instance.sh` wrap `main.py` for
the annual VNN-COMP verification competition's standardized interface:

```bash
./vnncomp_scripts/run_instance.sh v1 <category> <onnx_file> <vnnlib_file> <results_file> <timeout>
```

internally runs:
```bash
python3 src/main.py --net $ONNX_FILE --spec $VNNLIB_FILE --timeout $TIMEOUT \
                     --verbosity=2 --result_file $RESULTS_FILE --export_cex
```

### 4.7 Advanced settings via `--setting_file`

`--setting_file some.json` lets you override any field of `Settings`
(`src/setting.py` + `src/configure/advanced.py`) without touching code, e.g.:

```json
{
  "use_save_reasoning_step": true,
  "restart_max_runtime": 30.0,
  "mip_tightening_timeout_per_neuron": 5.0
}
```

Every key must already exist as a `Settings` attribute or `setup()` raises an
assertion error (`src/setting.py:88-90`). Notable tunables include restart
thresholds (`RestartSettings`), MIP stabilization timeouts/patience
(`MIPSettings`), abstraction method / batch sizes (`AbstractionSettings`),
and decomposition parameters (`DecompositionSettings`).

## 5. Non-CLI ways to run NeuralSAT (for context)

- **Python API**: `src/wrapper.py` exposes `neuralsat_verify(onnx_path,
  vnnlib_path, device, timeout, verbose, force_split)` and
  `neuralsat_falsify(...)` for embedding in other Python code, bypassing
  `argparse` entirely.
- **Web/API server**: `web/server.py` (Flask) exposes an HTTP endpoint that
  shells out to/calls the same verification logic, backing
  https://roars.dev/neuralsat/. Not part of the CLI but useful context if
  poking at request handling in that code path.
- **Tests**: `src/test.py` is a `unittest` suite invoking `Verifier` directly
  on the example networks/specs — a good reference for expected `sat`/`unsat`
  outcomes on the bundled examples, runnable via `python3 -m unittest
  test.py` from `src/`.

## 6. Quick troubleshooting notes

- No Gurobi license → `Settings` prints `"[!] Gurobi License not found!"` and
  disables MIP-dependent features (`src/setting.py:5-11`); verification still
  runs, just without MIP tightening/verify.
- No CUDA → `main.py` forces `args.device = 'cpu'` automatically
  (`src/main.py:71-72`), overriding a `--device cuda` request silently.
- Unsupported `--net` extension (anything but `.onnx`/`.pth`) raises
  `NotImplementedError('Unsupported network type')` (`src/main.py:82-83`).

## 7. What networks and properties NeuralSAT supports

NeuralSAT ingests an ONNX (or `.pth`) network and lowers it in two stages:
1. **`helper/network/onnx2pytorch`** converts the raw ONNX graph into an
   equivalent `torch.nn.Module` graph (`convert/operations.py`,
   `convert/layer.py`) — this stage is intentionally broad-coverage, with an
   operator implemented for essentially every op ONNX exporters commonly
   emit (see the full file list under `src/helper/network/onnx2pytorch/operations/`):
   `MatMul`/`Gemm`, `Conv`/`ConvTranspose`, `BatchNorm`, `InstanceNorm`, `LSTM`,
   `Pad`, `Reshape`/`Squeeze`/`Unsqueeze`/`Transpose`/`Tile`/`Expand`/`Resize`,
   `Slice`/`Split`/`Gather`/`GatherND`/`ScatterND`/`ScatterElements`/`TopK`/`OneHot`,
   `Add`/`Div`/`Where`/`Cast`/`Clip`/`PReLU`/`ThresholdedRelu`,
   `ReduceSum`, `ConstantOfShape`, `NonMaxSuppression`, `BitShift`, `Loop`, `Range`.
2. **`abstractor/auto_LiRPA`** (a vendored/modified copy of the auto_LiRPA
   bound-propagation library) then computes provable bounds through that
   graph. Only the subset of ops it has a `Bound*` implementation for can
   actually be *verified* (the rest can still be *run* — e.g. for the attack
   phase — but bound propagation through an unimplemented op will fail);
   dispatch is by ONNX op name → class name (`Bound{OpType}`), configured in
   `src/abstractor/auto_LiRPA/bound_general.py:658-663`.

### 7.1 Supported activation functions / nonlinearities

From `src/abstractor/auto_LiRPA/operators/*.py` (class names are the
`Bound*` bound-propagation implementations — this is the authoritative list
of what NeuralSAT can put a sound bound on):

| Category | Ops |
|---|---|
| Piecewise-linear | `ReLU` (`BoundRelu`, extends a shared `BoundTwoPieceLinear`), `LeakyReLU` (`BoundLeakyRelu`), `Sign`/`SignMerge` (`BoundSign`, `BoundSignMerge`), `HardTanh`/`Clip` (`BoundHardTanh`), `Floor` |
| S-shaped / smooth | `Sigmoid`, `Tanh`, `GELU` (`BoundGelu`), `Softplus` |
| Trigonometric | `Sin`, `Cos`, `Tan`, `Atan`, `Sec` |
| Power / algebraic | `Pow` (general power), `Sqr` (`x²`), `Sqrt`, `Reciprocal`, `Exp`, `Log` |
| Min/max & multiplicative | `Max`, `Min` (`BoundMinMax`), `Mul` (`BoundMul`, handles neuron×neuron products, e.g. attention/GLU-style nets), `Div` |
| Multi-piecewise / generic | `BoundMultiPiecewiseNonlinear` — a generalized piecewise-linear bound for arbitrary custom nonlinearities |
| Normalization | `BatchNorm` (`BoundBatchNormalization`), `LayerNorm` |
| Pooling | `MaxPool` (optimizable, i.e. gets its own alpha like an activation), `AveragePool`, `GlobalAveragePool` |
| Structural | `Linear`/`Gemm`/`MatMul`, `Conv`, `ConvTranspose`, `Pad`, `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten`/`Transpose`, `Concat`/`Slice`/`Split`, `Gather`/`GatherElements`, `Where`/`Not`/`Equal`, `Dropout` (identity at inference), `Softmax` (with a "skip last layer" quirk, see below), `RNN` |

So beyond plain ReLU FC/CNN/ResNet nets, NeuralSAT can bound networks using
**sigmoid, tanh, GELU, sin/cos, general power, and even neuron-product
layers** — i.e. it's not a ReLU-only tool, though ReLU is by far the most
optimized path (dedicated `BoundRelu`/`BoundTwoPieceLinear` machinery,
MIP-based stabilization in §8.3 is ReLU-only per `MILPTightener`'s assertion
in `src/tightener/cpu_tightener.py:18`).

### 7.2 Network architecture characteristics

- Layer types: fully-connected, convolutional (incl. transposed/dilated/grouped
  conv via `BoundConv`), residual connections (`Add`/`Concat` skip paths — any
  DAG structure via auto_LiRPA's graph model, not just chains), batch/layer
  norm, pooling, simple RNN cells.
- `src/helper/network/read_onnx.py:20-35` (`custom_quirks`) documents
  practical ONNX-export quirks NeuralSAT explicitly compensates for:
  `Reshape` batch-size handling, `Transpose`'s "merge batch size with
  channel" fix (common in GDVB-generated nets), `Softmax`/`Squeeze`
  "skip last layer" (treats a trailing softmax/squeeze as not part of the
  property-relevant output, matching VNN-COMP conventions where specs are
  stated over pre-softmax logits), and `Conv`+`BatchNorm` fusion.
- `read_onnx.py` also runs `onnxsim` (ONNX simplifier) when available to
  canonicalize/fold the graph before conversion, and there's an optional
  `DNNV`-based simplification path mentioned in `doc/INSTALL.md`.
- `--input_shape`/`--output_shape` CLI flags (§4.2) let you override
  shape inference for ONNX graphs with dynamic/ambiguous dimensions.
- `.pth` (raw PyTorch `state_dict`/scripted model) is supported as an
  alternative to ONNX via `helper/network/read_pth.py`, for workflows that
  skip ONNX export entirely (used e.g. by `main_dec.py`'s ResNet/VAE
  benchmarks).

### 7.3 Supported properties (VNNLIB specs)

Parsed by `helper/spec/read_vnnlib.py` + `helper/spec/objective.py`
(`src/helper/spec/objective.py:9-56` `Objective`, `:58-194` `DnfObjectives`),
following the standard VNN-COMP VNNLIB convention:
- **Input constraints**: an arbitrary per-dimension **box** (hyperrectangle)
  `lower_bound[i] <= x[i] <= upper_bound[i]` — not limited to a symmetric
  Lᵖ-ball; any axis-aligned box a `.vnnlib` file can express (ε-balls around
  a center point are just the common case).
- **Output constraints**: one or more linear inequalities `cs @ y <= rhs`
  over the network's output vector `y` (a conjunction/CNF of half-spaces —
  i.e. an output polytope).
- **Multiple properties in DNF**: a spec file can declare several
  `(input-box, output-polytope)` disjuncts (`DnfObjectives` = a *disjunction*
  of `Objective`s, each itself an implicit conjunction) — standard for
  multi-class robustness specs ("is NOT class 2 OR NOT class 5 OR ...").
  NeuralSAT proves `UNSAT` only if every disjunct is proven infeasible, and
  returns `SAT` as soon as one disjunct yields a genuine counterexample.
- Internally the whole DNN + input-box + negated-output-polytope conjunction
  is treated as one SAT formula (`α ∧ φ_in ∧ ¬φ_out`) per the DPLL(T)
  framing in the README — `UNSAT` of that formula ⇒ the property holds.
- Bound propagation supports the perturbation as an **L∞ (box)** or general
  **Lᵖ-norm** region at the auto_LiRPA layer (`PerturbationLpNorm`,
  `PerturbationL0Norm` in `abstractor/auto_LiRPA/perturbations.py`), though
  the VNNLIB/CLI path in practice always instantiates a box.

## 8. Search algorithm, abstractions, and optimizations

NeuralSAT's core (`src/verifier/verifier.py`, class `Verifier`, docstring
`"Branch-and-Bound verifier"`) implements a **DPLL(T)-style, GPU-batched
branch-and-bound (BaB)** search, matching the README's algorithmic framing:
boolean DPLL/CDCL over neuron activation-status variables, with a DNN-specific
theory solver providing bound-based deduction. All of this is driven from
`Verifier._parallel_dpll()` (`src/verifier/verifier.py:404-519`), one BaB
iteration per call:

```
1. (optional) MIP attack on the current worst domains
2. (optional) MIP/GPU stabilization ("Stabilize" — tighten neuron bounds)
3. pick_out(batch)      — pop a batch of open branches (a LIFO stack, see 8.2)
4. PGD attack           — cheap adversarial search for a counterexample
5. check full assignment — is every neuron already stable? (all decided ⇒ SAT/UNSAT check)
6. decision()           — branching heuristic picks what to split next ("Decide")
7. abstractor.forward() — recompute bounds for the two children ("Deduce")
8. domains_list.add()   — drop children already proved infeasible; keep the rest
```

### 8.1 Splitting mode: input splitting vs. hidden (ReLU) splitting

`Verifier.input_split` selects between two BaB regimes, chosen automatically
(overridable via `--force_split {input,hidden}`) using thresholds in
`GlobalSettings` (`src/setting.py:34-37`):
`input_splitting_threshold` (bound loosely > 0.5 ⇒ input split),
`hidden_splitting_threshold` (< 0.01 ⇒ hidden split), and
`safety_num_input_perturbed` (< 200 perturbed input dims ⇒ input split is
tractable). **Hidden/ReLU splitting** branches on individual neurons'
activation status (the "boolean abstraction" in the README); **input
splitting** instead bisects the input box — used when the property is
input-dominated (few perturbed dims, e.g. small-εeps robustness on ACAS Xu)
or as a restart fallback strategy.

### 8.2 Decide: branching heuristics (`src/heuristic/decision_heuristics.py`)

- **Hidden splitting**, selected by `decision_method`:
  - `smart` (default): **FSB-style filtered smart branching** — compute
    BaBSR scores (`heuristic/util.py:_compute_babsr_scores`) for all
    unstable ReLUs, take the top-k candidates, actually re-run a cheap
    bound pass for each candidate split (`_forward_hidden(..., simplify=True)`)
    and pick whichever candidate improves the bound the most
    (`smart_hidden_branching`, `get_topk_scores`).
  - `greedy`: brute-force lookahead over *every* unstable neuron (no top-k
    pruning) — most accurate, most expensive (`greedy_hidden_branching`).
  - `naive`: cheap proxy scores without any lookahead —
    `distance` (`min(u, -l)`), `polarity`, `scale`, or `width`
    (`naive_hidden_branching`), chosen randomly per call for diversity.
- **Input splitting**: `smart` uses the gradient-sensitivity of the output
  w.r.t. each input dim (`lAs`, i.e. the linear bound coefficients) times
  the dim's current width (`smart_input_branching`); `naive` just always
  bisects the currently-widest input dimension after a lookahead over the
  top-k widest dims (`naive_input_branching`).
- Every branch is a **two-way split** (child domains doubled: `torch.cat([v, v])`
  patterns throughout) — e.g. for ReLU, one child fixes the neuron active,
  the other inactive.

### 8.3 Deduce / Stabilize: bound tightening (the "theory solver")

- **Abstraction method** (`abstractor/abstractor.py`, class
  `NetworkAbstractor`, docstring `"Over-approximation method
  alpha-beta-CROWN"`): wraps `auto_LiRPA`'s `BoundedModule` and supports
  `backward` (CROWN), `forward`, `forward+backward`, and `crown-optimized`
  (**α-CROWN**: gradient-ascent-optimized per-neuron ReLU relaxation slopes,
  `abstractor/params.py:get_initialize_opt_params` — 100 iterations,
  Adam-style lr decay) — plus **β-CROWN** (`enable_beta_crown`,
  `get_beta_opt_params`) which adds learnable dual variables encoding the
  branch-and-bound split constraints directly into the bound computation
  (so parent-branch decisions tighten child bounds instead of just adding
  box constraints). `NetworkAbstractor.setup()` auto-falls-back through
  several `conv_mode` (`patches` vs `matrix`) and batch-size configurations
  if the default fails (large CNNs, ViT-style nets, OOM) — see
  `select_params`/`_init_module` (`src/abstractor/abstractor.py:53-120`).
- **MIP stabilization** (`src/tightener/cpu_tightener.py`, `MILPTightener`):
  periodically (governed by `mip_tightening_patience`) solves small
  per-neuron MILPs via Gurobi to prove individual ReLUs are actually stable
  (always-active/always-inactive) even though interval/CROWN bounds alone
  couldn't show it — directly shrinks the boolean search space ("reduces
  mistaken assignments by Decide", per the README). ReLU-only
  (asserted at `cpu_tightener.py:18`). An experimental GPU-based analogue
  exists (`tightener/gpu_tightener.py`, gated by `Settings.use_gpu_tightening`,
  marked TODO/in-progress in `configure/advanced.py:46`).
- **MIP verification/presolving** (`verifier/mip_solver.py`, `MIPSolver`):
  for small numbers of remaining objectives, hands the whole problem to
  Gurobi as a mixed-integer encoding for a direct complete answer, invoked
  before BaB starts (`Verifier._check_invoke_mip_presolving`,
  `verifier.py:107-121`) and gated by `Settings.use_mip_verify`.

### 8.4 Boolean layer: BCP, conflict clauses, and restarts

- **`SATSolver`** (`src/heuristic/sat_solver.py`): a tensorized boolean
  constraint propagation (BCP) engine — clauses are rows of a 2D int
  tensor; `assign()`/`bcp()` repeatedly resolve unit clauses (`_bcp_step`)
  until fixpoint or conflict, with an optional C++ (`haioc`) accelerated
  path (`multiple_assign_cpp`) for bulk assignment. This is instantiated
  per-domain when restart-learned clauses exist (`DomainsList.init_sat_solver`,
  `heuristic/util.py`).
- **Conflict-clause learning across restarts**: `DomainsList.save_conflict_clauses`
  records, for every branch pruned as infeasible, the decision literals that
  caused it; `Verifier._get_learned_conflict_clauses` harvests these and
  feeds them back in as `preconditions` on the *next* restart
  (`verifier.py:224-231`) — i.e. NeuralSAT doesn't discard search progress on
  restart, it converts it into learned clauses (CDCL-style), matching the
  README's `AnalyzeConflict` description.
- **Restart heuristic** (`src/heuristic/restart_heuristics.py` +
  `Verifier._check_restart`, `verifier.py:341-384`): triggers a restart with
  a *different, usually more powerful* strategy (escalating through
  `HIDDEN_SPLIT_RESTART_STRATEGIES` / `INPUT_SPLIT_RESTART_STRATEGIES`, e.g.
  plain `backward` → `crown-optimized` with a wider top-k) when: elapsed
  time exceeds a fraction of the total budget
  (`restart_max_runtime_percentage`), a fixed wall-clock threshold is hit
  (`restart_max_runtime`), or the number of remaining/visited branches
  exceeds configured caps (`restart_current_*_branches`,
  `restart_visited_*_branches`). Disabled entirely for short timeouts
  (`Verifier._heuristic_configure`, `verifier.py:145-148`) or via
  `--disable_restart`.
- **Domain management is a LIFO stack, not a global priority queue**
  (`helper/misc/tensor_storage.py` `TensorStorage.pop`/`append` — pop from
  and push to the same end) — so within a batch, search is depth-first over
  recently created branches; breadth comes from processing a whole `--batch`
  of open domains per BaB iteration together (parallel/batched BaB), not
  from a best-first global ordering. `DomainsList.minimum_lowers` /
  `pick_out_worst_domains` (used for the MIP-attack step and early stopping)
  do sort by bound when a "worst-first" view is specifically needed.

### 8.5 Adversarial attacks (fast path to `SAT`)

Run **before** BaB starts and periodically **during** it
(`Settings.attack_interval`, `verifier.py:437`), because finding a
counterexample is far cheaper than proving none exists:
- `RandomAttacker` (`attacker/random_attack.py`) — random sampling in the
  input box.
- `PGDAttacker` (`attacker/attacker.py`, `attacker/pgd_attack/`) — projected
  gradient descent, multi-restart, against all DNF objectives simultaneously.
- MIP attack (`attacker/mip_attack.py`) — Gurobi-based search seeded from
  the current worst branch-and-bound domains (`domains_list.pick_out_worst_domains`),
  gated by `Settings.use_mip_attack` (marked experimental/in-progress).
- Every candidate counterexample is validated against the true ONNX/PyTorch
  model (`helper/misc/check.py:check_solution`) before being accepted as
  `SAT`, guarding against numerical false positives from the attack itself.

### 8.6 Scaling to large networks: decompositional verification

For networks too large to bound/branch on in one shot (`main_dec.py`,
`verifier/dec_verifier.py`, `DecompositionalVerifier`): first tries ordinary
one-shot verification (`original_verify`), and on CUDA OOM (or per a
category's tuned profile in `configure/advanced.py:DecompositionSettings`)
falls back to `decompositional_verify`, which splits the network into
sub-networks and verifies them compositionally with tunable candidate/batch
counts and per-stage timeouts (`verify_candidate_batch`,
`verify_interm_timeout`, `verify_last_timeout`, `sequential_batch`) —
pre-tuned per architecture family (`resnet6/12/18/36`, `vae_base/wide/deep`).

### 8.7 Other performance techniques

- **Adaptive batch size**: shrinks the BaB batch on CUDA OOM
  (`verifier.py:190-200`) and grows it opportunistically when GPU memory
  headroom allows (`verifier.py:472-486`).
- **Shared/pruned slopes**: `share_alphas` setting reduces α-CROWN memory by
  sharing relaxation slopes across specs; `new_slopes()` prunes slopes not
  needed for hidden splitting to cut memory (`verifier.py:251`).
- **Early stopping**: `skip_initial_worst_bound`, `max_iterations`,
  `max_domains` bound the search when it's clearly not converging
  (`verifier.py:301-335`).
- **float64 proof mode**: when `use_save_reasoning_step` is set, switches
  the whole pipeline to float64 (`setting.py:92-94`) for numerically
  verifiable exported proofs (APTP format, `helper/proof/`), at a
  performance cost.

## 9. Web UI production deployment and CI/CD

This section documents how `roars.dev/neuralsat` (the web demo, §5) is
actually deployed — split across two independent pieces that know nothing
about each other's deploy mechanism.

### 9.1 Two independent deploy targets

| Piece | What | How it's served | How it deploys |
|---|---|---|---|
| **Frontend** | `web/index.html`, published to the `gh-pages` branch (root) | GitHub Pages, configured to serve the `gh-pages` branch. Reachable at `roars.dev/neuralsat` because the org's user/org Pages site (a separate repo) owns the custom domain `roars.dev`, and GitHub Pages cascades that domain to project pages under the same account. | **Automated** (§9.3): a GitHub Actions job publishes on every push to `develop`, mirroring `dynaroars/dig`'s pattern. Previously this was a manually-copied `docs/index.html` on the `develop` branch, which went stale twice before being replaced by this automation. |
| **Backend** | `web/server.py` (Flask, gunicorn) + the `src/` verifier it shells out to | Runs as a systemd service on a machine called **`taco`**, under a **`webapp`** user account, port 5050, tunneled *directly* to the public internet via `ngrok` (domain `shingle-unhinge-concert.ngrok-free.dev`, dedicated to NeuralSAT — no nginx in the path) since `taco` isn't otherwise publicly routed on that port. This mirrors `dynaroars/dig`'s own `dig-ngrok-tunnel.service`, which tunnels straight to gunicorn the same way. `taco` does have a real GPU (`NVIDIA GeForce RTX 3080 Ti` — visible via `/api/health`'s `gpu` field once GPU auto-detect, §"web UI" work, was added). | **Automated** (§9.2): GitHub Actions deploys on every push to `develop`. Not automated: any change to the systemd config file itself (§9.4). |

`taco`'s backend and ngrok tunnel were originally deployed under a user
`azan`; both are now fully migrated to run as `webapp` (backend in commit
`065eb64`'s doc fix + prior manual migration; the `ngrok-tunnel.service` →
`neuralsat-ngrok-tunnel.service` cutover, then run under `webapp` sharing
domain `oarless-chafflike-chung.ngrok-free.dev` with an unrelated
"CS Scheduler" service, fanned out by a checked-in `nginx`/`nginx-ngrok`
config that lived at `web/nginx.conf`/`web/nginx-ngrok.conf`).

That shared-domain arrangement has since been retired for NeuralSAT: the
tunnel now points at a **domain dedicated to NeuralSAT alone**
(`shingle-unhinge-concert.ngrok-free.dev`), `ngrok http 5050` straight to
gunicorn, no nginx involved — the same shape as `dynaroars/dig`'s own
tunnel. `web/nginx.conf` and `web/nginx-ngrok.conf` have been deleted from
this repo; CS Scheduler's continued public access (previously riding on
the same tunnel) is being handled independently of NeuralSAT's deploy
pipeline and isn't tracked here. The `web/*.service` file checked into
this repo is **reference/setup docs only** — nothing deploys it
automatically, so it can still drift from the live `taco` state if
changed by hand without a matching repo update (as happened once already,
fixed in `065eb64`). Don't trust it as ground truth for current deployment
state without checking; it's a template for manual `systemctl` setup, not
live configuration.

### 9.2 Automated backend deploy (`.github/workflows/deploy.yml`, job `deploy-backend`)

On every push to `develop` (or manual `workflow_dispatch`), a GitHub Actions
job:
1. Writes the `TACO_DEPLOY_SSH_KEY` repository secret to a key file and
   pins `taco.roars.dev`'s host key (hardcoded in the workflow, not
   `ssh-keyscan`'d at CI time, to avoid trusting the network for that).
2. `ssh -tt` (pty forced — required because `sudo` on `taco` has `use_pty`
   set) into `webapp@taco.roars.dev`.
3. That SSH key is **restricted server-side** via a forced command in
   `webapp`'s `~/.ssh/authorized_keys`:
   ```
   command="/home/webapp/neuralsat/web/deploy-backend.sh",no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-ed25519 ... neuralsat-ci-deploy
   ```
   No matter what the workflow (or a leaked secret) tries to run, only
   `web/deploy-backend.sh` ever executes.
4. `web/deploy-backend.sh` does exactly two things: `git pull origin develop`, then
   `sudo systemctl restart neuralsat-backend`. The restart is passwordless
   because of a **scoped sudoers NOPASSWD rule** for `webapp` limited to
   exactly `systemctl restart neuralsat-backend` (`sudo -l` on `taco` shows
   it, alongside equivalent grants for a second project, DIG, sharing the
   same host/user — `dig-backend`, `dig-ngrok-tunnel`).

Key lessons from standing this up (in case it breaks again):
- The `authorized_keys` `command=` value **must be quoted**
  (`command="/path",...`) — an unquoted value caused this sshd to reject
  the key outright at the pre-auth probe stage (before ever checking the
  signature), which looks identical to "wrong key" but isn't.
- Don't `scp` the deploy script onto the target ahead of the first
  automated pull as an untracked file — `git pull` refuses to fast-forward
  over an untracked file at a path the incoming commit also touches
  (`error: ... would be overwritten by merge`). Remove the untracked copy
  (after confirming it's byte-identical to what's committed) before the
  first real pull.
- Never leave an unrestricted duplicate of a scoped deploy key in
  `authorized_keys` "just for testing" — sshd matches the *first* line with
  a given key blob, so an unrestricted duplicate added *after* the
  restricted one is currently inert, but that's fragile, not a security
  boundary. Delete it.

### 9.3 Automated frontend deploy (`.github/workflows/deploy.yml`, job `deploy-frontend`)

Runs in the same workflow, independently of the backend job (no SSH/`taco`
involved at all — it only pushes within this repo):
1. Full checkout (`fetch-depth: 0`) of `develop`.
2. `web/deploy-frontend.sh` — mirrors `dynaroars/dig`'s `web/deploy-frontend.sh`
   exactly (both repos use identical `deploy-backend.sh`/`deploy-frontend.sh`
   naming, kept in sync deliberately): builds a flat git tree (via
   `git hash-object` + `git mktree`)
   containing just `web/*.html` (basenames only, no `web/` prefix, so
   files land at the `gh-pages` branch *root*), compares its hash to
   `origin/gh-pages`'s current tree, and — if
   different — creates a new commit on top of `origin/gh-pages` and pushes
   it. If unchanged, it's a no-op (`"gh-pages already up to date."`).
3. Pushing to `gh-pages` uses the default `GITHUB_TOKEN` (job declares
   `permissions: contents: write`) — no extra secret needed, since this
   never leaves GitHub's own infrastructure, unlike the backend job.

The `gh-pages` branch itself was bootstrapped once by hand (a root commit
with the same flat-tree layout) before this automation existed; from that
point on `deploy-frontend.sh`'s assumption that `origin/gh-pages` already
exists holds for every future run.

### 9.4 What CI/CD does *not* cover

- `web/*.service` — manual, and reference-only as noted above; changing
  the real systemd units on `taco` requires editing
  `/etc/systemd/system/*.service` there directly.
- Anything on the DIG side beyond its own analogous pipeline (separate repo
  `dynaroars/dig`, separate deploy key, same `web/deploy-backend.sh` /
  `web/deploy-frontend.sh` naming as this repo — deliberately kept
  identical across both projects so either is easy to reason about once
  you understand the other; triggers on push to `dev` not `develop`) — DIG
  and NeuralSAT share the `taco`/`webapp` host but have fully independent
  deploy keys, scripts, and workflows; a compromised or malfunctioning key
  for one cannot affect the other's service (each forced command is scoped
  to that project's own script and systemd unit).
- `web/neuralsat-ngrok-tunnel.service` — renamed from `ngrok-tunnel.service`
  to match DIG's `dig-ngrok-tunnel.service` convention, and **migrated off
  `azan` onto `webapp`** (completed and verified live). Cutover was a
  manual root-level job on `taco` (stop/disable the old unit,
  install/enable the new one, `daemon-reload`, plus a new scoped
  `NOPASSWD` sudoers rule so `webapp` can restart it going forward,
  mirroring `dig-ngrok-tunnel`'s grant) — CI never touches this service,
  unlike the backend/frontend deploy jobs. The old `ngrok-tunnel.service`
  unit is stopped and disabled on `taco` (removal of the file itself is a
  separate, optional cleanup step).

  It has since been repointed a second time, off the domain shared with
  CS Scheduler (`oarless-chafflike-chung.ngrok-free.dev`, fanned out via
  nginx) onto `shingle-unhinge-concert.ngrok-free.dev`, reserved
  specifically for NeuralSAT, with `ExecStart` now `ngrok http 5050`
  (straight to gunicorn — matching `dig-ngrok-tunnel.service` exactly,
  no nginx). This repo change (and the matching `NGROK_API` constant in
  `web/index.html`) is committed; the domain reservation itself and the
  swap of the live systemd unit on `taco` are manual steps outside CI,
  same as any other `web/*.service` change. CS Scheduler's continued
  public access after losing the shared tunnel is being handled
  separately, outside NeuralSAT's deploy pipeline.

# NeuralSAT Installation and Usage

> While NeuralSAT can be installed and run on any platforms satisfying its [dependencies](#installation), we mainly develop and test NeuralSAT on Linux.


## Content

- ```src```: source code
- ```third_party```: external libraries
- ```vnncomp_scripts```: scripts for competition


## Installation

### Prerequisites

- **Python 3.10 or 3.11** (recommended for best compatibility; Python 3.13+ may have package compatibility issues)
- Git (for cloning the repository)
- [Gurobi](https://www.gurobi.com/): Gurobi requires a license (a [free academic license](https://www.gurobi.com/downloads/free-academic-license/) is available).

### Setup


#### Step 1: Clone the repository

```bash
git clone https://github.com/dynaroars/neuralsat.git
cd neuralsat
```

#### Step 2: Choose your installation method

You can install NeuralSAT using either **pip** (recommended), **conda** with conda-forge or **uv**.

#### Option 1: PyPI installation

You can directory install all required packages by running:
```bash
pip install neuralsat
```



#### Option 2: pip installation 

- (Optional) Remove any pre-existing virtual environment

```bash
rm -rf neuralsat_env
```

- Create a Python virtual environment

```bash
python3 -m venv neuralsat_env
```

- Activate the virtual environment

```bash
# On Linux/Mac:
source neuralsat_env/bin/activate

# On Windows:
neuralsat_env\Scripts\activate
```

- Install required packages

```bash
pip install -r requirements.txt
```

- Gurobi issues? If gurobipy fails during pip install, run it separately:

```bash
pip install gurobipy
```

- (Optional) Install specific Pytorch C++/CUDA extensions

```bash
pip install "third_party/haioc"
```

- (Optional) Install `DNNV` for ONNX simplification

```bash

python3 -m venv dnnv_env
source dnnv_env/bin/activate 
pip install --no-deps git+https://github.com/dlshriver/DNNV.git@d4f59a01810cf4dac99f8f5e5b9d7a350cbfa8d7#egg=dnnv
deactivate
```

#### Option 3: conda with conda-forge

If you prefer conda, use conda-forge channels (free, no licensing restrictions):

- (Optional) Remove any pre-existing NeuralSAT environment

```bash
conda deactivate
conda env remove --name neuralsat
```

- Create a new conda environment named "neuralsat"

```bash
conda create -n neuralsat python=3.10 -c conda-forge
```

- Activate the newly created environment

```bash
conda activate neuralsat
```

- Install required packages from conda-forge and pip

```bash
# For GPU support (CUDA 11.8 - check your CUDA version with nvidia-smi):
conda install -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install -c conda-forge numpy scipy onnx onnxruntime pillow

# Install C++ standard library only if CXXABI errors occur
conda install -c conda-forge libstdcxx-ng

# Install Gurobi (requires separate license)
conda install -c gurobi gurobi

# Install remaining packages via pip
pip install -r requirements.txt
```

**Note:** Check your CUDA version with `nvidia-smi`. For CUDA 12.x use `pytorch-cuda=12.1`, for CUDA 11.x use `pytorch-cuda=11.8`.


#### Option 4: UV installation

If you need to manage specific Python versions (e.g., Python 3.10 for onnxsim compatibility), use [uv](https://docs.astral.sh/uv/):

- Run the setup script

```bash
./setup.sh
# Or alternatively:
bash setup.sh
```

### Linux (Ubuntu/Debian)

**System dependencies** (for PyTorch/ONNX builds):
```bash
sudo apt-get update
sudo apt-get install libbz2-dev lzma liblzma-dev
```


## 🚀 Usage

Run NeuralSAT from the repository root directory:

**If you installed with uv:**
```bash
uv run src/main.py [-h] --net NET --spec SPEC
```

**If you installed with pip or conda:**
```bash
python3 src/main.py [-h] --net NET --spec SPEC 
        [--batch BATCH] [--timeout TIMEOUT] [--device {cpu,cuda}] [--verbosity {0,1,2}] 
        [--result_file RESULT_FILE] [--export_cex] 
        [--disable_restart] [--disable_stabilize] 
```

### Options
Use ```-h``` or ```--help``` to see options that can be passed into **NeuralSAT**. 

- `--net`: Path to `ONNX` model.
- `--spec`: Path to `VNNLIB` specification file.
- `--batch`: Maximum number of parallel checking branches.
- `--timeout`: Timeout (in second) for verifying one instance.
- `--device`: Device to use (either `cpu` or `cuda`).
- `--verbosity`: Logging options (0: NOTSET, 1: INFO, 2: DEBUG).
- `--result_file`: File to export execution results (including counter-example if found).
- `--export_cex`: Enable writing counter-example to `result_file`.
- `--disable_restart`: disable RESTART heuristic.
- `--disable_stabilize`: disable STABILIZE.


### Examples

**Note:** If you installed with uv, replace `python3` with `uv run` in all examples below.

- Examples showing **NeuralSAT** verifies properties (i.e., returning `unsat``):

```bash
python3 main.py --net "example/onnx/mnistfc-medium-net-554.onnx" --spec "example/vnnlib/test.vnnlib"
# unsat,24.9284
```

```bash
python3 main.py --net "example/onnx/cifar10_2_255_simplified.onnx" --spec "example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib"
# unsat,17.9806
```

```bash
python3 main.py --net "example/onnx/ACASXU_run2a_1_1_batch_2000.onnx" --spec "example/vnnlib/prop_6.vnnlib" --disable_restart
# unsat,3.0907
```

```bash
python3 main.py --net "example/onnx/mnist-net_256x4.onnx" --spec "example/vnnlib/prop_1_0.03.vnnlib"
# unsat,139.4125
```

- Examples showing **NeuralSAT** disproves properties (i.e., returning `sat` and counterexample):

```bash
python3 main.py --net "example/onnx/mnist-net_256x2.onnx" --spec "example/vnnlib/prop_1_0.05.vnnlib"
# sat,0.7526
```

```bash
python3 main.py --net "example/onnx/ACASXU_run2a_1_9_batch_2000.onnx" --spec "example/vnnlib/prop_7.vnnlib" --disable_restart
# sat,6.1320
```

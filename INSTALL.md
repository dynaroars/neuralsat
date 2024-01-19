# NeuralSAT Installation and Usage

> While NeuralSAT can be installed and run on any platforms satisfying its [dependencies](#installation), we mainly develop and test NeuralSAT on Linux.


## Content

- ```neuralsat-pt201```: source code
- ```third_party```: external libraries
- ```vnncomp_scripts```: scripts for competition


## Installation

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Gurobi](https://www.gurobi.com/): Gurobi requires a license (a [free academic license](https://www.gurobi.com/downloads/free-academic-license/) is available).

- Remove pre-installed environment 

```bash
conda deactivate; conda env remove --name neuralsat
```

- Install required packages 

```bash
conda env create -f env.yaml
```

- Activate `conda` environment

```bash
conda activate neuralsat
```

- (Optional) Install specific Pytorch C++/CUDA extensions

```bash
pip install "third_party/haioc"
```

- (Optional) Install `DNNV` for ONNX simplification

```bash
conda deactivate; conda env remove --name dnnv
conda env create -f neuralsat-pt201/installation/env_dnnv.yaml
conda activate dnnv
pip install --no-deps git+https://github.com/dlshriver/DNNV.git@d4f59a01810cf4dac99f8f5e5b9d7a350cbfa8d7#egg=dnnv
```

## 🚀 Usage

```python
main.py [-h] --net NET --spec SPEC 
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

- Examples showing **NeuralSAT** verifies properties (i.e., returning `unsat``):

```python
python3 main.py --net "example/onnx/mnistfc-medium-net-554.onnx" --spec "example/vnnlib/test.vnnlib"
# unsat,24.9284
```

```python
python3 main.py --net "example/onnx/cifar10_2_255_simplified.onnx" --spec "example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib"
# unsat,17.9806
```

```python
python3 main.py --net "example/onnx/ACASXU_run2a_1_1_batch_2000.onnx" --spec "example/vnnlib/prop_6.vnnlib" --disable_restart
# unsat,3.0907
```


- Examples showing **NeuralSAT** disproves properties (i.e., returning `sat` and counterexample):

```python
python3 main.py --net "example/onnx/mnist-net_256x2.onnx" --spec "example/vnnlib/prop_1_0.05.vnnlib"
# sat,0.7526
```

```python
python3 main.py --net "example/onnx/ACASXU_run2a_1_9_batch_2000.onnx" --spec "example/vnnlib/prop_7.vnnlib" --disable_restart
# sat,6.1320
```




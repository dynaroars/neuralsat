# NeuralSAT **Installation and Usage

> While NeuralSAT can be installed and run on any platforms satisfying its [dependencies](#dependencies), we mainly develop and test NeuralSAT on Linux.


## Content

- ```neuralsat-pt201```: source code
- ```third_party```: external libraries
- ```vnncomp_scripts```: scripts for competition


## Installation


### Dependencies
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Gurobi](https://www.gurobi.com/): Gurobi requires a license (a [free academic license](https://www.gurobi.com/downloads/free-academic-license/) is available).

## Installation
- Make sure you have `Anaconda`/`Miniconda` and `Gurobi` properly installed.

- Remove pre-installed environment 

```bash
conda deactivate 
conda env remove --name neuralsat
```

- Install required packages 

```bash
conda env create -f env.yaml
```

- (Optional) Install specific Pytorch C++/CUDA extensions

```bash
pip install "third_party/haioc"
```


## 🚀 Usage

- Activate `conda` environment

```bash
conda activate neuralsat
```

- Minimal command

```python
python3 main.py --net PATH_TO_ONNX_MODEL --spec PATH_TO_VNNLIB_FILE
```

- More options

```python
python3 main.py --net PATH_TO_VNNLIB_FILE --spec PATH_TO_VNNLIB_FILE
               [--batch BATCH] [--timeout TIMEOUT] [--device {cpu,cuda}]
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


### Examples

- Examples showing **NeuralSAT** verifies properties (i.e., returning `unsat``):

```python
python3 main.py --net "example/mnistfc-medium-net-554.onnx" --spec "example/test.vnnlib"
# unsat,29.7011
```

```python
python3 main.py --net "example/cifar10_2_255_simplified.onnx" --spec "example/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib"
# unsat,20.0496
```

```python
python3 main.py --net "example/ACASXU_run2a_1_1_batch_2000.onnx" --spec "example/prop_6.vnnlib"
# unsat,4.3972
```


- Examples showing **NeuralSAT** disproves properties (i.e., returning `sat` and counterexample):

```python
python3 main.py --net "example/ACASXU_run2a_1_9_batch_2000.onnx" --spec "example/prop_7.vnnlib"
# sat,3.6618
# adv (first 5): tensor([-0.3284, -0.4299, -0.4991,  0.0000,  0.0156])
```

```python
python3 main.py --net "example/mnist-net_256x2.onnx" --spec "example/prop_1_0.05.vnnlib"
# sat,1.4306
# adv (first 5): tensor([0.0000, 0.0500, 0.0500, 0.0000, 0.0500])
```



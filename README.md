**NeuralSAT**: A DPLL(T)-based Constraint Solving Approach to Verifying Deep Neural Networks
====================

**NeuralSAT** is a technique and prototype tool for verifying DNNs.  It combines ideas from DPLL(T)/CDCL algorithms in SAT/SMT solving with a abstraction-based theory solver to reason about DNN properties. The tool is under active development, and periodically we evaluate the tool on existing standard benchmarks such as `ACAS Xu`, `MNISTFC`, `CIFAR2020` and compare the performance of the prototype to other state-of-the-art DNN verifiers.

**NeuralSAT** takes as input the formula $\alpha$ representing the DNN `N` (with non-linear ReLU activation) and the formulae $\phi_{in}\Rightarrow \phi_{out}$ representing the property $\phi$ to be proved. Internally, **NeuralSAT** checks the satisfiability of the formula: $\alpha \land \phi_{in} \land \overline{\phi_{out}}$. **NeuralSAT** returns *`UNSAT`* if the formula is not satisfiable, indicating  `N` satisfies $\phi$, and *`SAT`* if the formula is satisfiable, indicating the `N` does not satisfy $\phi$.

**NeuralSAT** uses a  DPLL(T)-based algorithm to check unsatisfiability. **NeuralSAT** applies DPLL/CDCL to assign values to boolean variables and checks for conflicts the assignment has with the real-valued constraints of the DNN and the property of interest. If conflicts arise, **NeuralSAT** determines the assignment decisions causing the conflicts, backtracks to erase such decisions, and learns clauses to avoid those decisions in the future. **NeuralSAT** repeats these decisions and checking steps until it finds a full assignment for all boolean variables, in which it returns *`SAT`*, or until it no longer can backtrack, in which it returns *`UNSAT`*.

Content
====================
- ```neuralsat```: Containing source code for **NeuralSAT**.

- ```benchmark```: Containing benchmarks taken from [VNNCOMP'21](https://github.com/stanleybak/vnncomp2021).

- ```third_party```: Containing external libraries.


Getting Started
====================

## Dependencies
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Gurobi](https://www.gurobi.com/): Gurobi requires a license (a [free academic license](https://www.gurobi.com/downloads/free-academic-license/) is available).

## Installation
- Make sure you have `Anaconda`/`Miniconda` and `Gurobi` properly installed.
- (Optional) Run `conda deactivate; conda env remove --name neuralsat` to remove installed environment
- Run `conda env create -f env.yaml` to install required packages.

## Usages

- Activate `conda` environment

```bash
conda activate neuralsat
```

- Navigate to **NeuralSAT** folder.

```bash
cd neuralsat
```

- Minimal command

```python
python3 main.py --net ONNX_PATH --spec VNNLIB_PATH
```

- More options

```python
python3 main.py --net ONNX_PATH --spec VNNLIB_PATH 
               [--batch BATCH] [--timeout TIMEOUT] [--device {cpu,cuda}]
```

## Options
Use ```-h``` or ```--help``` to see options that can be passed into **NeuralSAT**. 

- `--net`: Load pretrained `ONNX` model from this specified path.
- `--spec`: Path to `VNNLIB` specification file.
- `--batch`: Maximum number of parallel splits.
- `--timeout`: Timeout (in second) for verifying one instance.
- `--device`: Select device to run **NeuralSAT**.



## Examples

- Examples showing **NeuralSAT** verifies properties (i.e., UNSAT results):

```python
python3 main.py --net "example/mnistfc-medium-net-554.onnx" --spec "example/test.vnnlib"
# UNSAT,29.7011
```

```python
python3 main.py --net "example/cifar10_2_255_simplified.onnx" --spec "example/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib"
# UNSAT,20.0496
```

```python
python3 main.py --net "example/ACASXU_run2a_1_1_batch_2000.onnx" --spec "example/prop_6.vnnlib"
# UNSAT,4.3972
```


- Examples showing **NeuralSAT** disproving properties (i.e., SAT results):

```python
python3 main.py --net "example/ACASXU_run2a_1_9_batch_2000.onnx" --spec "example/prop_7.vnnlib"
# SAT,3.6618
# adv (first 5): tensor([-0.3284, -0.4299, -0.4991,  0.0000,  0.0156])
```

```python
python3 main.py --net "example/mnist-net_256x2.onnx" --spec "example/prop_1_0.05.vnnlib"
# SAT,1.4306
# adv (first 5): tensor([0.0000, 0.0500, 0.0500, 0.0000, 0.0500])
```

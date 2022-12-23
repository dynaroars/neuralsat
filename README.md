*NeuralSAT*: A DPLL(T)-based Constraint Solving Approach to Verifying Deep Neural Networks
====================

*NeuralSAT* is a technique and prototype tool for verifying DNNs.  It combines ideas from DPLL(T) and CDCL algorithms in SAT/SMT solving with a abstraction-based theory solver to reason about DNN properties. The tool is under active development and has not released any official versions, though periodically we evaluate the tool on existing standard benchmarks such as ACAS Xu, MNIST, CIFAR and compare the performance of the prototype to other state-of-the-art DNN verifiers.

*NeuralSAT* takes as input the formula $\alpha$ representing the DNN $N$ (with non-linear ReLU activation) and the formulae $\phi_{in}\Rightarrow \phi_{out}$ representing the property $\phi$ to be proved. Internally, *NeuralSAT* checks the satisfiability of the formula: $\alpha \land \phi_{in} \land \overline{\phi_{out}}$. *NeuralSAT* returns **UNSAT** if the formula is not satisfiable, indicating  $N$ satisfies $\phi$, and **SAT** if the formula is satisfiable, indicating the $N$ does not satisfy $\phi$.

*NeuralSAT* uses a  DPLL(T)-based algorithm to check unsatisfiability. *NeuralSAT* applies DPLL/CDCL to assign values to boolean variables and checks for conflicts the assignment has with the real-valued constraints of the DNN and the property of interest. If conflicts arise, *NeuralSAT* determines the assignment decisions causing the conflicts, backtracks to erase such decisions, and learns clauses to avoid those decisions in the future. *NeuralSAT* repeats these decisions and checking steps until it finds a full assignment for all boolean variables, in which it returns **SAT**, or until it no longer can backtrack, in which it returns **UNSAT**.

Content
====================
- ```neuralsat```: Containing source code for *NeuralSAT*.

- ```benchmark```: Containing benchmarks taken from [VNNCOMP'21](https://github.com/stanleybak/vnncomp2021).



Getting Started
====================

## Dependencies
- Python 3.9
- PyTorch
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (11.6)
- [Gurobi](https://www.gurobi.com/): Gurobi requires a license (a free academic license is available).

## Installation
- Make sure you have CUDA and Gurobi properly installed.
- Clone this repository.
- Navigate to ```neuralsat```.
- Run ```pip install -r requirements.txt``` to install required pip packages.
- Follow the instruction from [pytorch.org](https://pytorch.org/get-started/locally/) to install PyTorch.

## Usages

- Navigate to *NeuralSAT* folder.

```bash
cd neuralsat
```

- Minimal command

```python
python3 main.py --net ONNX_PATH --spec VNNLIB_PATH
```

- More options

```python
python3 main.py --net ONNX_PATH --spec VNNLIB_PATH [--device {cpu,cuda}] [--timeout TIMEOUT] [--summary OUTPUT_FILE] [--solution]
```


## Options
<!-- - talk about the flags avaliable -->
Use ```-h``` or ```--help``` to see options that can be passed into *NeuralSAT*. 

- `--net`: Load pretrained `ONNX` model from this specified path.
- `--spec`: Path to `VNNLIB` specification file.
- `--device`: Select device to run *NeuralSAT*, `cpu`/`cuda` (GPU).
- `--summary`: Path to result file (format in result file: `[STAT],[RUNTIME]`).
- `--solution`: Get a solution (counterexample) if *NeuralSAT* returns `SAT`.
- `--timeout`: Timeout (in second) for verifying one instance.

## Example

- UNSAT case

```python
python3 main.py --net ../benchmark/mnistfc/nnet/mnist-net_256x2.onnx --spec ../benchmark/mnistfc/spec/prop_0_0.03.vnnlib --device cuda
# UNSAT,4.603
```

- SAT case

```python
python3 main.py --net ../benchmark/mnistfc/nnet/mnist-net_256x2.onnx --spec ../benchmark/mnistfc/spec/prop_1_0.05.vnnlib --solution
# SAT,0.123
# adv (first 5): tensor([0.0000, 0.0000, 0.0250, 0.0125, 0.0500])
```

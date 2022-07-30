# Lagrangian Decomposition for Neural Network Bounds

This repository contains the code implementing the dual iterative algorithms for neural network bounds computations 
described in: [Lagrangian Decomposition for Neural Network Verification](https://arxiv.org/abs/2002.10410). 
If you use it in your research, please cite:

```
@Article{Bunel2020,
    title={Lagrangian Decomposition for Neural Network Verification},
    author={Bunel, Rudy and De Palma, Alessandro and Desmaison, Alban and Dvijotham, Krishnamurthy and Kohli, Pushmeet  and Torr, Philip H. S. and Kumar, M. Pawan},
    journal={Conference on Uncertainty in Artificial Intelligence},
    year={2020}
}
```

## Neural Network bounds
The repository provides code for algorithms to compute output bounds for ReLU-based neural networks (and, 
more generally, piecewise-linear networks, which can be transformed into equivalent ReLUs):
- `LinearizedNetwork` in `plnn_bounds/network_linear_approximation.py` represents the [PLANET](https://github.com/progirep/planet) relaxation of the network in Gurobi 
and uses the commercial solver to compute the model's output bounds.
- `SaddleLP` in `plnn_bounds/proxlp_solver/solver.py` implements the dual iterative algorithms presented in 
"Lagrangian Decomposition for Neural Network Verification" in PyTorch, based on the Lagrangian Decomposition of the activation's 
convex relaxations.
- `DJRelaxationLP` in `plnn_bounds/proxlp_solver/dj_relaxation.py` implements the Lagrangian relaxation-based dual iterative algorithm presented in 
"[A Dual Approach to Scalable Verification of Deep Networks](https://arxiv.org/abs/1803.06567)" in PyTorch.

These classes offer two main interfaces (see `tools/cifar_runner.py` and `tools/cifar_bound_comparison.py` for detailed 
usage, including algorithm parametrization):
- Given some pre-computed intermediate bounds, compute the bounds on the neural network output: 
call `build_model_using_bounds`, then `compute_lower_bound`.
- Compute bounds for activations of all network layers, one after the other (each layer's computation will use the 
bounds computed for the previous one): `define_linear_approximation`.

The computed neural network bounds can be employed in two different ways: alone, to perform incomplete 
verification; as the bounding part of branch and bound (to perform complete verification).

## Implementation details

The dual iterative algorithms (`SaddleLP`, `DJRelaxationLP`) **batch** the computations of each layer output 
lower/upper bounds in order to compute them in parallel. 

## Repository structure
* `./plnn_bounds/` contains the code for the dual iterative algorithms described above.
* `./tools/` contains code to interface the bounds computation classes, run experiments on CIFAR10, and
analyse their results. In particular, in addition to the paper's incomplete verification experiments, 
`tools/cifar_runner.py` runs the bounding algorithms on single CIFAR10 images, logging the optimization progress, which 
can be plotted via `tools/tuning_plotting.py`. 
* `./scripts/` is a set of bash/python scripts, instrumenting the tools of `./tools` to
  reproduce the results of the paper.
* `./networks/` contains the two trained networks employed in the incomplete verification part of the paper.
  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 

  
### Installation
We assume the user's Python environment is based on Anaconda.

```bash
git clone --recursive https://github.com/oval-group/decomposition-plnn-bounds.git

cd decomposition-plnn-bounds

# Install gurobipy 
conda config --add channels http://conda.anaconda.org/gurobi
python setup.py install

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch 

# Install the code of this repository
python setup.py install
```

### Running the experiments
If you have setup everything according to the previous instructions, you should
be able to replicate the incomplete verification experiments of the paper:

```bash
## Execute incomplete verification experiments (edit hardware-specific lines 23-24)   
python ./scripts/run_incomplete_verification.py

## Analyse the results
# (might have to `pip install matplotlib` to generate curves)
./scripts/plot_incomplete_verification.sh
```

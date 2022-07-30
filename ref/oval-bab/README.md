# OVAL - Branch-and-Bound-based Neural Network Verification

This repository contains PyTorch code for state-of-the-art GPU-accelerated neural network verification based on 
Branch and Bound (BaB), developed by the [OVAL](https://www.robots.ox.ac.uk/~oval/) research group at the University of Oxford. 
See the "publications" section below for references. 

Complete neural network verification can be cast as a non-convex global minimization problem, which can be solved via BaB.
BaB operates by computing lower and upper bounds to the global minimum, which can be iteratively tightened by dividing the feasible domain into subproblems.
Lower bounds can be obtained by solving a convex relaxation of the network via an *incomplete verifier*. Upper bounds are obtained via *falsification algorithms* 
(heuristics, such as adversarial attacks for adversarial robustness properties). 
The *branching strategy*, the algorithm employed to create the subproblems, strongly influences the tightness of the lower bounds. 
  

### Incomplete Verifiers
The OVAL framework currently includes the following incomplete verifiers for piecewise-linear networks:
#### Loose yet fast
- IBP ([Gowal et al. 2018](https://arxiv.org/pdf/1810.12715.pdf)) (`Propagation` in `plnn/proxlp_solver/propagation.py`).
- Propagation-based algorithms: CROWN ([Zhang et al. 2018](https://arxiv.org/abs/1811.00866)), Fast-Lin ([Wong and Kolter 2018](https://arxiv.org/abs/1711.00851)) 
(`Propagation` in `plnn/proxlp_solver/propagation.py`).
#### Convex hull of element-wise activations
- Dual solvers for the convex hull of element-wise activations, based on Lagrangian Decomposition ([Bunel et al. 2020b](https://arxiv.org/abs/2002.10410)) (`SaddleLP` in `plnn/proxlp_solver/solver.py`).
- Gurobi solver for the Planet ([Ehlers 2017](https://arxiv.org/abs/1705.01320)) relaxation (`LinearizedNetwork` in `plnn/network_linear_approximation.py`). 
- Beta-CROWN ([Wang et al. 2021](https://arxiv.org/abs/2103.06624)), a state-of-the-art solver for the Planet relaxation (`Propagation` in `plnn/proxlp_solver/propagation.py`).
- DeepVerify ([Dvijotham et al. 2018](https://arxiv.org/abs/1803.06567)) (`DJRelaxationLP` in `plnn/proxlp_solver/dj_relaxation.py`).
#### Tighter bounds
- Active Set ([De Palma et al. 2021c](https://openreview.net/forum?id=uQfOy7LrlTR)), and Saddle Point ([De Palma et al. 2021a](https://arxiv.org/abs/2101.05844)), state-of-the-art fast dual solvers for the tighter linear relaxation by 
[Anderson et al. (2020)](https://arxiv.org/abs/1811.01988) (`ExpLP` in `plnn/explp_solver/solver.py`).
- Gurobi cutting-plane solver for the relaxation by [Anderson et al. (2020)](https://arxiv.org/abs/1811.01988) 
(`AndersonLinearizedNetwork` in `plnn/anderson_linear_approximation.py`).

These incomplete verifiers can be employed both to compute bounds on the networks' intermediate activations 
(intermediate bounds), or lower bounds on the network's output.
Two main interfaces are available:
- Given some pre-computed intermediate bounds, compute the bounds on the neural network output: 
call `build_model_using_bounds`, then `compute_lower_bound`.
- Compute bounds for activations of all network layers, one after the other (each layer's computation will use the 
bounds computed for the previous one): `define_linear_approximation`.

Incomplete verifiers of different bounding tightness and cost can be combined by relying on stratification (section 6.3 of [De Palma et al. (2021a)](https://arxiv.org/abs/2101.05844)). 

### Branching Strategies for Complete Verification

Subdomains are created by splitting the input domain or by splitting piecewise-linear activations into their linear components. The following two branching
strategies are available (`plnn/branch_and_bound/branching_scores.py`):
- The inexpensive input splitting heuristic from "BaB" in [Bunel et al. (2020a)](http://www.jmlr.org/papers/v21/19-468.html).
- The SR activation splitting heuristic from from "BaBSR" in [Bunel et al. (2020a)](http://www.jmlr.org/papers/v21/19-468.html).
- FSB ([De Palma et al. 2021b](https://arxiv.org/abs/2104.06718)): a state-of-the-art activation splitting strategy combining SR with inexpensive strong branching to significantly reduce verification times.  
 
### Falsification Algorithms (upper bounding)

Upper bounding is performed using a generalization of the MI-FGSM attack ([Dong et al. 2017](https://arxiv.org/abs/1710.06081)) to general property falsification (`MI_FGSM_Attack_CAN` in `adv_exp/mi_fgsm_attack_canonical_form.py`).


## Publications

The OVAL framework was developed as part of the following publications:
- ["A Unified View of Piecewise Linear Neural Network Verification"](https://arxiv.org/abs/1711.00455);
- ["Branch and Bound for Piecewise Linear Neural Network Verification"](http://www.jmlr.org/papers/v21/19-468.html);
- ["Lagrangian Decomposition for Neural Network Verification"](https://arxiv.org/abs/2002.10410);
- ["Scaling the Convex Barrier with Active Sets"](https://openreview.net/forum?id=uQfOy7LrlTR);
- ["Scaling the Convex Barrier with Sparse Dual Algorithms"](https://arxiv.org/abs/2101.05844);
- ["Improved Branch and Bound for Neural Network Verification via Lagrangian Decomposition"](https://arxiv.org/abs/2104.06718).

If you use our code in your research, please cite the papers associated to the employed algorithms, along with 
([Bunel et al. 2020a](http://www.jmlr.org/papers/v21/19-468.html)) and 
([De Palma et al. 2021b](https://arxiv.org/abs/2104.06718)) for the BaB framework.

Additionally, learning-based algorithms for branching and upper bounding (not included in the framework) were presented in:
- ["Neural Network Branching for Neural Network Verification"](https://arxiv.org/abs/1912.01329);
- ["Generating Adversarial Examples with Graph Neural Networks"](https://arxiv.org/abs/2105.14644).

  
## Running the code

The repository provides two main functionalities: incomplete verification (bounding) and complete verification (branch and bound).
We now detail how to run the OVAL framework in each of the two cases.

### Complete verification

**File `./tools/local_robustness_from_onnx.py` provides a script to verify that an `.onnx` network is robust to perturbations
in `l_inf` norm to its input points (local robustness verification).**
The comments in the file indicate which blocks should be modified to perform verification of different input-output 
specifications (which will need to be represented in canonical form, see section 2.1 of [De Palma et al. (2021b)](https://arxiv.org/abs/2104.06718)), 
and detail its input interface.

Note that an `.onnx` input is not necessary: the user can alternatively specify a piecewise-linear network by directly 
passing a list of layers to the `bab_from_json` function, as long as the network is in canonical form.

The configuration/parameters of OVAL BaB are passed via a `.json` configuration file (see those in `./bab_configs/`).

As an example, execute the following command:
```
python tools/local_robustness_from_onnx.py --network_filename ./models/onnx/cifar_base_kw.onnx
```

*Details on how to produce a .json configuration file (see `./bab_configs/`) for the OVAL framework will be soon added to the repository*

### Incomplete verification

All the bounding algorithms contained in the repository share a common interface (see "Incomplete Verifiers" above).
**File `./tools/bounds_from_onnx.py` provides a guide to the interface and usage of the bounding (incomplete verification) algorithms.**   
The guide supports bounding computations for a loaded `.onnx` network, or for small randomly sampled networks. 
As for the complete verification use-case, the user can modify the code to pass a piecewise-linear network as a list of PyTorch layers.
In addition, the file contains an example on how to directly run branch and bound without resorting to .json 
configuration files. In this case, a small timeout is set to compute tighter bounds on the first output neuron of the loaded network.

As an example, execute the following command:
```
python tools/bounds_from_onnx.py --network_filename ./models/onnx/cifar_base_kw.onnx
```


### VNN-COMP
Scripts to run the code within the context of [VNNCOMP2021](https://github.com/stanleybak/vnncomp2021) are contained in `./vnncomp_scripts/`.
Verification on a property expressed in the [vnnlib](http://www.vnnlib.org/) format 
(ONNX network for the network + .vnnlib file for the property) --or on a list of properties in a .csv file-- 
can be executed by running `./tools/bab_tools/bab_from_vnnlib.py` (check the code for details).

In addition, code to run the framework on the OVAL and COLT datasets from VNNCOMP 2020 
(see section 8.1 of [De Palma et al. (2021b)](https://arxiv.org/abs/2104.06718)) is contained in `./tools/bab_tools/bab_runner.py`.

### Supported network structures

The network to be verified/bounded (either via `.onnx` parsing or directly from a list of `torch` layers) must have the following form, 
where optional layers are enclosed in brackets:

    ["pre-layer transforms"] (this includes additions, multiplications --this covers normalization--, reshapings, flattenings, etc) -> 
    linear (nn.Linear or nn.Conv2d) -> ReLU() -> ["pre-layer transforms"] -> linear -> ReLU() -> ["pre-layer transforms"] -> [linear]
  
## Installation
The code was implemented assuming to be run under `python3.6`.
We assume the user's Python environment is based on Anaconda.

Installation steps (see also `vnncomp_scripts/install_tool.sh`):
```bash
git clone --recursive https://github.com/oval-group/oval-bab.git

cd oval-bab

# Create a conda environment
conda create -y -n oval python=3.6
conda activate oval

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
conda install -y pytorch torchvision cudatoolkit -c pytorch 

# Install the code of this repository
pip install .
```

#### Optional: Gurobi
In order to run the Gurobi-based solvers (outperformed by dual algorithms on networks with more than 1k neurons), 
a Gurobi license and installation is required.

Gurobi can be obtained from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
Note that to run the code for VNN-COMP '21, an installation of Gurobi is not needed. 

```bash
# Install gurobipy 
conda config --add channels http://conda.anaconda.org/gurobi
pip install .
# might need
# conda install gurobi
```  
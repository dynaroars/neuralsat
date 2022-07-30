## Fast and Complete: Enabling *Complete* Neural Network Verification with *Rapid* and Massively Parallel *Incomplete* Verifiers

Fast-and-Complete is an efficient and GPU based verifier for formally checking
the correctness or robustness of neural networks. It empolys an efficient
backward bound propagation algorithm
([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[LiRPA](https://arxiv.org/pdf/2002.12920.pdf))
with optimized bounds as a base incomplete solver, allowing massively parallel
acceleration on GPUs, and combines CROWN with *batch* branch and bound (BaB).
Unlike most existing neural network verifiers, Fast-and-Complete can eliminate
most cost of solving slow linear program (LP) problems on CPU, and can achieve
one or two orders of magnitudes speedup compared to traditional CPU based
verifiers.

<p align="center">
<img src="http://huan-zhang.com/images/paper/lirpa_verify.png" width="100%" height="100%">
</p>

Please refer to our paper for more details:

[Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers](https://arxiv.org/pdf/2011.13824.pdf)  
ICLR 2021  
**Kaidi Xu\*, Huan Zhang\*, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin and Cho-Jui Hsieh**  
(\* Equal contribution)

<img align="left" width="6%" height="4%" src="https://huan-zhang.com/images/new_logo.png">

**Please also checkout our new work [*β*-CROWN](https://github.com/KaidiXu/Beta-CROWN)**, which encodes the split constraints in branch and bound into CROWN and fully removes the use of LP solvers in complete NN verification, allowing us to further tighten the bound and fully exploit GPU acceleration.

### Installation

Our implementation utilizes the [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA)
library which is a high quality implementation of
[CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN) and other linear
relaxation based bound propagation algorithms on general neural network
architectures with GPU acceleration.

To run our code, please first clone our repository, create a new Python 3.7+ environment and install dependencies:


```bash
git clone https://github.com/KaidiXu/LiRPA_Verify
pip install -r requirements.txt
cd src  # All code files are in the src/ folder
```

Part of the implementation is based on [the Gurobi solver](http://www.gurobi.com/) to solve Linear Programming
 problems for infeasible split checking. Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
Note that Fast-and-Complete only needs to occasionally use LP to eliminate infeasible branching and guarantee completeness.
See our recent work [*β*-CROWN](https://github.com/KaidiXu/Beta-CROWN) for totally avoiding LP and exploiting full GPU acceleration in complete neural network verification.
  
We also provide an `environment.yml` file for creating a conda environment with necessary packages
including gurobi:

```bash
conda env create -f environment.yml
conda activate fastandcomplete
```

Our code is tested on Ubuntu 20.04 with PyTorch 1.7 and Python 3.7.


### Complete verification

We use a set of benchmarking CIFAR-10 models (base-easy, base-med, base-hard, wide and deep) and 
their corresponding properties provided in [this repository](https://github.com/oval-group/GNN_branching/tree/a18ea1a4db0b9375ea8eb8b9f37e888d2f1d9692/cifar_exp).  This set
of models and properties have become a standard benchmark in a few papers in complete
verification.

Our code is in the `src` folder. To reproduce our results, for example, on CIFAR-10 Base model with easy properties, please run:

```bash
python bab_verification.py --load "models/cifar_base_kw.pth" --model cifar_model --data CIFAR-easy --batch_size 400 --timeout 3600 
```

On CIFAR-10 Base model with med properties:

```bash
python bab_verification.py --load "models/cifar_base_kw.pth" --model cifar_model --data CIFAR-med --batch_size 400 --timeout 3600 
```

On CIFAR-10 Base model with hard properties:

```bash
python bab_verification.py --load "models/cifar_base_kw.pth" --model cifar_model --data CIFAR-hard --batch_size 400 --timeout 3600 
```

On CIFAR-10 Wide model:

```bash
python bab_verification.py --load "models/cifar_wide_kw.pth" --model cifar_model_wide --data CIFAR-wide --batch_size 200 --timeout 3600 
```

On CIFAR-10 Deep model:

```bash
python bab_verification.py --load "models/cifar_deep_kw.pth" --model cifar_model_deep --data CIFAR-deep --batch_size 150 --timeout 3600
```

After finishing running the command, you should see the reported mean and median of the running time and the number of branches for all properties in the dataset.


### Branching Heuristics

In this implementation we use the simple [BaBSR](https://www.jmlr.org/papers/volume21/19-468/19-468.pdf) branching heurstic without modifications. A better branching heurstic can further enhance verification performance. For a fair comparison to our implementation, the same branching heurstic should be used.

### Verification without LP

Note that we also support verification without using gurobi: you can add `--no_LP --max_subproblems_list 100000` 
to always solving bounds by Optimized CROWN. In this mode the verifier runs faster but cannot guarantee completeness. To ensure completeness without LP, use our more powerful verification technique: [*β*-CROWN](https://github.com/KaidiXu/Beta-CROWN)!

### BibTex entry

```
@inproceedings{
    xu2021fast,
    title={Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers},
    author={Kaidi Xu and Huan Zhang and Shiqi Wang and Yihan Wang and Suman Jana and Xue Lin and Cho-Jui Hsieh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=nVZtXBI6LNn}
}
```


+++
title = 'About'
description = 'A minimal functional NeuralSAT: how each layer of a neural network is bounded, and how branch and bound and DPLL(T) use those bounds.'
+++

This is a small, functional implementation of the core of
[NeuralSAT](https://github.com/dynaroars/neuralsat): given a neural network,
a point $x_0$ and a radius $\varepsilon$, decide whether the network keeps its
prediction for **every** input in the region around $x_0$. This site explains
how the code works: how each layer is bounded, and how two search algorithms
use those bounds to give a proof.

## The question

Let $f$ be the network, $y$ the label it predicts at $x_0$, and
$\mathcal{B} = \{x : \lVert x - x_0 \rVert_\infty \le \varepsilon\}$. Append a
final linear layer that outputs the margins $f_y(x) - f_k(x)$ for every other
class $k$. Then the question is whether all margins stay positive on
$\mathcal{B}$ (optionally cut by extra convex constraints). A search answers
with one of three statuses:

| status | meaning |
|---|---|
| `unsat` | proven: no input in the region changes the prediction |
| `sat` | a counterexample was found, and it is a real input of the region |
| `unknown` | the budget ran out, or no neuron is left to split and the bounds are still not enough |

## How the pieces fit

```text
nn.Sequential
   │  parse: one dict per layer, with its input and output shape
   ▼
layers ──► abstraction ──► bounds of every neuron on a region (sound, not exact)
              ▲                       │
              │ theory call           ▼
           search (BaB or DPLL(T)): split an unstable ReLU, bound again,
                                    prove the pieces, or find a counterexample
                                      │
                                      ▼
                              unsat / sat / unknown
```

| part | where | what it does |
|---|---|---|
| layers | `abstraction/lirpa/layer/` | one file per layer type: forward, and how to bound it |
| CROWN, α-CROWN | `abstraction/lirpa/` | backward substitution through the layers |
| IBP, zonotope | `abstraction/ibp/`, `abstraction/zonotope/` | two cheaper abstractions that propagate boxes or zonotopes forward |
| NeuralSAT abstraction | `abstraction/neuralsat/` | the same bounds over a region with extra convex constraints |
| BaB, DPLL(T) | `search/` | the two searches that call an abstraction |
| experiment, site | `writeup/` | this site, and the code that produced its numbers |

Everything is plain functions (no classes) over `nn.Sequential` models built
from Linear, Conv2d, BatchNorm, Flatten, Identity/Dropout, AvgPool2d,
MaxPool2d, ReLU, LeakyReLU, Sigmoid and Tanh.

## Where to start

- [Walkthrough](../walkthrough/): a tiny network bounded by hand, every
  substitution shown with its numbers. Read this first.
- [Bounding](../bounding/): the method in general: back-substitution,
  concretization, α-CROWN, phase splits and the constrained abstraction.
- [Layers](../layers/): how each layer type is bounded, with figures.
- [Search](../search/): what BaB and DPLL(T) do, step by step, on a real run.
- [Leaderboard](../): which abstraction and search verify the most
  instances.
- [Results](../results/): the experiment, and the checks against
  auto_LiRPA.
- [Reproduce](../reproduce/): the commands.

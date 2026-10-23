+++
title = 'Results'
toc = true
description = 'How the leaderboard experiment is run, checks against auto_LiRPA, and the cost of the dense and the patch form.'
+++

## The experiment behind the leaderboard

Ten configurations on the same instances: {BaB, DPLL(T)} x {IBP, zonotope,
CROWN, α-CROWN, NeuralSAT}. The ranking and the cactus plot are on the
[front page](../).

- **Instance.** A random `8-24-24-3` ReLU network, a point $x_0$ and a radius
  $\varepsilon$. Is the predicted label still the argmax on the $\ell_\infty$ ball
  around $x_0$, cut by a halfspace through $x_0$ and an $\ell_2$ ball of
  radius $0.8\,\varepsilon \sqrt{d}$?
- **Radius.** $\varepsilon$ is 1.02, 1.05, 1.1 or 1.2 times the largest radius
  that plain CROWN verifies (found by bisection), cycling by instance. Every
  instance starts just beyond what bounding alone proves.
- **Abstractions.** IBP, zonotope, CROWN and α-CROWN (10 Adam steps) bound the
  ball only and cannot use the constraints (they only filter
  counterexamples). The NeuralSAT abstraction (30 primal-dual steps) uses them
  in its bounds. A gain of NeuralSAT therefore measures what the constraints
  buy, not tighter bounds on the same region.
- **Searches.** Both start with a PGD attack, then split unstable ReLUs.
- **Status.** `unsat` (proven), `sat` (a counterexample was found), or
  `unknown` (stuck, or out of the 2 s timeout). Solved means `unsat` or `sat`.
- **Score.** The number of solved instances; ties go to the lower PAR-2 (the
  mean runtime with an unsolved instance counted as twice the limit).

**Reading the ranking.**

- With the same search, an abstraction that keeps dependencies (zonotope,
  CROWN, α-CROWN) solves more than IBP, which forgets them.
- The NeuralSAT abstraction costs more per bound call (tens of primal-dual
  steps against one forward or backward pass) and also solves the harder
  instances, so its solved time is higher.
- BaB and DPLL(T) solve about the same instances. With the explanation of
  [Search](../search/) a DPLL(T) conflict costs no extra bound call. While
  developing, both searches made the same number of bound calls on the
  instances they both solve (52 each), and a learned clause was as long as the
  decision path in every case I looked at.

## Against auto_LiRPA

CROWN output bounds of the same random networks, $\varepsilon = 0.1$, float32,
compared with the CROWN of auto_LiRPA (the copy in the `alpha-beta-CROWN`
repository). The largest absolute difference over lower and upper bounds:

| network | difference |
|---|---|
| conv, ReLU | 3e-8 |
| conv, BatchNorm, ReLU | 4e-8 |
| conv, LeakyReLU | 4e-8 |
| conv, ReLU, MaxPool | 4e-8 |
| conv, ReLU, AvgPool | 3e-8 |
| two convs (strides, groups) | 3e-8 |
| conv, Sigmoid | 4e-7 |
| conv, Tanh | 4e-6 |
| MLP, Tanh | 3.5e-5 |
| a CNN with every layer type | 1e-7 |
| a network ending in ReLU | 3e-8 |
| deep MLP, ReLU and LeakyReLU | 3e-8 |

Everything without Sigmoid or Tanh agrees to float32 rounding. The S-shaped
layers differ by a few 1e-6 to 1e-5, the expected size of the gap between a
table with step 0.01 and a bisection (see [Layers](../layers/#sigmoid-and-tanh)).
The same holds with the patch form on, over 17 architectures.

## Cost of the dense and the patch form

Time and peak memory of CROWN on three CNNs, 10 threads, one process per run
(about 0.9 GB of each peak is the Python and torch baseline):

| network | dense | patches |
|---|---|---|
| MNIST: conv16, conv32 (stride 2), fc100 | 0.98 s, 1.20 GB | 0.83 s, 0.96 GB |
| CIFAR: the same layers on 3x32x32 | 1.10 s, 1.33 GB | 0.84 s, 0.97 GB |
| CIFAR wide: conv16 (stride 1), conv32 (stride 2) | 3.25 s, 5.79 GB | 0.91 s, 1.04 GB |

The output bounds are identical. The saving grows with the size of the first
conv layers: on the MNIST net the second conv's patch tensor has 156,800
numbers against 1.23 million for the dense matrix.

## What is checked

- 40 tests cover the layers, the abstractions and the searches; run
  `make test`.
- Layer by layer: affine backward against the explicit Jacobian, every
  relaxation against the true function on a grid, whole-CNN soundness by
  sampling, and exactness on an affine CNN.
- Patches against dense: the same coefficients on 7 networks.
- Searches: `unsat` for a tiny radius, `sat` with a verified counterexample for
  a large one, on an MLP and on a CNN.
- Mutating a relaxation (the S-shaped tangent, the ReLU chord, the max-pool
  linearity test) makes the tests fail, so they do detect a wrong line.

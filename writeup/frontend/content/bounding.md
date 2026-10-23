+++
title = 'Bounding'
toc = true
description = 'Linear bound propagation: back-substitution, concretization, alpha-CROWN, phase splits and the constrained NeuralSAT abstraction.'
+++

A bound is **sound** if it holds for every input in the region, and
**tight** if it is close to the true range. Everything here is sound; the
methods differ in tightness and cost.

## Symbolic bounds and back-substitution

For the output $z$ of some layer, CROWN keeps two linear functions of the
network input $x$:

$$
\underline{A}\,x + \underline{b} \;\le\; z \;\le\; \overline{A}\,x + \overline{b}
$$

It starts at the layer itself, where $A = I$ and $b = 0$, and walks the layers
backward. Each layer rewrites the bound in terms of **its own input**:

- an **affine** layer $v = W u + c$ is exact: $(A, b) \mapsto (AW,\; Ac + b)$ on
  both sides;
- a **non-affine** layer has no exact linear form. Its neurons are replaced by
  two lines, valid on the interval $[l, u]$ of the layer's input:
  $d^{lo}\,u + e^{lo} \le v \le d^{hi}\,u + e^{hi}$.

For a non-affine layer the sign of each coefficient decides which line a
neuron takes. With $A^+ = \max(A, 0)$ and $A^- = \min(A, 0)$, a **lower**
bound uses the lower line where the coefficient is positive and the upper
line where it is negative:

$$
A v + b \;\ge\; \left(A^+ D^{lo} + A^- D^{hi}\right) u \;+\; \left(A^+ e^{lo} + A^- e^{hi} + b\right)
$$

with $D = \mathrm{diag}(d)$. The upper bound swaps the two lines. This is
`activation.substitute` in the code. The [walkthrough](../walkthrough/) shows
it on numbers; [Layers](../layers/) gives the lines of every layer.

## Concretization

When the walk reaches the input, the bound is $a \cdot x + b$ and the region
is the box $\lVert x - x_0 \rVert_\infty \le \varepsilon$. Hölder's inequality
gives its exact minimum:

$$
\min_{\lVert x - x_0 \rVert_\infty \le \varepsilon} a \cdot x
\;=\; a \cdot x_0 \;-\; \varepsilon\, \lVert a \rVert_1
$$

Each input coordinate moves to the end of its interval that makes its term
smallest. This is `crown.concretize`.

## Which layers get concrete bounds

A non-affine layer needs the interval of its input, so the layer **before**
every non-affine layer is bounded, and so is the last layer. The rule of
`crown.layer_bounds`, as in auto_LiRPA:

- an affine layer, and the last layer, is bounded by back-substitution to the
  input and concretization;
- a non-affine layer in the middle is bounded by **interval propagation** from
  the bounds of its input. ReLU, LeakyReLU, Sigmoid, Tanh and max pooling are
  monotone, so the output interval is exactly $[f(l), f(u)]$. This is tighter
  than back-substituting through the lines, which could put the lower bound of
  a ReLU output below 0.

## α-CROWN

The lower line of an unstable ReLU, $a \ge \alpha z$, is valid for **every**
$\alpha \in [0, 1]$. CROWN picks $\alpha$ by a rule; α-CROWN
(`abstraction/lirpa/alphacrown.py`) treats it as a parameter, per neuron, with
separate values for the lower and the upper walk:

1. for each bounded layer in turn, start from the CROWN rule;
2. run Adam on the $\alpha$ of all earlier ReLUs to shrink the width
   $\sum (\overline{z} - \underline{z})$ of that layer;
3. keep the best bound seen, which starts at the CROWN bound.

Every iterate is sound because $\alpha$ is clipped to $[0, 1]$ inside the
relaxation, so the result is never looser than CROWN. Only ReLU slopes are
tuned; the other layers keep their fixed lines.

## Phase splits

The searches fix a ReLU or LeakyReLU neuron to **active** ($z \ge 0$) or
**inactive** ($z \le 0$). `crown.clamp_split` applies this to the bounds of the
neuron's input: an active neuron gets $l \leftarrow \max(l, 0)$, an inactive
one $u \leftarrow \min(u, 0)$. The relaxation then sees a stable neuron and uses
its exact line, which is sound inside the subdomain where the split holds. If
the clamp makes $l > u$, the subdomain is empty and counts as proven.

A split is a **no-op** when the neuron was already stable the same way. With
`core=True`, `layer_bounds` also reports which splits changed a bound
(`split_effect`); DPLL(T) uses that to explain its conflicts.

## Patches

The coefficient matrix of a conv layer has one row per output neuron, but each
row only touches a small window of the layer below. When the start neurons form a
conv-shaped layer, the coefficients are kept as **patches** (the window of every
start neuron) instead of a dense matrix, like auto_LiRPA. Patches pass through
Conv2d, BatchNorm, the pointwise activations, AvgPool2d and MaxPool2d; Linear
and Flatten start nodes are dense. `patches=False` selects the dense reference,
and tests check that both give the same coefficients. The
[results](../results/#cost-of-the-dense-and-the-patch-form) show the saving.

## Cheaper abstractions: IBP and zonotope

CROWN walks backward through the network once per bounded layer. Two other
abstractions go **forward** once and keep a small state per layer. They have the
same interface as CROWN (`layer_bounds`), so the searches use them as the
bound function without any change.

| abstraction | state per layer | affine layer | non-affine layer | keeps dependencies |
|---|---|---|---|---|
| IBP (`abstraction/ibp/`) | a box, $[l, u]$ | center maps normally, radius with $\lvert W \rvert$ | exact image of the interval | no |
| zonotope (`abstraction/zonotope/`) | a center and generators | exact | a strip around a line, one new generator | yes |
| CROWN | linear bounds on the input | exact | two lines | yes |

**IBP** is the cheapest and the loosest. An affine layer $v = Wu + c$ maps the
box with center $m$ and radius $r$ to the box with center $Wm + c$ and radius
$\lvert W \rvert\, r$, which is exact for one layer from the input. But after two
layers it treats neurons as independent: for $z_1 = x_1 + x_2$, $z_2 = x_1 - x_2$
and $y = z_1 + z_2 = 2 x_1$ on $x \in [-1, 1]^2$ it gets $y \in [-4, 4]$, while the
exact range is $[-2, 2]$.

**Zonotope** (DeepZ) keeps every neuron as $c + \sum_k \varepsilon_k g_k$ with
$\varepsilon_k \in [-1, 1]$, starting with one generator $\varepsilon\, e_i$ per input.
An affine layer maps the center and each generator exactly, so a purely affine
network is bounded **exactly** and the example above gives $[-2, 2]$. A non-affine
layer cannot be mapped exactly. It replaces each neuron $f(z)$ by a strip
$\lambda z + [\text{low}, \text{high}]$ around a line of slope $\lambda$, which is the
line $+$ a new generator of half-width $(\text{high} - \text{low}) / 2$:

| layer | slope $\lambda$ | strip |
|---|---|---|
| ReLU, unstable | $\tfrac{u}{u - l}$ (the chord) | $[0,\; -\lambda l]$ |
| ReLU, stable | $1$ or $0$ | a point, no new generator |
| LeakyReLU | the chord slope | $[0,\; l(s - \lambda)]$ |
| Sigmoid, Tanh | $\min(f'(l), f'(u))$ | $[f(l) - \lambda l,\; f(u) - \lambda u]$ |
| MaxPool | $1$ on the input $x_i$ of the largest lower bound | $[0,\; U - l_i]$ above it ($0$ if the window is linear) |

The bounds of a layer are $c \mp \sum_k \lvert g_k \rvert$. The number of generators grows
with the number of unstable neurons, which is the cost of the zonotope. A phase
split works the same way as for CROWN: the clamped bounds make the neuron stable
for the next transformer, so it is exact inside the subdomain.

## The NeuralSAT abstraction

The region of a verification query is often the ball **cut by convex
constraints** $h_j(x) \le 0$, for example a halfspace through $x_0$ or an
$\ell_2$ ball. Hölder's closed form no longer applies, because the minimum of
$a \cdot x$ over the ball can sit outside the constraints. The NeuralSAT
abstraction (`abstraction/neuralsat/`) keeps the same back-substitution and
changes only the last step, `concretize`.

**Lagrangian.** For every $\mu_j \ge 0$,

$$
L(x, \mu) = a \cdot x + b + \sum_j \mu_j \max\left(0,\, h_j(x)\right)
$$

is at most $a \cdot x + b$ wherever the constraints hold (the penalty is zero
there and never negative), so $\min_{\mathcal{B}} L$ is a sound lower bound of
the bound over the constrained region, **for every** $\mu \ge 0$.

**Certified bound.** $L$ is convex, so its tangent at any point $x$ lies below it.
With a subgradient $g$ of $L(\cdot, \mu)$ at $x$, the minimum of the tangent
over the ball has a closed form:

$$
\min_{x' \in \mathcal{B}} L(x', \mu) \;\ge\; L(x, \mu) \;+\; g \cdot (x_0 - x) \;-\; \varepsilon\, \lVert g \rVert_1
$$

This value is sound for **every** $x$ and every $\mu \ge 0$, so the search can
stop at any iteration and the best value seen is a valid bound.

**Search for a good $x$ and $\mu$.** `concretize` runs a projected primal-dual
loop, and keeps the largest certified bound:

1. start at the Hölder minimizer of $a \cdot x$ over the ball, with
   $\mu = \lVert a \rVert_2$ per row; the bound with $\mu = 0$ is the CROWN bound;
2. at each step take certified bounds at $x$ with the current $\mu$, and with the
   best single multiplier per constraint (below);
3. primal step: $x \leftarrow \Pi_{\mathcal{B}}(x - \eta g)$ with a
   Barzilai-Borwein step size $\eta$;
4. dual step: $\mu_j \leftarrow \max(0, \mu_j + \tau\, h_j(x))$;
5. stop after `patience` steps without a gain of more than `tol`.

**Exact multiplier for one constraint.** Along one multiplier the certified bound
is concave and piecewise linear: $g$ is affine in $\mu_j$, so the
$-\varepsilon\lVert g \rVert_1$ term only bends where a coordinate of $g$ changes
sign. `best_multiplier` sorts those breakpoints and finds the maximum without
step sizes. On the toy problem "minimize $-(x_1 + x_2)$ on $[-1,1]^2$ with
$x_1 + x_2 \le 0$" the box alone gives $-2$ and this gives the exact optimum $0$.

With no constraints the abstraction returns exactly the CROWN bounds. The
constraints are plain functions `h(x)` (`halfspace`, `l2_ball`, or any convex
callable written with torch operations).

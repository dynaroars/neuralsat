+++
title = 'Layers'
toc = true
description = 'How each supported layer is bounded: exact affine layers, and the two lines of every non-affine layer.'
+++

Each layer type is one file in `abstraction/lirpa/layer/` with the same four
functions: `parse` (torch module to a dict), `forward`, `backward` (the
back-substitution of [Bounding](../bounding/)) and, for non-affine layers,
`interval` (the exact output interval). The table says what each layer does
to a symbolic bound.

| layer | kind | what the back-substitution does |
|---|---|---|
| Linear, Conv2d, BatchNorm, AvgPool2d | affine | exact: $(A, b) \mapsto (A J, A c + b)$ |
| Flatten, Identity, Dropout | affine | the coefficients are unchanged |
| ReLU, LeakyReLU | piecewise linear | two lines per neuron; can be split by the search |
| Sigmoid, Tanh | S-shaped | two lines per neuron from tangents and a chord |
| MaxPool2d | window maximum | one input and one constant per window |

## Affine layers

An affine layer $v = J u + c$ costs nothing: both bounds are rewritten with
the same map, $(A, b) \mapsto (A J,\; A c + b)$. The code never builds $J$ for
convolutions and pooling. `affine.backward` pulls every row of $A$ back with
the vector-Jacobian product of the layer function (for a convolution that is
the transposed convolution), and gets $c$ as the layer's value at $0$.

- **Linear.** $J = W$, $c = $ bias.
- **Conv2d.** Stride, padding, dilation and groups are all handled by the
  same product. Zero padding only.
- **BatchNorm** (evaluation mode). With the running statistics it is a
  per-channel affine map $y = s \odot x + t$, where
  $s = \gamma / \sqrt{\sigma^2 + \epsilon}$ and $t = \beta - \mu s$. The
  coefficients are multiplied by $s$ per channel.
- **AvgPool2d.** Each output is the mean of a window, a fixed linear map.
- **Flatten, Identity, Dropout.** The coefficients are over the flattened
  values already, so nothing changes. Dropout is the identity in evaluation
  mode.

No relaxation is involved, so no precision is lost here. The first layer of a
network, bounded over the input box, is exact.

## ReLU

For $a = \max(0, z)$ with $z \in [l, u]$ there are three cases:

- **active**, $l \ge 0$: $a = z$, both lines are $z$;
- **inactive**, $u \le 0$: $a = 0$, both lines are $0$;
- **unstable**, $l < 0 < u$: the upper line is the **chord** from $(l, 0)$ to
  $(u, u)$, slope $\tfrac{u}{u - l}$ and intercept $-\tfrac{l u}{u - l}$. The
  lower line is $a \ge \alpha z$ for any $\alpha \in [0, 1]$.

{{< fig "relu" "ReLU lines on three intervals. Left: unstable with u < -l, lower slope 0. Middle: unstable with u > -l, lower slope 1. Right: stable active, where the lines meet the function." >}}

CROWN picks $\alpha$ to make the region between the two lines small. The two
candidates leave triangles of area $\tfrac12 (u - l)\,u$ for $\alpha = 0$ and
$\tfrac12 (u - l)(-l)$ for $\alpha = 1$, so $\alpha = 1$ wins exactly when
$u > -l$. This is the rule in `relu.relax`, and the [walkthrough](../walkthrough/)
shows the areas. α-CROWN replaces the rule by a learned $\alpha$.

**Why it is sound.** On $[l, u]$ the function is convex, so it lies below its
chord; and $\max(0, z) \ge \alpha z$ for $\alpha \in [0, 1]$.

## LeakyReLU

For $a = \max(z, s z)$ with slope $s \in [0, 1]$ ($s = 0$ is ReLU):

- active: $a = z$; inactive: $a = s z$ (exact lines in both cases);
- unstable: the upper line is the chord through $(l, s l)$ and $(u, u)$, with
  slope $c = \tfrac{u - s l}{u - l}$. The lower line is $a \ge \alpha z$ for any
  $\alpha \in [s, 1]$; the default is $1$ if $c > 0.5$ and $s$ otherwise.

{{< fig "leakyrelu" "LeakyReLU with slope 0.2. The default lower slope follows the chord slope." >}}

## Sigmoid and Tanh

Both are S-shaped: convex for $z < 0$ and concave for $z > 0$. The lines
follow auto_LiRPA. With $m = (l + u) / 2$ the midpoint, a **chord** is the line
through the two endpoints and a **tangent at $p$** touches the curve at $p$:

| interval | lower line | upper line |
|---|---|---|
| $u \le 0$ (convex) | tangent at $m$ | chord |
| $l \ge 0$ (concave) | chord | tangent at $m$ |
| $l < 0 < u$ | the chord if it is a valid lower bound, else the tangent at $d_{lo}$ | the chord if it is a valid upper bound, else the tangent at $d_{hi}$ |

For an interval across 0, $d_{lo} \le 0$ is the point closest to 0 whose
tangent stays below $f(u)$, and $d_{hi} \ge 0$ the point closest to 0 whose
tangent stays above $f(l)$. A tangent from the convex side touches the curve
and then runs under the concave part, which is why it can serve as a lower
line. When $d_{lo} \ge m$ (or $d_{hi} < m$) the code uses the tangent at the
midpoint instead, so the lines change smoothly with the interval.

{{< fig "sigmoid" "Sigmoid on three intervals: below zero, above zero, and across zero." >}}

{{< fig "tanh" "Tanh on the same three kinds of interval." >}}

auto_LiRPA reads the tangent points $d_{lo}, d_{hi}$ from a table with step
$0.01$; here they are found by bisection (`sshaped.tangent_points`) keeping the
point on the safe side, so the lines are sound and differ from auto_LiRPA's in
the third decimal.

## MaxPool2d

For one pooling window with inputs $x_1, \dots, x_n$ in $[l_i, u_i]$, let
$i^\star$ be the input with the **largest lower bound**:

- **lower line:** $y \ge x_{i^\star}$, since the maximum is at least any input;
- **upper line:** if $l_{i^\star} \ge u_j$ for every $j \ne i^\star$, the window is
  **linear** and $y = x_{i^\star}$ exactly; otherwise $y \le \max_j u_j$, a
  constant.

{{< fig "maxpool" "Two windows of four inputs. Left: x1 beats every other upper bound, so the upper line is x1 itself. Right: the intervals overlap, so the upper line is the constant max u." >}}

Each output neuron reads one input, so the back-substitution is a scatter-add
of the coefficients at the chosen input positions. As in auto_LiRPA the windows
must not overlap (`stride == kernel_size`, no padding, no dilation, no
`ceil_mode`), which `parse` checks.

## What the output interval is

Besides the lines, every non-affine layer gives the exact interval of its
output (`interval`): $[f(l), f(u)]$ for the monotone ReLU, LeakyReLU, Sigmoid
and Tanh, and max pooling of the lower and upper bounds for MaxPool. That is
what the next layer uses as its input interval, and it is the whole of IBP.

## The same layers for the other abstractions

Each non-affine layer module has one more function, `zonotope`, which turns the
zonotope of its input into the zonotope of its output (the strips of
[Bounding](../bounding/#cheaper-abstractions-ibp-and-zonotope)). Affine layers
need nothing new: IBP runs them with `ops.absolute` (absolute weights, no bias)
and the zonotope pushes its center and generators through `forward`.

## How the layers are tested

- every `backward` of an affine layer is compared with the explicit Jacobian
  from autograd;
- every relaxation is checked against the true function on a dense grid of
  hundreds of random intervals, including stable and degenerate ones;
- a CNN with every layer type is sampled, and each bounded value must lie inside
  its bounds;
- for a purely affine CNN the bounds equal the exact range;
- the numbers agree with auto_LiRPA (see [Results](../results/)).

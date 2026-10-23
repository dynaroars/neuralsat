+++
title = 'Reproduce'
toc = true
description = 'The repository layout and the commands.'
+++

## Layout

```text
abstraction/
  lirpa/             CROWN and alpha-CROWN
    layer/           one file per layer type (+ patches.py, shared helpers)
    crown.py         backsub, concretize, layer_bounds, margin layers
    alphacrown.py
  ibp/               interval bound propagation
  zonotope/          the zonotope abstraction (DeepZ)
  neuralsat/         the constrained abstraction
    constraint.py    halfspace, l2_ball
    concretize.py    Lagrangian, certified bound, primal-dual loop
    neuralsat.py
search/              bab.py, dpll.py, common.py
test/                pytest tests
writeup/
  backend/           the experiment and the plots (Python)
    exp/             instances, runner, thread ramp, traces
    plot/            cactus plot, progress table, layer figures
    data/            raw results: one CSV per configuration
  frontend/          this site (Hugo)
  makefile
```

## Commands

From the project root:

```bash
make test        # the tests
make lint        # ruff, PEP 8 with 79 columns
```

From `writeup/` (or `make -C writeup <target>` from anywhere):

```bash
make exp         # run the ten configurations (resumable, one after the other)
make plot        # the ranking and the cactus figures for the front page
make progress    # done / unsat / sat / unknown per configuration
make figures     # the layer figures of the site
make walkthrough # regenerate the walkthrough page from the real code
make view        # this site, live, in the browser
make site        # the static site in frontend/public/
make clean       # delete the results; then: make clean exp plot
```

`make exp` skips the configurations that already finished, so after adding an
abstraction only the new ones run. It writes `backend/data/csv/{search}_{abstraction}.csv`, one row per
instance with the columns `idx, search, abstraction, eps, status, error, dt`,
and a rerun skips the instances already in the file. Exceptions go to
`*_errors.csv` next to it.

Using the code on your own network:

```python
import torch
from abstraction.lirpa import crown
from search.bab import bab

model = torch.nn.Sequential(...)      # Linear, Conv2d, ReLU, ...
x0 = torch.rand(1, 28, 28)            # one input, without the batch dimension
print(crown.verify(model, x0, 1 / 255, label))   # bounds only
print(bab(model, x0, 1 / 255, label))            # ('unsat' | 'sat' | 'unknown', witness)
```

The input keeps its shape, the radius is for the $\ell_\infty$ ball, and
`bound_fn=` selects another abstraction: `ibp.layer_bounds`,
`zonotope.layer_bounds`, `alphacrown.layer_bounds`, or
`neuralsat.layer_bounds` with `constraints=`.

## Style of the code

Functions only (no classes), PEP 8, `nn.Sequential` models only. New layer
types go in `abstraction/lirpa/layer/`: a module with `TYPES`, `parse`,
`forward`, `backward` (and `interval` for a non-affine layer), registered in
`layer/__init__.py`.

## Credits

The step-by-step [walkthrough](../walkthrough/) follows the explanations of
[hocdot/abstract-domain-explainer](https://github.com/hocdot/abstract-domain-explainer).
The layer relaxations and the patch form follow
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA), which is also
the reference the bounds were checked against.

haioc 
======

This repo contains a small PyTorch's C++/Cuda extension of operations.
Current list of implemented operations:

| Name             | Differentiable |
|:-----------------|:--------------:|
| `any_eq_any`     |       ❌        |
| `fill_if_eq_any` |       ❌        |

## Installation

#### From prebuilt wheels

#### From source
To install globally, clone this repo and run:

```
pip install .
```

Or build inplace binary with:

```
python setup.py build_ext --inplace
```

## Usage

### Example of `any_eq_any` and `fill_if_eq_any`:

```python
import torch

import haioc

data = torch.randperm(700 * 99).sub_(700 * 99 // 2).view(700, 99).int()
xs = torch.arange(0, 500).int()

delete_mask = haioc.any_eq_any(data, xs)
zero_mask = haioc.fill_if_eq_any(data, -xs, 0.)
```


Do check [`tests/test_haioc.py`](tests/test_haioc.py).

## License

The code is released under the MIT No Attribution license. See [`LICENSE.txt`](LICENSE.txt) for details.

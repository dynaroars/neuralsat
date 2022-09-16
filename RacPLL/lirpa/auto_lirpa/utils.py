from collections import defaultdict, namedtuple
import sys


reduction_sum = lambda x: x.sum(1, keepdim=True)
reduction_mean = lambda x: x.mean(1, keepdim=True)
reduction_max = lambda x: x.max(1, keepdim=True).values
reduction_min = lambda x: x.min(1, keepdim=True).values


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func


def stop_criterion_sum(threshold=0):
    return lambda x: (x.sum(1, keepdim=True) > threshold)

# unpack tuple, dict, list into one single list
# TODO: not sure if the order matches graph.inputs()
def unpack_inputs(inputs, device=None):
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        res = []
        for item in inputs: 
            res += unpack_inputs(item, device=device)
        return res
    else:
        if device is not None:
            inputs = inputs.to(device)
        return [inputs]


# Create a namedtuple with defaults
def namedtuple_with_defaults(name, attr, defaults):
    assert sys.version_info.major == 3
    if sys.version_info.major >= 7:
        return namedtuple(name, attr, defaults=defaults)
    else:
        # The defaults argument is not available in Python < 3.7
        t = namedtuple(name, attr)
        t.__new__.__defaults__ = defaults
        return t



def batched_index_select(x, dim, index):
    # Assuming the x has a batch dimension.
    # index has dimensin [spec, batch].
    if x.ndim == 4:
        # Alphas for fully connected layers, shape [2, spec, batch, neurons]
        index = index.unsqueeze(-1).unsqueeze(0).expand(x.size(0), -1, -1, x.size(3))
    elif x.ndim == 6:
        # Alphas for fully connected layers, shape [2, spec, batch, c, h, w].
        index = index.view(1, index.size(0), index.size(1), *([1] * (x.ndim - 3))).expand(x.size(0), -1, -1, *x.shape[3:])
    elif x.ndim == 3:
        # Weights.
        x = x.expand(index.size(0), -1, -1)
        index = index.unsqueeze(-1).expand(-1, -1, x.size(2))
    elif x.ndim == 2:
        # Bias.
        x = x.expand(index.size(0), -1)
    else:
        raise ValueError
    return torch.gather(x, dim, index)


        
eyeC = namedtuple('eyeC', 'shape device')
OneHotC = namedtuple('OneHotC', 'shape device index coeffs')

LinearBound = namedtuple_with_defaults('LinearBound', ('lw', 'lb', 'uw', 'ub', 'lower', 'upper', 'from_input', 'nominal', 'lower_offset', 'upper_offset'), defaults=(None,) * 10)
Patches = namedtuple_with_defaults('Patches', ('patches', 'stride', 'padding', 'shape', 'identity', 'unstable_idx', 'output_shape'), defaults=(None, 1, 0, None, 0, None, None))

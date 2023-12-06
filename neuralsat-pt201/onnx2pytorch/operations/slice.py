import torch
from torch import nn

from onnx2pytorch.utils import PRINT_DEBUG

def _to_positive_step(orig_slice, N):
    """
    Convert a slice object with a negative step to one with a positive step.
    Accessing an iterable with the positive-stepped slice, followed by flipping
    the result, should be equivalent to accessing the tensor with the original
    slice. Computing positive-step slice requires using N, the length of the
    iterable being sliced. This is because PyTorch currently does not support
    slicing a tensor with a negative step.
    """
    # Get rid of backward slices
    start, stop, step = orig_slice.indices(N)

    # Get number of steps and remainder
    n, r = divmod(stop - start, step)
    if n < 0 or (n == 0 and r == 0):
        return slice(0, 0, 1)
    if r != 0:  # a "stop" index, not a last index
        n += 1

    if step < 0:
        start, stop, step = start + (n - 1) * step, start - step, -step
    else:  # step > 0, step == 0 is not allowed
        stop = start + n * step
    stop = min(stop, N)

    return slice(start, stop, step)


class Slice(nn.Module):
    def __init__(self, dim=None, starts=None, ends=None, steps=None):
        self.dim = [dim] if isinstance(dim, int) else dim
        self.starts = starts
        self.ends = ends
        self.steps = steps
        super().__init__()

    def _fixup_params(self, shape, start, end, axes, steps):
        if start < 0:
            start += shape[axes]
            if end < start:
                end += shape[axes]
        if end < 0:
            if end == -9223372036854775807:  # -inf in ONNX
                end = 0  # only possible when step == -1
            else:
                end += shape[axes]
        if steps == -1:
            start, end = end, start + 1  # TODO: more test more negative step size.
        
        try:
            end = min(end, shape[axes])
        except:
            print('[+] Error:')
            print('\t- start:', start)
            print('\t- end:', end)
            print('\t- shape:', shape)
            print('\t- axes:', axes)
            raise
        return start, end

    # Older Pytorch version only passes steps as input.
    def forward(
            self, x: torch.Tensor, starts=None, ends=None, axes=None, steps=None
    ):
        
        if PRINT_DEBUG:
            print('SLICE:', x.shape, x.is_floating_point())
            print('[+] Before:')
            print('\t- start:', starts, self.starts)
            print('\t- end:', ends, self.ends)
            print('\t- axes:', axes, self.dim)
            print('\t- steps:', steps)
            
        if x.ndim == 1 and x.is_floating_point():
            x = x[None]
        
        start = self.starts if starts is None else starts
        if isinstance(start, tuple):
            assert len(start) == 1
            start = start[0]
        elif isinstance(start, torch.Tensor) and start.numel() == 1:
            start = start.item()
            
        end = self.ends if ends is None else ends
        if isinstance(end, tuple):
            assert len(end) == 1
            end = end[0]
        elif isinstance(end, torch.Tensor) and end.numel() == 1:
            end = end.item()
        
        axes = self.dim if axes is None else axes
        if (isinstance(axes, list) and len(axes) == 1): 
            axes = axes[0]
            # axes = max(axes, 1)
        elif isinstance(axes, torch.Tensor) and axes.numel() == 1:
            axes = axes.item()
            # axes = max(axes, 1)
        
        if x.is_floating_point():
            axes = max(axes, 1) # change batch size dim
        
        steps = self.steps if steps is None else steps
        steps = 1 if steps is None else steps
        if isinstance(steps, torch.Tensor) and steps.numel() == 1:
            steps = steps.item()    
        
        # assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape
        
        if PRINT_DEBUG:
            print('[+] After:')
            print('\t- start:', start)
            print('\t- end:', end)
            print('\t- axes:', axes)
            print('\t- steps:', steps)
            print('\t- shape:', shape)
            
        start, end = self._fixup_params(shape, start, end, axes, steps)
        
        if PRINT_DEBUG:
            print('[+] Fixed:')
            print('\t- start:', start)
            print('\t- end:', end)
            print('\t- axes:', axes)
            print('\t- steps:', steps)
            print('\t- shape:', shape)
        
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))

        if PRINT_DEBUG:
            print('\t- final:', final)
            print('\t- final:', final.shape)
            assert final.numel() > 0
        return final

    

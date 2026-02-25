from torch import Tensor
import torch
import torch.nn as nn
import torch._C as _C
import copy
import torch.utils.checkpoint as checkpoint

# Global setting for offloading
OFFLOAD_TO_CPU = 1 
COMPUTE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Force CUDA initialization
if torch.cuda.is_available():
    torch.zeros(1).cuda()

class BoundedTensor(Tensor):
    
    @staticmethod
    def __new__(cls, x, ptb=None, *args, **kwargs):
        if isinstance(x, Tensor):
            if OFFLOAD_TO_CPU:
                x_cpu = x.detach().cpu()
            else:
                x_cpu = x
            
            tensor = super().__new__(cls, x_cpu, *args, **kwargs)
            tensor.data = x_cpu.data
            tensor.requires_grad = x.requires_grad
            return tensor
        else:
            return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, ptb=None):
        self.ptb = ptb

    def __repr__(self):
        loc = "CPU" if self.device.type == 'cpu' else "GPU"
        ptb_info = self.ptb.__repr__() if hasattr(self, 'ptb') and self.ptb is not None else 'no ptb'
        return '<BoundedTensor [{}]: {}, {}>'.format(loc, super().__repr__(), ptb_info)

    def clone(self, *args, **kwargs):
        tensor = BoundedTensor(super().clone(*args, **kwargs), copy.deepcopy(self.ptb))
        return tensor

    def _func(self, func, *args, **kwargs):
        temp = func(*args, **kwargs)
        new_obj = BoundedTensor(temp, self.ptb)
        return new_obj

    def to(self, *args, **kwargs):
        if hasattr(self.ptb, 'x_L') and isinstance(self.ptb.x_L, Tensor):
            self.ptb.x_L = self.ptb.x_L.to(*args, **kwargs)
        if hasattr(self.ptb, 'x_U') and isinstance(self.ptb.x_U, Tensor):
            self.ptb.x_U = self.ptb.x_U.to(*args, **kwargs)
        if hasattr(self.ptb, 'eps') and isinstance(self.ptb.eps, Tensor):
            self.ptb.eps = self.ptb.eps.to(*args, **kwargs)
        return self._func(super().to, *args, **kwargs)

    @classmethod
    def _convert(cls, ret):
        if cls is Tensor: return ret
        if isinstance(ret, Tensor): return ret.as_subclass(cls)

        if isinstance(ret, (list, tuple)):
            converted = [cls._convert(r) for r in ret]
            if type(ret) is tuple: return tuple(converted)
            if type(ret) is list: return list(converted)
            try:
                return type(ret)(tuple(converted))
            except TypeError:
                return type(ret)(*converted)
            except Exception:
                return tuple(converted)

        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        if not all(issubclass(cls, t) for t in types): return NotImplemented

        # --- SYNC / OFFLOAD LOGIC ---
        
        def to_gpu(x):
            if isinstance(x, (list, tuple)):
                return type(x)(to_gpu(item) for item in x)
            
            if isinstance(x, Tensor):
                if x.shape != x.data.shape:
                    raw = x.data
                else:
                    raw = x
                return raw.detach().to(COMPUTE_DEVICE).requires_grad_(x.requires_grad)
            return x

        def to_cpu(x):
            if isinstance(x, (list, tuple)):
                return type(x)(to_cpu(item) for item in x)
            
            if isinstance(x, Tensor):
                with _C.DisableTorchFunction():
                    if x.shape != x.data.shape:
                        out = x.data.cpu()
                    else:
                        out = x.cpu()
                    return out.as_subclass(Tensor)
            return x

        def check_requires_grad(x):
            if isinstance(x, (list, tuple)):
                return any(check_requires_grad(item) for item in x)
            if isinstance(x, Tensor):
                return x.requires_grad
            return False

        should_offload = OFFLOAD_TO_CPU and torch.cuda.is_available()

        if should_offload:
            cpu_args = [to_cpu(a) for a in args]
            cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
            
            needs_grad = any(check_requires_grad(a) for a in cpu_args)

            def closure(*c_args):
                # 1. Move Args to GPU
                g_args = [to_gpu(a) for a in c_args]
                # 2. Move Kwargs to GPU (Fix for Batch Norm / Other ops)
                g_kwargs = {k: to_gpu(v) for k, v in cpu_kwargs.items()}
                
                with _C.DisableTorchFunction():
                    g_res = func(*g_args, **g_kwargs)
                
                # 3. Move Result to CPU
                def recursive_to_cpu(res):
                    if isinstance(res, Tensor):
                        return res.cpu()
                    elif isinstance(res, (list, tuple)):
                        converted_items = [recursive_to_cpu(r) for r in res]
                        try:
                            return type(res)(tuple(converted_items))
                        except TypeError:
                            return type(res)(*converted_items)
                        except Exception:
                            return type(res)(converted_items)
                    return res
                
                return recursive_to_cpu(g_res)

            if needs_grad:
                tensor_inputs = []
                def extract_tensors(x):
                    if isinstance(x, (list, tuple)):
                        for item in x: extract_tensors(item)
                    elif isinstance(x, Tensor):
                        tensor_inputs.append(x)
                
                for a in cpu_args: extract_tensors(a)
                
                has_nested = any(isinstance(a, (list, tuple)) for a in cpu_args)
                
                if has_nested:
                    ret = closure(*cpu_args)
                else:
                    tensor_inputs = [a for a in cpu_args if isinstance(a, Tensor)]
                    def checkpoint_wrapper(*t_inputs):
                        iter_t = iter(t_inputs)
                        reconstructed_args = []
                        for a in cpu_args:
                            if isinstance(a, Tensor):
                                reconstructed_args.append(next(iter_t))
                            else:
                                reconstructed_args.append(a)
                        return closure(*reconstructed_args)
                    
                    ret = checkpoint.checkpoint(checkpoint_wrapper, *tensor_inputs, use_reentrant=False)
            else:
                ret = closure(*cpu_args)

        else:
            with _C.DisableTorchFunction():
                ret = func(*args, **kwargs)

        return cls._convert(ret)


class BoundedParameter(nn.Parameter):
    def __new__(cls, data, ptb, requires_grad=True):
        return BoundedTensor._make_subclass(cls, data, requires_grad)

    def __init__(self, data, ptb, requires_grad=True):
        self.ptb = ptb
        self.requires_grad = requires_grad

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.ptb, self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'BoundedParameter containing:\n{}\n{}'.format(
            self.data.__repr__(), self.ptb.__repr__())
    
    def __reduce_ex__(self, proto):
        raise NotImplementedError
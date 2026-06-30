from .constant import BoundConstant
from ..patches import Patches
from .base import *


class BoundConcat(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.IBP_rets = None
        self.ibp_intermediate = True

    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        self.axis = self.make_axis_non_negative(self.axis)
        return torch.cat(x, dim=int(self.axis))

    def interval_propagate(self, *v):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == torch.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))

        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb=ptb)
        else:
            raise RuntimeError(f"BoundConcat does not support inputs with norm {norms}")

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        self.axis = self.make_axis_non_negative(self.axis, 'output')
        assert self.axis > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                ret = list(torch.split(last_A, self.input_size, dim=self.axis + 1))
                # Skip unused input nodes to reduce the cost of computing unused intermediate bounds
                for i in range(len(ret)):
                    if (ret[i] == 0).all():
                        ret[i] = None
                return ret
            elif isinstance(last_A, Patches):
                assert len(self.input_shape) == 4 and self.axis == 1, "Split channel dimension is supported; others are unimplemented."
                # Patches shape can be [out_c, batch, out_h, out_w, in_c, patch_h, patch_w]
                # Or [spec, batch, in_c, patch_h, patch_w]  (sparse)
                new_patches = torch.split(last_A.patches, self.input_size, dim=-3)  # split the in_c dimension is easy.
                return [last_A.create_similar(p) for p in new_patches]
            else:
                raise RuntimeError(f'Unsupported type for last_A: {type(last_A)}')

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)

        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0

        # To avoid issues in other parts of the code, we prune unused
        # lA and uA only when they are both unused.
        for i in range(len(lA)):
            if lA[i] is None and uA[i] is not None:
                lA[i] = torch.zeros_like(uA[i])
            elif lA[i] is not None and uA[i] is None:
                uA[i] = torch.zeros_like(lA[i])

        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        self.axis = self.make_axis_non_negative(self.axis)
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)



BoundConcatFromSequence = BoundConcat


class BoundSlice(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.start = attr["starts"][0] if "starts" in attr else None
        self.end = attr["ends"][0] if "ends" in attr else None
        self.axes = attr["axes"][0] if "axes" in attr else None
        self.use_default_ibp = False
        self.ibp_intermediate = True

    def __repr__(self):
        attrs = {}
        if (len(self.inputs) == 5 and all(isinstance(item, BoundConstant) and item.value.numel() == 1 for item in self.inputs[1:])):
            attrs['start'] = self.inputs[1].value.item()
            attrs['end'] = self.inputs[2].value.item()
            attrs['axes'] = self.inputs[3].value.item()
            attrs['step'] = self.inputs[4].value.item()
        return super().__repr__(attrs)

    def _fixup_params(self, shape, start, end, axes, steps):
        if start < 0:
            start += shape[axes]
        if end < 0:
            if end == -9223372036854775807:  # -inf in ONNX
                end = 0  # only possible when step == -1
            else:
                end += shape[axes]
        if steps == -1:
            start, end = end, start + 1  # TODO: more test more negative step size.
        end = min(end, shape[axes])
        return start, end

    # Older Pytorch version only passes steps as input.
    def forward(self, x, start=None, end=None, axes=None, steps=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        axes = self.axes if axes is None else axes
        assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape if isinstance(x, Tensor) else [len(x)]
        start, end = self._fixup_params(shape, start, end, axes, steps)
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))
        return final

    def interval_propagate(self, *v):
        lb = tuple(map(lambda x:x[0],v))
        ub = tuple(map(lambda x:x[1],v))
        return Interval.make_interval(self.forward(*lb), self.forward(*ub))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        def _bound_oneside(A, start, end, axes, steps):
            if A is None:
                return None
            if isinstance(A, torch.Tensor):
                # Reuse the batch and spec dimension of A, and replace other shapes with input.
                A_shape = A.shape[:2] + self.input_shape[1:]
                new_A = torch.zeros(size=A_shape, device=A.device, requires_grad=A.requires_grad)
                # Fill part of the new_A based on start, end, axes and steps.
                # Skip the spec dimension at the front (axes + 1).
                dim = axes if axes < 0 else axes + 1
                indices = torch.arange(start, end, device=A.device)
                new_A = torch.index_copy(new_A, dim=dim, index=indices, source=A)
            elif isinstance(A, Patches):
                assert A.unstable_idx is None
                assert len(self.input_shape) == 4 and axes == 1, "Slice is only supported on channel dimension."
                patches = A.patches
                # patches shape is [out_c, batch, out_h, out_w, in_c, patch_h, patch_w].
                new_patches_shape = patches.shape[:4] + (self.input_shape[1], ) + patches.shape[-2:]
                new_patches = torch.zeros(size=new_patches_shape, device=patches.device, requires_grad=patches.requires_grad)
                indices = torch.arange(start, end, device=patches.device)
                new_patches = torch.index_copy(new_patches, dim=-3, index=indices, source=patches)
                # Only the in_c dimension is changed.
                new_A = A.create_similar(new_patches)
            else:
                raise ValueError(f'Unsupport A type {type(A)}')
            return new_A

        start, end, axes = x[1].value.item(), x[2].value.item(), x[3].value.item()
        steps = x[4].value.item() if len(x) == 5 else 1  # If step is not specified, it is 1.
        # Other step size untested, do not enable for now.
        assert steps == 1 and axes == int(axes) and start == int(start) and end == int(end)
        start, end = self._fixup_params(self.input_shape, start, end, axes, steps)
        # Find the original shape of A.
        lA = _bound_oneside(last_lA, start, end, axes, steps)
        uA = _bound_oneside(last_uA, start, end, axes, steps)
        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], 0, 0

    def bound_forward(self, dim_in, *inputs):
        assert len(inputs) == 5 or len(inputs) == 4
        start = inputs[1].lb.item()
        end = inputs[2].lb.item()
        axis = self.make_axis_non_negative(inputs[3].lb.item())
        assert axis > 0, "Slicing along the batch dimension is not supported yet"
        steps = inputs[4].lb.item() if len(inputs) == 5 else 1  # If step is not specified, it is 1.
        assert steps in [1, -1]
        x = inputs[0]
        shape = x.lb.shape
        start, end = self._fixup_params(shape, start, end, axis, steps)
        lw = torch.narrow(x.lw, dim=axis+1, start=start, length=end - start)
        uw = torch.narrow(x.uw, dim=axis+1, start=start, length=end - start)
        lb = torch.narrow(x.lb, dim=axis, start=start, length=end - start)
        ub = torch.narrow(x.ub, dim=axis, start=start, length=end - start)
        if steps == -1:
            lw = torch.flip(lw, dims=tuple(axis+1))
            uw = torch.flip(uw, dims=tuple(axis+1))
            lb = torch.flip(lb, dims=tuple(axis))
            ub = torch.flip(ub, dims=tuple(axis))
        return LinearBound(lw, lb, uw, ub)


class BoundSplit(Bound):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.use_default_ibp = True
        if 'split' in attr:
            self.split = attr['split']
        else:
            self.split = None

    def forward(self, *x):
        data = x[0]
        split = self.split if self.split is not None else x[1].tolist()
        if self.axis == -1:
            self.axis = len(data.shape) - 1
        return torch.split(data, split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert self.axis > 0
        split = self.split if self.split is not None else x[1].value.tolist()
        pre = sum(split[:self.output_index])
        suc = sum(split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(*last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):], device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(*last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):], device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        assert self.axis > 0 and self.from_input
        split = self.split if self.split is not None else x[1].lb.tolist()
        x = x[0]
        lw = torch.split(x.lw, split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])

from collections import OrderedDict, deque, defaultdict
import torch.nn as nn
import torch
import time
import os

from .parse_graph import parse_module
from .misc import bound_op_map
from .bound_tensor import *
from .operators import *
from .utils import *

class BoundedModule(nn.Module):

    def __init__(self, model, global_input, bound_opts=None, auto_batch_dim=True, device='auto', verbose=False, custom_ops={}):
        super().__init__()

        if isinstance(model, BoundedModule):
            raise
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return


        if bound_opts is None:
            bound_opts = {}

        default_bound_opts = {
            'ibp_relative': False, 
            'conv_mode': 'patches', 
            'sparse_intermediate_bounds': True, 
            'sparse_conv_intermediate_bounds': True
        }

        default_bound_opts.update(bound_opts)

        self.bound_opts = default_bound_opts
        # self.verbose = verbose
        self.custom_ops = custom_ops

        if auto_batch_dim:
            self.init_batch_size = -1

        if device == 'auto':
            try:
                self.device = next(model.parameters()).device
            except StopIteration:  # Model has no parameters. We use the device of input tensor.
                self.device = global_input.device
        else:
            self.device = device


        # self.global_input = global_input
        # self.ibp_relative = self.bound_opts.get('ibp_relative', False)
        self.conv_mode = self.bound_opts.get('conv_mode', 'patches')

        self._convert(model, global_input)

    def __call__(self, *input, **kwargs):

        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            opt = "forward"

        # for kwarg in [
        #     'disable_multi_gpu', 'no_replicas', 'get_property',
        #     'node_class', 'att_name']:
        #     if kwarg in kwargs:
        #         kwargs.pop(kwarg)

        if opt == "compute_bounds":
            return self.compute_bounds(**kwargs)
        else:
            return self.forward(*input, **kwargs)


    def forward(self, *x, final_node_name=None, clear_forward_only=False):
        r"""Standard forward computation for the network.

        Args:
            x (tuple or None): Input to the model.

            final_node_name (str, optional): The name of the final node in the model. The value
            on the corresponding node will be returned.

            clear_forward_only (bool, default `False`): Whether only standard forward values stored
            on the nodes should be cleared. If `True`, only standard forward values stored on the 
            nodes will be cleared. Otherwise, bound information on the nodes will also be cleared.

        Returns:
            output: The output of the model, or if `final_node_name` is not `None`, return the 
            value on the corresponding node instead.
        """
        self._set_input(*x, clear_forward_only=clear_forward_only)
        degree_in = {}
        queue = deque()
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)

        print(queue)

        forward_values = {}
        final_output = None
        while len(queue) > 0:
            l = queue.popleft()
            print(l)
            inp = [forward_values[l_pre] for l_pre in l.input_name]
            for l_pre in l.inputs:
                # print('l_pre', l_pre, l_pre.from_input)
                l.from_input = l.from_input or l_pre.from_input

            fv = l.forward(*inp)

            if isinstance(fv, torch.Size) or isinstance(fv, tuple):
                fv = torch.tensor(fv, device=self.device)

            # print(l.forward_value)
            object.__setattr__(l, 'forward_value', fv)

            if not hasattr(l, 'batch_dim'):
                inp_batch_dim = [l_pre.batch_dim for l_pre in l.inputs]
                try:
                    l.batch_dim = l.infer_batch_dim(self.init_batch_size, *inp_batch_dim)
                except:
                    raise Exception(
                        'Fail to infer the batch dimension of ({})[{}]: forward_value shape {}, input batch dimensions {}'.format(
                            l, l.name, l.forward_value.shape, inp_batch_dim))
            
            forward_values[l.name] = l.forward_value

            # Unperturbed node but it is not a root node. Save forward_value to value.
            # (Can be used in forward bounds.)
            if not l.from_input and len(l.inputs) > 0:
                print('lllll', l)
                l.value = l.forward_value

            for l_next in l.output_name:
                degree_in[l_next] -= 1
                if degree_in[l_next] == 0:  # all inputs of this node have already set
                    queue.append(self._modules[l_next])



        if final_node_name:
            return forward_values[final_node_name]
        else:
            out = deque([forward_values[n] for n in self.output_name])

            def _fill_template(template):
                if template is None:
                    return out.popleft()
                elif isinstance(template, list) or isinstance(template, tuple):
                    res = []
                    for t in template:
                        res.append(_fill_template(t))
                    return tuple(res) if isinstance(template, tuple) else res
                elif isinstance(template, dict):
                    res = {}
                    for key in template:
                        res[key] = _fill_template(template[key])
                    return res
                else:
                    raise NotImplementedError

            return _fill_template(self.output_template)







            # exit()








    def _set_input(self, *x, new_interval=None, clear_forward_only=False):
        self._clear_and_set_new(new_interval=new_interval, clear_forward_only=clear_forward_only)
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            node = self._modules[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, (BoundedTensor, BoundedParameter)):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None

        self._mark_perturbed_nodes()
        if self.init_batch_size == -1:
            self.init_batch_size = inputs_unpacked[0].shape[0]


    """Mark the graph nodes and determine which nodes need perturbation."""
    def _mark_perturbed_nodes(self):
        degree_in = {}
        queue = deque()
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)  # in_degree ==0 -> root node
            print(key, l, l.input_name, l.output_name)

        while len(queue) > 0:
            l = queue.popleft()
            for name_next in l.output_name:
                node_next = self._modules[name_next]
                if isinstance(l, BoundedShape):
                    pass
                else:
                    node_next.perturbed = node_next.perturbed or l.perturbed
                degree_in[name_next] -= 1
                if degree_in[name_next] == 0:  # all inputs of this node have been visited, now put it in queue.
                    queue.append(node_next)



    def _clear_and_set_new(self, new_interval, clear_forward_only=False):
        for l in self._modules.values():
            if clear_forward_only:
                if hasattr(l, 'forward_value'):
                    delattr(l, 'forward_value')         
            else:
                for attr in ['lower', 'upper', 'interval', 'forward_value', 'd', 'lA', 'lower_d']:
                    if hasattr(l, attr):
                        # print(attr)
                        delattr(l, attr)

            for attr in ['zero_backward_coeffs_l', 'zero_backward_coeffs_u', 'zero_lA_mtx', 'zero_uA_mtx']:
                setattr(l, attr, False)

            # Given an interval here to make IBP/CROWN start from this node
            if (new_interval is not None) and (l.name in new_interval.keys()):
                l.interval = tuple(new_interval[l.name][:2])
                l.lower = new_interval[l.name][0]
                l.upper = new_interval[l.name][1]

            # Mark all nodes as non-perturbed except for weights.
            if not hasattr(l, 'perturbation') or l.perturbation is None:
                l.perturbed = False


    def _to(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, device) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, device) for item in obj])
        elif isinstance(obj, dict):
            res = {}
            for key in obj:
                res[key] = self._to(obj[key], device)
            return res
        else:
            raise NotImplementedError(type(obj))

    def _get_node_input(self, nodesOP, nodesIn, node):
        ret = []
        ori_names = []
        for i in range(len(node.inputs)):
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIn:
                if io.name == node.inputs[i]:
                    ret.append(io.bound_node)
                    ori_names.append(io.ori_name)
                    break

            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))

        return ret, ori_names

    def _convert_nodes(self, model, global_input):
        nodesOP, nodesIn, nodesOut, template = parse_module(model, global_input)

        for i in range(0, len(nodesIn)):
            if nodesIn[i].param is not None:
                nodesIn[i] = nodesIn[i]._replace(param=nodesIn[i].param.to(self.device))
        
        global_input_unpacked = unpack_inputs(global_input)

        # Convert input nodes and parameters.
        for i, n in enumerate(nodesIn):
            if n.input_index is not None:
                nodesIn[i] = nodesIn[i]._replace(bound_node=BoundedInput(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=global_input_unpacked[nodesIn[i].input_index],
                    perturbation=nodesIn[i].perturbation))
            else:
                bound_class = BoundedParams if isinstance(nodesIn[i].param, nn.Parameter) else BoundedBuffers 
                nodesIn[i] = nodesIn[i]._replace(bound_node=bound_class(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=nodesIn[i].param, perturbation=nodesIn[i].perturbation))

        # Convert other operation nodes.
        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs, ori_names = self._get_node_input(nodesOP, nodesIn, nodesOP[n])

            if nodesOP[n].op in self.custom_ops:
                op = self.custom_ops[nodesOP[n].op]
            elif nodesOP[n].op in bound_op_map:
                op = bound_op_map[nodesOP[n].op]
            elif nodesOP[n].op.startswith('aten::ATen'):
                op = eval('BoundATen{}'.format(attr['operator'].capitalize()))
            elif nodesOP[n].op.startswith('onnx::'):
                op = eval('Bounded{}'.format(nodesOP[n].op[6:]))
            else:
                raise NotImplementedError(f'There are unsupported operations {nodesOP[n].op}')


            if nodesOP[n].op == 'onnx::BatchNormalization':
                # BatchNormalization node needs model.training flag to set running mean and vars
                # set training=False to avoid wrongly updating running mean/vars during bound wrapper
                nodesOP[n] = nodesOP[n]._replace(bound_node=op(nodesOP[n].inputs, nodesOP[n].name, None, 
                    attr, inputs, nodesOP[n].output_index, self.bound_opts, self.device, False))
            else:
                nodesOP[n] = nodesOP[n]._replace(bound_node=op(nodesOP[n].inputs, nodesOP[n].name, None, 
                    attr, inputs, nodesOP[n].output_index, self.bound_opts, self.device))

        return nodesOP, nodesIn, nodesOut, template


    def _convert(self, model, global_input):
        'convert a Pytorch model to a model with bounds'

        if not isinstance(global_input, tuple):
            global_input = (global_input, )

        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(model, global_input)
        global_input = self._to(global_input, self.device)

        while True:
            self._build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            nodesOP, nodesIn, finished = self._split_complex(nodesOP, nodesIn)
            if finished:
                break

            exit()


    def _build_graph(self, nodesOP, nodesIn, nodesOut, template):
        nodes = [node.bound_node for node in nodesOP + nodesIn]
        self.final_name = nodesOut[0]
        self.input_name, self.input_index, self.root_name = [], [], []
        for node in nodesIn:
            self.root_name.append(node.name)
            if node.input_index is not None:
                self.input_name.append(node.name)
                self.input_index.append(node.input_index)

        self.output_name = nodesOut
        self.output_template = template

        for l in nodes:
            self._modules[l.name] = l
            l.output_name = []
            if isinstance(l.input_name, str):
                l.input_name = [l.input_name]

        for l in nodes:
            for l_pre in l.input_name:
                self._modules[l_pre].output_name.append(l.name)

        for l in nodes:
            if self.conv_mode != 'patches' and len(l.input_name) == 0:
                if not l.name in self.root_name:
                    # Add independent nodes that do not appear in `nodesIn`.
                    # Note that these nodes are added in the last, since 
                    # order matters in the current implementation because 
                    # `root[0]` is used in some places.
                    self.root_name.append(l.name)


    def _split_complex(self, nodesOP, nodesIn):
        finished = True
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n].bound_node, 'complex') and \
                    nodesOP[n].bound_node.complex:
                finished = False
                _nodesOP, _nodesIn, _nodesOut, _template = self._convert_nodes(
                    nodesOP[n].bound_node.model, nodesOP[n].bound_node.input)
                # assuming each supported complex operation only has one output
                assert len(_nodesOut) == 1

                name_base = nodesOP[n].name + '/split'
                rename_dict = {}
                for node in _nodesOP + _nodesIn:
                    rename_dict[node.name] = name_base + node.name
                num_inputs = len(nodesOP[n].bound_node.input)
                for i in range(num_inputs):
                    rename_dict[_nodesIn[i].name] = nodesOP[n].inputs[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                def rename(node):
                    node.bound_node.name = rename_dict[node.bound_node.name]
                    node.bound_node.input_name = [
                        rename_dict[name] for name in node.bound_node.input_name]
                    node = node._replace(
                        name=rename_dict[node.name],
                        inputs=node.bound_node.input_name)                        
                    return node

                for i in range(len(_nodesOP)):
                    _nodesOP[i] = rename(_nodesOP[i])
                for i in range(len(_nodesIn)):
                    _nodesIn[i] = rename(_nodesIn[i])
                output_name = _nodesOP[-1].name
                # Any input node of some node within the complex node should be 
                # replaced with the corresponding input node of the complex node.
                for node in _nodesOP:
                    for i in range(len(node.bound_node.inputs)):
                        if node.bound_node.input_name[i] in nodesOP[n].inputs:
                            index = nodesOP[n].inputs.index(node.bound_node.input_name[i])
                            node.bound_node.inputs[i] = nodesOP[n].bound_node.inputs[index]
                # For any output node of this complex node, modify its input node
                for node in nodesOP:
                    if output_name in node.bound_node.input_name:
                        index = node.bound_node.input_name.index(output_name)
                        node.bound_node.inputs[index] = _nodesOP[-1].bound_node

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIn = nodesIn + _nodesIn[num_inputs:]

                break

        return nodesOP, nodesIn, finished











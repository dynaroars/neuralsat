from collections import OrderedDict, deque, defaultdict
import torch.optim as optim
import torch.nn as nn
from math import prod
import torch
import time
import math
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
            # 'ibp_relative': False, 
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

        self.final_shape = model(*unpack_inputs(global_input, device=self.device)).shape
        self.bound_opts.update({'final_shape': self.final_shape})
        self._convert(model, global_input)
        self._mark_perturbed_nodes()


        # set the default values here
        optimize_bound_args = {'ob_iteration': 5, 'ob_beta': False, 'ob_alpha': True, 'ob_alpha_share_slopes': False,
                               'ob_opt_coeffs': False, 'ob_opt_bias': False,
                               'ob_optimizer': 'adam', 'ob_verbose': 0,
                               'ob_keep_best': True, 'ob_update_by_layer': True, 'ob_lr': 0.5,
                               'ob_lr_beta': 0.05, 'ob_init': True,
                               'ob_single_node_split': True, 'ob_lr_intermediate_beta': 0.1,
                               'ob_lr_coeffs': 0.01, 'ob_intermediate_beta': False, 'ob_intermediate_refinement_layers': [-1],
                               'ob_loss_reduction_func': reduction_sum, 
                               'ob_stop_criterion_func': lambda x: False,
                               'ob_input_grad': False,
                               'ob_lr_decay': 0.98 }
        # change by bound_opts
        optimize_bound_args.update(self.bound_opts.get('optimize_bound_args', {}))
        self.bound_opts.update({'optimize_bound_args': optimize_bound_args})

        # save relu layers for convenience
        self.relus = []  
        for l in self._modules.values():
            if isinstance(l, BoundedRelu):
                self.relus.append(l)

        self.optimizable_activations = []
        for l in self._modules.values():
            if isinstance(l, BoundedOptimizableActivation):
                self.optimizable_activations.append(l)

    def set_bound_opts(self, new_opts):
        for k, v in new_opts.items():
            assert v is not dict, 'only support change optimize_bound_args'
            self.bound_opts[k].update(v)


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

        # # print(queue)

        forward_values = {}
        final_output = None
        while len(queue) > 0:
            l = queue.popleft()
            # # print(l)
            inp = [forward_values[l_pre] for l_pre in l.input_name]
            for l_pre in l.inputs:
                # # print('l_pre', l_pre, l_pre.from_input)
                l.from_input = l.from_input or l_pre.from_input

            fv = l.forward(*inp)

            if isinstance(fv, torch.Size) or isinstance(fv, tuple):
                fv = torch.tensor(fv, device=self.device)

            # # print(l.forward_value)
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
                # print('lllll', l)
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
            # # print(key, l, l.input_name, l.output_name)

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
                        # # print(attr)
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

        # self._get_node_name_map()

        # # load self.ori_state_dict again to avoid the running means/vars changed during forward()
        # self.load_state_dict(self.ori_state_dict)
        # model.load_state_dict(self.ori_state_dict)
        # delattr(self, 'ori_state_dict')

        # The final node used in the last time calling `compute_bounds`
        self.last_final_node = None


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
            if hasattr(nodesOP[n].bound_node, 'complex') and nodesOP[n].bound_node.complex:
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

    def get_optimized_bounds(self, x=None, aux=None, C=None, IBP=False, forward=False, method='backward',
                             bound_lower=True, bound_upper=False, reuse_ibp=False, return_A=False, final_node_name=None,
                             average_A=False, new_interval=None, reference_bounds=None, aux_reference_bounds=None, needed_A_dict=None):

        # print('\n\n================================== get_optimized_bounds ================================== ')
        # optimize CROWN lower bound by alpha and beta
        opts = self.bound_opts['optimize_bound_args']
        iteration = opts['ob_iteration']
        beta = opts['ob_beta']
        alpha = opts['ob_alpha']

        opt_choice = opts['ob_optimizer']
        opt_coeffs = opts['ob_opt_coeffs']
        opt_bias = opts['ob_opt_bias']

        verbose = opts['ob_verbose']

        single_node_split = opts['ob_single_node_split'] 
        
        keep_best = opts['ob_keep_best']
        update_by_layer = opts['ob_update_by_layer']
        init = opts['ob_init']
        lr = opts['ob_lr']
        lr_beta = opts['ob_lr_beta']
        lr_intermediate_beta = opts['ob_lr_intermediate_beta']
        lr_decay = opts['ob_lr_decay']
        lr_coeffs = opts['ob_lr_coeffs'] 

        loss_reduction_func = opts['ob_loss_reduction_func']
        stop_criterion_func = opts['ob_stop_criterion_func']
        input_grad = opts['ob_input_grad']

        sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)

        if C is not None:
            # raise NotImplementedError()
            self.final_shape = C.size()[:2]
            self.bound_opts.update({'final_shape': self.final_shape})

        if init:
            self.init_slope(x, share_slopes=opts['ob_alpha_share_slopes'], method=method, c=C)


        alphas = []
        parameters = []

        for m in self.optimizable_activations:
            if alpha:
                alphas.extend(list(m.alpha.values()))

        if alpha:
            # Alpha has shape (2, output_shape, batch_dim, node_shape)
            parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
            # best_alpha is a dictionary of dictionary. 
            # Each key is the alpha variable for one relu layer, and each value is a dictionary contains all relu layers after that layer as keys.
            best_alphas = OrderedDict()
            for m in self.optimizable_activations:
                best_alphas[m.name] = {}
                for alpha_m in m.alpha:
                    best_alphas[m.name][alpha_m] = m.alpha[alpha_m].clone().detach()
                    # # print(alpha_m, m.alpha[alpha_m].requires_grad)
                    m.alpha[alpha_m].requires_grad_()


        if beta:
            raise

        if opt_choice == "adam-autolr":
            opt = AdamElementLR(parameters, lr=lr)
        elif opt_choice == "adam":
            opt = optim.Adam(parameters, lr=lr)
        elif opt_choice == 'sgd':
            opt = optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(opt_choice)

        # print('\t- optimization:', opt)

        # Create a weight vector to scale learning rate.
        loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        last_l = math.inf
        last_total_loss = torch.tensor(1e8, device=x[0].device, dtype=x[0].dtype)
        best_l = torch.zeros([x[0].shape[0], 1], device=x[0].device, dtype=x[0].dtype) + 1e8

        best_intermediate_bounds = []  # TODO: this should be a dictionary to handle more general architectures.

        if aux_reference_bounds is None:
            aux_reference_bounds = {}

        for i in range(iteration):
            tic = time.time()
            # print('\n\n[+] Iteration ==================================> ', i)
            intermediate_constr = None

            if not update_by_layer:
                reference_bounds = new_interval  # If we still optimize all intermediate neurons, we can use new_interval as reference bounds.

            ret = self.compute_bounds(x, aux, C, method=method, IBP=IBP, forward=forward,
                                    bound_lower=bound_lower, bound_upper=bound_upper, reuse_ibp=reuse_ibp,
                                    return_A=return_A, final_node_name=final_node_name, average_A=average_A,
                                    # If we set neuron bounds individually, or if we are optimizing intermediate layer bounds using beta, we do not set new_interval.
                                    # When intermediate betas are used, we must set new_interval to None because we want to recompute all intermediate layer bounds.
                                    new_interval=partial_new_interval if beta and intermediate_beta_enabled else new_interval if update_by_layer else None,
                                    # This is the currently tightest interval, which will be used to pass split constraints when intermediate betas are used.
                                    reference_bounds=reference_bounds,
                                    # This is the interval used for checking for unstable neurons.
                                    aux_reference_bounds=aux_reference_bounds if sparse_intermediate_bounds else None,
                                    # These are intermediate layer beta variables and their corresponding A matrices and biases.
                                    intermediate_constr=intermediate_constr, needed_A_dict=needed_A_dict)

            if i == 0:
                best_ret = ret
                for model in self.optimizable_activations:
                    best_intermediate_bounds.append([model.inputs[0].lower.clone().detach(), model.inputs[0].upper.clone().detach()])
                    if sparse_intermediate_bounds:
                        aux_reference_bounds[model.inputs[0].name] = best_intermediate_bounds[-1]

        
            ret_l, ret_u = ret[0], ret[1]

            if beta and opt_bias and not single_node_split:
                raise

            l = ret_l
            if ret_l is not None and ret_l.shape[1] != 1:  # Reduction over the spec dimension.
                l = loss_reduction_func(ret_l)

            u = ret_u
            if ret_u is not None and ret_u.shape[1] != 1:
                u = loss_reduction_func(ret_u)

            loss_ = l if bound_lower else -u
            stop_criterion = stop_criterion_func(ret_l) if bound_lower else stop_criterion_func(-ret_u)
            total_loss = -1 * loss_

            if type(stop_criterion) == bool:
                loss = total_loss.sum() * (not stop_criterion)
            else:
                loss = (total_loss * stop_criterion.logical_not()).sum()

            with torch.no_grad():
                # Save varibles if this is the best iteration.
                if keep_best and (total_loss < best_l).any():
                    # we only pick up the results improved in a batch
                    idx = (total_loss < best_l).squeeze()
                    best_l[idx] = total_loss[idx]

                    if ret[0] is not None:
                        best_ret[0][idx] = ret[0][idx]
                    if ret[1] is not None:
                        best_ret[1][idx] = ret[1][idx]

                    if return_A:
                        best_ret = (best_ret[0], best_ret[1], ret[2])

                    for ii, model in enumerate(self.optimizable_activations):
                        best_intermediate_bounds[ii][0][idx] = torch.max(best_intermediate_bounds[ii][0][idx], model.inputs[0].lower[idx])
                        best_intermediate_bounds[ii][1][idx] = torch.min(best_intermediate_bounds[ii][1][idx], model.inputs[0].upper[idx])

                        if alpha:
                            for alpha_m in model.alpha:
                                best_alphas[model.name][alpha_m][:,:,idx] = model.alpha[alpha_m][:,:,idx].clone().detach()
                        if beta and single_node_split:
                            best_betas[ii][idx] = betas[ii][idx].clone().detach()


                    if not single_node_split and beta:
                        raise

            if isinstance(stop_criterion, torch.Tensor) and stop_criterion.all():
                # print(f"\nall verified at {i}th iter")
                break

            current_lr = []
            for param_group in opt.param_groups:
                current_lr.append(param_group['lr'])

            opt.zero_grad(set_to_none=True)

            if input_grad and x[0].ptb.x_L.grad is not None:
                x[0].ptb.x_L.grad = None
                x[0].ptb.x_U.grad = None

            loss.backward()

            if opt_choice == "adam-autolr":
                opt.step(lr_scale=[loss_weight, loss_weight])
            else:
                opt.step()

            if beta:
                raise

            if alpha:
                for m in self.relus:
                    for m_start_node, v in m.alpha.items():
                        v.data = torch.clamp(v.data, 0., 1.)


            with torch.no_grad():
                if beta and opt_choice == "adam-autolr" and i > iteration * 0.2:
                    raise

            scheduler.step()
            last_l = loss.item()
            last_total_loss = total_loss.detach().clone()



        if keep_best:
            # Set all variables to their saved best values.
            with torch.no_grad():
                for idx, model in enumerate(self.optimizable_activations):
                    if alpha:
                        model.alpha = best_alphas[model.name]

                    model.inputs[0].lower.data = best_intermediate_bounds[idx][0].data
                    model.inputs[0].upper.data = best_intermediate_bounds[idx][1].data
                    if beta:
                        raise


        if new_interval is not None and not update_by_layer:
            for l in self._modules.values():
                if l.name in new_interval.keys() and hasattr(l, "lower"):
                    l.lower = torch.max(l.lower, new_interval[l.name][0])
                    l.upper = torch.min(l.upper, new_interval[l.name][1])
                    infeasible_neurons = l.lower > l.upper
                    if infeasible_neurons.any():
                        raise
        # print("best_l after optimization:", best_l.sum().item())

        # print('alpha/beta optimization time:', time.time() - start)

        return best_ret


    def init_slope(self, x, share_slopes=False, method='backward', c=None, bound_lower=True, bound_upper=True):
        if method != 'forward':
            x = x[0]

        # initialize the parameters
        for node in self.optimizable_activations:
            node.opt_init()

        with torch.no_grad():
            if method == 'forward':
                raise
            else:
                l, u = self.compute_bounds(x=(x,), IBP=False, C=c, method=method, return_A=False, bound_lower=bound_lower, bound_upper=bound_upper)

        init_intermediate_bounds = {}
        for node in self.optimizable_activations:
            if method == 'forward':
                raise
            else:
                start_nodes = []
                for nj in self.backward_from[node.name]:
                    # print('[+] backward_from node', nj)
                    if nj.name == self.final_name:
                        size_final = self.final_shape[-1] if c is None else c.size(1)                    
                        start_nodes.append((self.final_name, size_final))       
                        continue                    
                    if share_slopes:
                        # all intermediate neurons from the same layer share the same set of slopes.
                        output_shape = 1
                    elif isinstance(node, BoundedRelu) and node.patch_size and nj.name in node.patch_size:
                        # Patches mode. 
                        # Use output channel size as the spec size. This still shares some alpha, but better than no sharing.
                        # The patch size is [out_ch, batch, out_h, out_w, in_ch, H, W]. We use out_ch as the output shape.
                        output_shape = node.patch_size[nj.name][0]
                    else:
                        output_shape = math.prod(nj.lower.shape[1:])
                    start_nodes.append((nj.name, output_shape))
            node.init_opt_parameters(start_nodes)
            node.opt_start()
            init_intermediate_bounds[node.inputs[0].name] = ([node.inputs[0].lower.detach(), node.inputs[0].upper.detach()])
            # # print(init_intermediate_bounds)

        return l, u, init_intermediate_bounds







    def get_unstable_locations(self, node, aux_reference_bounds):
        # FIXME (09/19): this is for ReLU only!
        unstable_masks = torch.logical_and(aux_reference_bounds[node.name][0] < 0, aux_reference_bounds[node.name][1] > 0)
        # unstable_masks = torch.ones(dtype=torch.bool, size=(batch_size, dim), device=self.device)
        if unstable_masks.ndim > 2:
            # Flatten the conv layer shape.
            unstable_masks = unstable_masks.view(unstable_masks.size(0), -1)
        # For simplicity, merge unstable locations for all elements in this batch. TODO: use individual unstable mask.
        unstable_locs = unstable_masks.sum(dim=0).bool()
        # This is a 1-d indices, shared by all elements in this batch.
        unstable_idx = unstable_locs.nonzero().squeeze(1)
        unstable_size = unstable_idx.numel()
        # # print(f'layer {node.name} unstable {unstable_size} total {node.output_shape}')
        return unstable_idx, unstable_size




    def compute_bounds(self, x=None, aux=None, C=None, method='backward', IBP=False, forward=False, bound_lower=True, bound_upper=True, reuse_ibp=False, return_A=False, needed_A_dict=None, final_node_name=None, average_A=False, new_interval=None, return_b=False, b_dict=None, reference_bounds=None, intermediate_constr=None, alpha_idx=None, aux_reference_bounds=None, need_A_only=False):
        # Several shortcuts.
        method = method.lower() if method is not None else method
        if method == 'ibp':
            # Pure IBP bounds.
            method = None
            IBP = True
        elif method in ['ibp+backward', 'ibp+crown', 'crown-ibp']:
            method = 'backward'
            IBP = True
        elif method == 'crown':
            method = 'backward'
        elif method == 'forward':
            forward = True
        elif method == 'forward+backward':
            method = 'backward'
            forward = True
        elif method in ['crown-optimized', 'alpha-crown']:
            ret = []
            if bound_lower:
                ret1 = self.get_optimized_bounds(x=x, IBP=False, C=C, method='backward', new_interval=new_interval, reference_bounds=reference_bounds,
                                                 bound_lower=bound_lower, bound_upper=False, return_A=False, aux_reference_bounds=aux_reference_bounds,
                                                 needed_A_dict=needed_A_dict)
            if bound_upper:
                ret2 = self.get_optimized_bounds(x=x, IBP=False, C=C, method='backward', new_interval=new_interval, reference_bounds=reference_bounds,
                                                 bound_lower=False, bound_upper=bound_upper, return_A=return_A, aux_reference_bounds=aux_reference_bounds,
                                                 needed_A_dict=needed_A_dict)
            if bound_lower and bound_upper:
                return ret1[0], ret2[1]
            elif bound_lower:
                return ret1
            elif bound_upper:
                return ret2


        if aux_reference_bounds is None:
            aux_reference_bounds = {}

        if reference_bounds is None:
            reference_bounds = {}

        A_dict = {} if return_A else None

        self.backward_from = dict([(node, []) for node in self._modules])

        if x is not None:
            self._set_input(*x, new_interval=new_interval)


        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper

        root = [self._modules[name] for name in self.root_name]
        batch_size = root[0].value.shape[0]
        dim_in = 0

        for i in range(len(root)):
            value = root[i].forward()
            # # print(i, root[i])
            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:   
                # # print('alo', root[i].perturbation)
                root[i].linear, root[i].center, root[i].aux = root[i].perturbation.init(value, aux=aux, forward=forward)
                # This input/parameter has perturbation. Create an interval object.
                root[i].interval = Interval(root[i].linear.lower, root[i].linear.upper, ptb=root[i].perturbation)

                if forward:
                    root[i].dim = root[i].linear.lw.shape[1]
                    dim_in += root[i].dim

            else:
                # This inpute/parameter does not has perturbation. 
                # Use plain tuple defaulting to Linf perturbation.
                root[i].interval = (value, value)
                root[i].forward_value= root[i].value = root[i].lower = root[i].upper = value
                # # print('root w/o perturbation', root[i])

            root[i].lower, root[i].upper = root[i].interval

        if forward:
            self._init_forward(root, dim_in)

        final = self._modules[self.final_name] if final_node_name is None else self._modules[final_node_name]
        if IBP:
            res = self._IBP_general(node=final, C=C)
            self.ibp_lower, self.ibp_upper = res

        if method is None:
            return self.ibp_lower, self.ibp_upper     


        # ######################################################################
        if C is None:
            # C is an identity matrix by default 
            if final.output_shape is None:
                raise ValueError('C is not provided while node {} has no default shape'.format(final.shape))
            dim_output = int(math.prod(final.output_shape[1:]))
            # TODO: use an eyeC object here.
            C = torch.eye(dim_output, device=self.device).expand(batch_size, dim_output, dim_output)


        # check whether weights are perturbed and set nonlinear for the BoundMatMul operation
        for n in self._modules.values():
            if type(n) in [BoundedLinear, BoundedConv, BoundedBatchNormalization]:
                n.nonlinear = False
                for l_name in n.input_name[1:]:
                    node = self._modules[l_name]
                    if hasattr(node, 'perturbation'):
                        if node.perturbation is not None:
                            n.nonlinear = True

        # print('final', final)

        if final != self.last_final_node:
            self.last_final_node = final
            for i in self._modules.values():
                i.used = False
            final.used = True
            queue = deque([final])
            while len(queue) > 0:
                n = queue.popleft()
                for n_pre_name in n.input_name:
                    n_pre = self._modules[n_pre_name]
                    if not n_pre.used:
                        n_pre.used = True
                        queue.append(n_pre)

        for i in self._modules.values():
            if isinstance(i, BoundedRelu):
                for l_name in i.input_name:
                    node = self._modules[l_name]
                    if isinstance(node, BoundedConv):
                        node.relu_followed = True # whether this Conv is followed by a ReLU

        for i in self._modules.values():  # for all nodes
            if not i.used:
                continue

            if hasattr(i, 'nonlinear') and i.nonlinear:
                # print('------------------------- [+] nonlinear', i)
                for l_name in i.input_name:
                    node = self._modules[l_name]
                    # print('\t- ', node, node.perturbed, node.from_input, hasattr(node, 'lower'))
                    if not hasattr(node, 'lower'):
                        if not node.perturbed and hasattr(node, 'forward_value'):
                            node.interval = node.lower, node.upper = node.forward_value, node.forward_value
                            continue

                        if not node.from_input and hasattr(node, 'forward_value'):
                            node.lower = node.upper = node.forward_value
                            continue


                        if forward:
                            raise
                        else:
                            # FIXME: Put ReLU after reshape will cause problem!
                            if (isinstance(node, BoundedActivation) or isinstance(node, BoundedTranspose)) and \
                                    hasattr(self._modules[node.input_name[0]], 'lower'):
                                node.lower = node.forward(self._modules[node.input_name[0]].lower)
                                node.upper = node.forward(self._modules[node.input_name[0]].upper)
                            elif isinstance(node, BoundedReshape) and hasattr(self._modules[node.input_name[0]], 'lower') and \
                                    hasattr(self._modules[node.input_name[1]], 'value'):
                                # Node for input value.
                                val_input = self._modules[node.input_name[0]]
                                # Node for input parameter (e.g., shape, permute)
                                arg_input = self._modules[node.input_name[1]]
                                node.lower = node.forward(val_input.lower, arg_input.value)
                                node.upper = node.forward(val_input.upper, arg_input.value)
                            else:
                                first_layer_flag = False
                                # This is the list of all intermediate layers where we need to refine.
                                # print('\t- ', intermediate_constr)
                                if intermediate_constr is not None:
                                    intermediate_beta_enabled_layers = [k for v in intermediate_constr.values() for k in v]
                                else:
                                    intermediate_beta_enabled_layers = []

                                # Here we avoid creating a big C matrix in the first linear layer.
                                # Disable this optimization when we have beta for intermediate layer bounds.
                                if type(node) == BoundedLinear or type(node) == BoundedConv and node.name not in intermediate_beta_enabled_layers:
                                    for l_pre in node.input_name:
                                        if type(self._modules[l_pre]) == BoundedInput:
                                            node.lower, node.upper = self._IBP_general(node)
                                            first_layer_flag = True
                                            break

                                # print('\t-  first_layer_flag', first_layer_flag)

                                if not first_layer_flag:
                                    reduced_dim = False  # Only partial neurons (unstable neurons) are bounded.
                                    unstable_idx = None
                                    unstable_size = 99999
                                    dim = int(math.prod(node.output_shape[1:]))
                                    # print('\t- ', node, dim)
                                    # print('\t-  node.name:', node.name)
                                    # print('\t-  node.output_name:', node.output_name)
                                    # print('\t-  aux_reference_bounds:', aux_reference_bounds)
                                    # print('\t-  self._modules[node.output_name[0]]:', self._modules[node.output_name[0]])
                                    sparse_intermediate_bounds = node.name in aux_reference_bounds and self.bound_opts.get('sparse_intermediate_bounds', False) and isinstance(self._modules[node.output_name[0]], BoundedRelu)
                                    # print('\t-  sparse_intermediate_bounds:', sparse_intermediate_bounds)
                                    sparse_conv_intermediate_bounds = self.bound_opts.get('sparse_conv_intermediate_bounds', False)

                                    # FIXME: C matrix shape incorrect for BoundParams.

                                    if (isinstance(node, BoundedLinear) or isinstance(node, BoundedMatMul)) and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
                                        if sparse_intermediate_bounds:
                                            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
                                            # Also, if we are checking against LP solver we will refine all neurons and do not use this optimization.
                                            # For each batch element, we find the unstable neurons.
                                            unstable_idx, unstable_size = self.get_unstable_locations(node, aux_reference_bounds)
                                            if unstable_size == 0: # Do nothing, no bounds will be computed.
                                                reduced_dim = True
                                                unstable_idx = []
                                            elif unstable_size < 0.9 * dim and unstable_size > 0:
                                                # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                                                newC = OneHotC([batch_size, unstable_size, *node.output_shape[1:]], self.device, unstable_idx, None)
                                                reduced_dim = True
                                            else:
                                                unstable_idx = None
                                        # print('\t-  reduced_dim:', reduced_dim)
                                        if not reduced_dim:
                                            newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
                                            # print('\t-  newC from', node, newC)

                                    elif (isinstance(node, BoundedConv) or isinstance(node, BoundedBatchNormalization)) and node.mode == "patches":
                                        if sparse_intermediate_bounds:
                                            unstable_idx, unstable_size = self.get_unstable_conv_locations(node, aux_reference_bounds)
                                            if unstable_size == 0:
                                                reduced_dim = True # Do nothing, no bounds will be computed.
                                                unstable_idx = []
                                            # We sum over the channel direction, so need to multiply that.
                                            elif sparse_conv_intermediate_bounds and unstable_size < 0.8 * dim:
                                                # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                                                # The shape of patches is [unstable_size, batch, C, H, W].
                                                newC = Patches(patches=None, stride=1, padding=0, shape=[
                                                    unstable_size, batch_size, node.output_shape[-3], 1, 1],
                                                    identity=1, unstable_idx=unstable_idx, output_shape=node.output_shape)
                                                reduced_dim = True
                                            else:
                                                unstable_idx = None
                                        # Here we create an Identity Patches object
                                        # print('\t-  reduced_dim:', reduced_dim)
                                        if not reduced_dim:
                                            newC = Patches(None, 1, 0,
                                                           [node.output_shape[-3], batch_size, node.output_shape[-2], node.output_shape[-1],
                                                            node.output_shape[-3], 1, 1], 1, output_shape=node.output_shape)
                                        
                                            # print('\t-  newC from', node, newC)

                                    elif isinstance(node, BoundAdd) and node.mode == "patches":
                                        if sparse_intermediate_bounds:
                                            unstable_idx, unstable_size = self.get_unstable_conv_locations(node, aux_reference_bounds)
                                            if unstable_size == 0:
                                                # Do nothing, no bounds will be computed.
                                                reduced_dim = True
                                                unstable_idx = []
                                            elif sparse_conv_intermediate_bounds and unstable_size < 0.8 * dim:
                                                num_channel = node.output_shape[-3]
                                                # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
                                                patches = (torch.eye(num_channel, device=self.device)).view(num_channel, 1, 1, 1, num_channel, 1, 1)
                                                # Expand to (out_c, 1, unstable_size, out_c, 1, 1).
                                                patches = patches.expand(-1, 1, node.output_shape[-2], node.output_shape[-1], -1, 1, 1)
                                                patches = patches[unstable_idx[0], :, unstable_idx[1], unstable_idx[2]]
                                                # Expand with the batch dimension. Final shape (unstable_size, batch_size, out_c, 1, 1).
                                                patches = patches.expand(-1, batch_size, -1, -1, -1)
                                                newC = Patches(patches, 1, 0, patches.shape, unstable_idx=unstable_idx, output_shape=node.output_shape)
                                                reduced_dim = True
                                            else:
                                                unstable_idx = None
                                        if not reduced_dim:
                                            num_channel = node.output_shape[-3]
                                            # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
                                            patches = (torch.eye(num_channel, device=self.device)).view(num_channel, 1, 1, 1, num_channel, 1, 1)
                                            # Expand to (out_c, batch, out_h, out_w, out_c, 1, 1).
                                            patches = patches.expand(-1, batch_size, node.output_shape[-2], node.output_shape[-1], -1, 1, 1)
                                            newC = Patches(patches, 1, 0, patches.shape, output_shape=node.output_shape)
                                            # print('\t-  newC from', node, newC)
                                    else:
                                        if sparse_intermediate_bounds:
                                            unstable_idx, unstable_size = self.get_unstable_locations(node, aux_reference_bounds)
                                            if unstable_size == 0: # Do nothing, no bounds will be computed.
                                                reduced_dim = True
                                                unstable_idx = []
                                            # Number of unstable neurons after merging.
                                            elif unstable_size < 0.9 * dim:
                                                # Create a C matrix.
                                                newC = torch.zeros([1, unstable_size, dim], device=self.device)
                                                # Fill the corresponding elements to 1.0
                                                newC[0, torch.arange(unstable_size), unstable_idx] = 1.0
                                                newC = newC.expand(batch_size, -1, -1).view(batch_size, unstable_size, *node.output_shape[1:])
                                                reduced_dim = True
                                                # # print(f'layer {node.name} total {dim} unstable {unstable_size} newC {newC.size()}')
                                            else:
                                                unstable_idx = None
                                        if not reduced_dim:
                                            if dim > 1000:
                                                raise
                                            newC = torch.eye(dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, dim, *node.output_shape[1:])
                                            # print('\t-  newC from', node, newC)

                                    # print('\t-  unstable_idx:', unstable_idx)
                                    # print('\t-  unstable_size:', unstable_size)
                                    if unstable_idx is None or unstable_size > 0:
                                        self._backward_general(C=newC, node=node, root=root, return_A=False, intermediate_constr=intermediate_constr, unstable_idx=unstable_idx)

                                    if reduced_dim:
                                        if unstable_size > 0:
                                            # If we only calculated unstable neurons, we need to scatter the results back based on reference bounds.
                                            if isinstance(unstable_idx, tuple):
                                                new_lower = aux_reference_bounds[node.name][0].detach().clone()
                                                new_upper = aux_reference_bounds[node.name][1].detach().clone()
                                                # Conv layer with patches, the unstable_idx is a 3-element tuple for 3 indices (C, H,W) of unstable neurons.
                                                new_lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = node.lower
                                                new_upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = node.upper
                                            else:
                                                # Other layers.
                                                new_lower = aux_reference_bounds[node.name][0].detach().clone().view(batch_size, -1)
                                                new_upper = aux_reference_bounds[node.name][1].detach().clone().view(batch_size, -1)
                                                new_lower[:, unstable_idx] = node.lower.view(batch_size, -1)
                                                new_upper[:, unstable_idx] = node.upper.view(batch_size, -1)
                                            # # print(f'{node.name} {node} bound diff {(new_lower.view(-1) - aux_reference_bounds[node.name][0].view(-1)).abs().sum()} {(new_upper.view(-1) - aux_reference_bounds[node.name][1].view(-1)).abs().sum()}')
                                            node.lower = new_lower.view(batch_size, *node.output_shape[1:])
                                            node.upper = new_upper.view(batch_size, *node.output_shape[1:])
                                        else:
                                            # No unstable neurons. Skip the update.
                                            node.lower = aux_reference_bounds[node.name][0].detach().clone()
                                            node.upper = aux_reference_bounds[node.name][1].detach().clone()
                                    # node.lower and node.upper (intermediate bounds) are computed in the above function.
                                    # If we have bound references, we set them here to always obtain a better set of bounds.
                                    if node.name in reference_bounds:
                                        # Initially, the reference bound and the computed bound can be exactly the same when intermediate layer beta is 0. This will prevent gradients flow. So we need a small guard here.
                                        if intermediate_constr is not None:
                                            # Intermediate layer beta is used.
                                            # Note that we cannot just take the reference bounds if they are better - this makes alphas have zero gradients.
                                            node.lower = torch.max((0.9 * reference_bounds[node.name][0] + 0.1 * node.lower), node.lower)
                                            node.upper = torch.min((0.9 * reference_bounds[node.name][1] + 0.1 * node.upper), node.upper)
                                            # Additionally, if the reference bounds say a neuron is stable, we always keep it. (FIXME: this is for ReLU only).
                                            lower_stable = reference_bounds[node.name][0] >= 0.
                                            node.lower[lower_stable] = reference_bounds[node.name][0][lower_stable]
                                            upper_stable = reference_bounds[node.name][1] <= 0.
                                            node.upper[upper_stable] = reference_bounds[node.name][1][upper_stable]
                                        else:
                                            # MIP solved intermediate layer bounds.
                                            # Set the intermediate layer bounds using reference bounds, always choosing the tighter one.
                                            node.lower = torch.max(reference_bounds[node.name][0] - 1e-5, node.lower)
                                            node.upper = torch.min(reference_bounds[node.name][1] + 1e-5, node.upper)
                                        # Otherwise, we only use reference bounds to check which neurons are unstable.


        # print('\n\nBackward outside from {} ({})'.format(final, final.name))
        # print('\t-  unstable_idx:', unstable_idx)
        return self._backward_general(C=C, node=final, root=root, bound_lower=bound_lower, bound_upper=bound_upper,
                                      return_A=return_A, needed_A_dict=needed_A_dict, average_A=average_A, A_dict=A_dict,
                                      return_b=return_b, b_dict=b_dict, unstable_idx=alpha_idx, need_A_only=need_A_only)

    def _backward_general(self, C=None, node=None, root=None, bound_lower=True, bound_upper=True, return_A=False, needed_A_dict=None, average_A=False, A_dict=None, return_b=False, b_dict=None, intermediate_constr=None, unstable_idx=None, need_A_only=False):
        # print('\n\nBackward from {} ({})'.format(node, node.name))

        degree_out = {}
        for l in self._modules.values():
            l.bounded = True
            l.lA = l.uA = None
            degree_out[l.name] = 0

        queue = deque([node])
        all_nodes_before = []
        while len(queue) > 0:
            l = queue.popleft()
            # print('\t - queue:', l)
            self.backward_from[l.name].append(node)
            for l_pre in l.input_name:
                all_nodes_before.append(l_pre)
                degree_out[l_pre] += 1  # calculate the out degree
                if self._modules[l_pre].bounded:
                    self._modules[l_pre].bounded = False
                    queue.append(self._modules[l_pre])

        node.bounded = True


        # print('\t- C', C)

        if isinstance(C, Patches):
            if C.unstable_idx is None:
                # Patches have size (out_c, batch, out_h, out_w, c, h, w).
                out_c, batch_size, out_h, out_w = C.shape[:4]
                output_dim = out_c * out_h * out_w
            else:
                # Patches have size (unstable_size, batch, c, h, w).
                output_dim, batch_size = C.shape[:2]
        else:
            batch_size, output_dim = C.shape[:2]


        # print('\t- batch_size', batch_size)
        # print('\t- output_dim', output_dim)
            
        # The C matrix specified by the user has shape (batch, spec) but internally we have (spec, batch) format.
        if not isinstance(C, (eyeC, Patches, OneHotC)):
            C = C.transpose(0, 1)
        elif isinstance(C, eyeC):
            C = C._replace(shape=(C.shape[1], C.shape[0], *C.shape[2:]))
        elif isinstance(C, OneHotC):
            C = C._replace(shape=(C.shape[1], C.shape[0], *C.shape[2:]), index=C.index.transpose(0,-1), coeffs=None if C.coeffs is None else C.coeffs.transpose(0,-1))


        # Initially, l.lA or l.uA will be set to C for this node.
        node.lA = C if bound_lower else None
        node.uA = C if bound_upper else None
        lb = ub = torch.tensor(0., device=self.device)

        # Save intermediate layer A matrices when required.
        A_record = {}

        queue = deque([node])
        while len(queue) > 0:
            l = queue.popleft()  # backward from l
            # print('[+] processing', l)
            l.bounded = True

            if return_b:
                b_dict[l.name] = { 'lower_b': lb, 'upper_b': ub }

            if l.name in self.root_name or l == root: 
                continue

            for l_pre in l.input_name:  # if all the succeeds are done, then we can turn to this node in the next iteration.
                _l = self._modules[l_pre]
                degree_out[l_pre] -= 1
                if degree_out[l_pre] == 0:
                    queue.append(_l)

            # # print('\t- lA', l, l.lA)
            # # print('\t- uA', l, l.uA)
            if l.lA is not None or l.uA is not None:

                # Propagate lA and uA to a preceding node 
                def add_bound(node, lA, uA):
                    if lA is not None:
                        if node.lA is None:
                            # First A added to this node.
                            node.zero_lA_mtx = l.zero_backward_coeffs_l
                            node.lA = lA
                        else:
                            node.zero_lA_mtx = node.zero_lA_mtx and l.zero_backward_coeffs_l
                            node.lA = self._addA(node.lA, lA)

                    if uA is not None:
                        if node.uA is None:
                            # First A added to this node.
                            node.zero_uA_mtx = l.zero_backward_coeffs_u
                            node.uA = uA
                        else:
                            node.zero_uA_mtx = node.zero_uA_mtx and l.zero_backward_coeffs_u
                            node.uA = self._addA(node.uA, uA)

                # FIXME make fixed nodes have fixed `forward_value` that is never cleaned out
                # print('\t- perturbed', l.perturbed)
                if not l.perturbed and hasattr(l, 'forward_value'):
                    lb = lb + l.get_bias(l.lA, l.forward_value)  # FIXME (09/16): shape for the bias of BoundConstant.
                    ub = ub + l.get_bias(l.uA, l.forward_value)
                    continue

                # print('\t- zero_lA_mtx', l.zero_lA_mtx)
                # print('\t- zero_uA_mtx', l.zero_uA_mtx)
                if l.zero_uA_mtx and l.zero_lA_mtx:
                    # A matrices are all zero, no need to propagate.
                    # print('\t- All zeros')
                    continue

                if isinstance(l, BoundedRelu):
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *l.inputs, start_node=node, unstable_idx=unstable_idx,
                                                           beta_for_intermediate_layers=intermediate_constr is not None)  # TODO: unify this interface.
                elif isinstance(l, BoundedOptimizableActivation):
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *l.inputs,
                    start_shape=(math.prod(node.output_shape[1:]) if node.name != self.final_name else C.shape[0]), start_node=node)
                else:
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *l.inputs)


                if lb.ndim > 0 and type(lower_b) == torch.Tensor and self.conv_mode == 'patches':
                    # When we use patches mode, it's possible that we need to add two bias
                    # one is from the Tensor mode and one is from the patches mode
                    # And we need to detect this case and reshape the bias
                    if lower_b.ndim < lb.ndim:
                        lb = lb.transpose(0,1).reshape(lb.size(1), lb.size(0), -1)
                        lb = lb.expand(lb.size(0), lb.size(1), lower_b.size(0)//lb.size(1))
                        lb = lb.reshape(lb.size(0), -1).t()
                        ub = ub.transpose(0,1).reshape(ub.size(1), ub.size(0), -1)
                        ub = ub.expand(ub.size(0), ub.size(1), upper_b.size(0)//ub.size(1))
                        ub = ub.reshape(ub.size(0), -1).t()
                    elif lower_b.ndim > lb.ndim:
                        lower_b = lower_b.transpose(0,1).reshape(lower_b.size(1), -1).t()
                        upper_b = upper_b.transpose(0,1).reshape(upper_b.size(1), -1).t()

                lb = lb + lower_b
                ub = ub + upper_b



                # print(lb.sum())
                # print(ub.sum())


                for i, l_pre in enumerate(l.input_name):
                    _l = self._modules[l_pre]
                    # # print('==================>', l, l_pre, _l, i)
                    add_bound(_l, lA=A[i][0], uA=A[i][1])


        if lb.ndim >= 2:
            lb = lb.transpose(0, 1)
        if ub.ndim >= 2:
            ub = ub.transpose(0, 1)

                # exit()
        # print(root)
        for i in range(len(root)):
            if root[i].lA is None and root[i].uA is None: 
                continue
            if average_A and isinstance(root[i], BoundedParams):
                lA = root[i].lA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].lA.shape) if bound_lower else None
                uA = root[i].uA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].uA.shape) if bound_upper else None
            else:
                lA, uA = root[i].lA, root[i].uA

            if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
                lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None

            if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].uA, Patches):
                uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None

            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
                # print(root[i], root[i].perturbation)
                if isinstance(root[i], BoundedParams):
                    raise NotImplementedError()
                else:
                    lb = lb + root[i].perturbation.concretize(root[i].center, lA, sign=-1, aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(root[i].center, uA, sign=+1, aux=root[i].aux) if bound_upper else None

            else:
                if i < self.num_global_inputs:
                    # Input node so there is a batch dimension
                    fv = root[i].forward_value.view(batch_size, -1, 1)
                    batch_size_ = batch_size
                else:
                    # Parameter node so there is no batch dimension
                    fv = root[i].forward_value.view(-1, 1)
                    batch_size_ = 1
                if isinstance(lA, eyeC):
                    lb = lb + fv.view(batch_size_, -1) if bound_lower else None
                else:
                    lb = lb + lA.matmul(fv).squeeze(-1) if bound_lower else None
                if isinstance(uA, eyeC):
                    ub = ub + fv.view(batch_size_, -1) if bound_upper else None
                else:
                    ub = ub + uA.matmul(fv).squeeze(-1) if bound_upper else None

        if isinstance(C, Patches) and C.unstable_idx is not None:
            # Sparse patches; the output shape is (unstable_size, ).
            output_shape = [C.shape[0]]
        elif math.prod(node.output_shape[1:]) != output_dim and not isinstance(C, Patches):
            # For the output node, the shape of the bound follows C 
            # instead of the original output shape
            # TODO Maybe don't set node.lower and node.upper in this case?
            # Currently some codes still depend on node.lower and node.upper
            output_shape = [-1]
        else:
            # Generally, the shape of the bounds match the output shape of the node
            output_shape = node.output_shape[1:]
        lb = node.lower = lb.view(batch_size, *output_shape) if bound_lower else None
        ub = node.upper = ub.view(batch_size, *output_shape) if bound_upper else None

        if return_A: 
            # return lb, ub, A_dict
            raise

        return lb, ub



    def _IBP_general(self, node=None, C=None):

        if self.bound_opts.get('loss_fusion', False):
            res = self._IBP_loss_fusion(node, C)
            if res is not None:
                return res

        # # print('\n\n[+] processing node', node)
        if not node.perturbed and hasattr(node, 'forward_value'):
            # # print('w/o forward_value')
            node.lower, node.upper = node.interval = (node.forward_value, node.forward_value)

        if not hasattr(node, 'interval'):
            # # print('w/o interval')
            for n in node.inputs:
                if not hasattr(n, 'interval'):
                    # print('interval', n, n.name)
                    self._IBP_general(n)
            # # print('continue processing node', node)

            inp = [n_pre.interval for n_pre in node.inputs]
            if C is not None and isinstance(node, BoundLinear) and not node.is_input_perturbed(1):
                return node.interval_propagate(*inp, C=C)

            node.interval = node.interval_propagate(*inp)

            node.lower, node.upper = node.interval
            if isinstance(node.lower, torch.Size):
                raise
                node.lower = torch.tensor(node.lower)
                node.interval = (node.lower, node.upper)
            if isinstance(node.upper, torch.Size):
                raise
                node.upper = torch.tensor(node.upper)
                node.interval = (node.lower, node.upper)

        if C is not None:
            return BoundedLinear.interval_propagate(None, node.interval, C=C)
        else:
            return node.interval

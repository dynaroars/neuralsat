import copy
import gurobipy as grb
import math
import time
import torch

from itertools import product
from plnn_bounds.modules import View, Flatten
from plnn_bounds.naive_approximation import NaiveNetwork
from torch import nn
from torch.nn import functional as F

class LinearizedNetwork(NaiveNetwork):
    """
    Approximate a ReLU-based neural network via the PLANET (convex) relaxation. Computes network bounds using Gurobi.
    """
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)
        # Skip all gradient computation for the weights of the Net
        for param in self.net.parameters():
            param.requires_grad = False

    def get_lower_bound(self, domain, force_optim=False):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain, force_optim)
        return self.compute_lower_bound(node=(-1, 0))

    def compute_lower_bound(self, node=(-1, None), upper_bound=False, counterexample_verification=False,
                            time_limit_per_layer=None, ub_only=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        ub_only: (optional) Compute upper bounds only, meaningful only when node[1] = None
        '''
        layer_with_var_to_opt = self.prerelu_gurobi_vars[node[0]]
        is_batch = (node[1] is None)

        device = self.input_domain.device
        self.lb_input = []  # store lower bound input

        # List of time limits per bounds.
        time_limits = copy.copy(time_limit_per_layer)
        if time_limits is not None:
            # The only way we can impose time_limits is if we have access to
            # intermediate values so that we can extract them before
            # terminating the model (on the LP, it's not possible to get
            # intermediate result if the model hasn't gone to convergence)

            # For that, we need to enforce that Gurobi uses the dual simplex
            # algorithm. This line of code imposes that.
            self.model.setParam('Method', 1)  # 1 means dual simplex

        if time_limits is not None:
            layer_budget = time_limits.pop(0)
            if not is_batch:
                nb_opt = 1
            else:
                nb_opt = 2 * self.lower_bounds[node[0]].numel()
            opt_model = lambda: optimize_model(self.model, layer_budget / float(nb_opt))
        else:
            opt_model = lambda: optimize_model(self.model, None)

        if not is_batch:
            if isinstance(node[1], int):
                var_to_opt = layer_with_var_to_opt[node[1]]
            elif (isinstance(node[1], tuple) and isinstance(layer_with_var_to_opt, list)):
                # This is the nested list format
                to_query = layer_with_var_to_opt
                for idx in node[1]:
                    to_query = to_query[idx]
                var_to_opt = to_query
            else:
                raise NotImplementedError

            opt_direct = grb.GRB.MAXIMIZE if upper_bound else grb.GRB.MINIMIZE
            # We will make sure that the objective function is properly set up
            self.model.setObjective(var_to_opt, opt_direct)

            # We will now compute the requested lower bound
            c_b = opt_model()

            return torch.tensor(c_b, device=device)
        else:
            print("Batch Gurobi stuff")
            new_lbs = []
            new_ubs = []
            if isinstance(layer_with_var_to_opt, list):
                for var_idx, var in enumerate(layer_with_var_to_opt):
                    # Do the maximizing
                    self.model.setObjective(var, grb.GRB.MAXIMIZE)
                    c_ub = opt_model()
                    new_ubs.append(c_ub)
                    if not ub_only:
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        c_lb = opt_model()
                        new_lbs.append(c_lb)
                    else:
                        print("skipping lower bounds")
            else:
                new_lbs = self.lower_bounds[node[0]].clone()
                new_ubs = self.upper_bounds[node[0]].clone()
                bound_shape = new_lbs.shape
                for chan_idx, row_idx, col_idx in product(range(bound_shape[0]),
                                                          range(bound_shape[1]),
                                                          range(bound_shape[2])):
                    var = layer_with_var_to_opt[chan_idx, row_idx, col_idx]

                    # Do the maximizing
                    self.model.setObjective(var, grb.GRB.MAXIMIZE)
                    c_ub = opt_model()
                    new_ubs[chan_idx, row_idx, col_idx] = c_ub
                    # print(f"UB was {curr_ub}, now is {new_ubs[chan_idx, row_idx, col_idx]}")
                    if not ub_only:
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        c_lb = opt_model()
                        new_lbs[chan_idx, row_idx, col_idx] = c_lb
                        # print(f"LB was {curr_lb}, now is {new_lbs[chan_idx, row_idx, col_idx]}")
                    else:
                        print("skipping lower bounds")

            return torch.tensor(new_lbs, device=device), torch.tensor(new_ubs, device=device)

    def build_model_using_bounds(self, input_domain, intermediate_bounds, n_threads=1):
        # Create the network model via intermediate bounds passed as arguments.
        self.input_domain = input_domain
        self.gurobi_vars = []
        self.prerelu_gurobi_vars = []
        self.lower_bounds, self.upper_bounds = copy.deepcopy(intermediate_bounds)

        device = self.input_domain.device

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', n_threads)

        ## Do the input layer, which is a special case
        inp_lbs, inp_ubs, inp_gurobi_vars = self.create_input_variables(input_domain)
        self.lower_bounds[0] = torch.tensor(inp_lbs, device=device)
        self.upper_bounds[0] = torch.tensor(inp_ubs, device=device)
        self.gurobi_vars.append(inp_gurobi_vars)
        self.prerelu_gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        x_idx = 1
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                pre_vars = self.gurobi_vars[x_idx - 1]
                if self.lower_bounds[x_idx - 1].dim() > 1:
                    pre_vars = []
                    for chan_idx in range(len(self.gurobi_vars[x_idx - 1])):
                        for row_idx in range(len(self.gurobi_vars[x_idx - 1][chan_idx])):
                            pre_vars.extend(self.gurobi_vars[x_idx - 1][chan_idx][row_idx])

                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, pre_vars)

                    v = self.model.addVar(lb=self.lower_bounds[x_idx][neuron_idx],
                                          ub=self.upper_bounds[x_idx][neuron_idx],
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    new_layer_gurobi_vars.append(v)
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lbs = self.lower_bounds[x_idx - 1].unsqueeze(0)
                out_lbs = self.lower_bounds[x_idx].unsqueeze(0)

                # The first layer doesn't have any optimization, so it doesn't take any budget
                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx >= pre_lbs.size(2)):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx >= pre_lbs.size(3)):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        lin_expr += coeff * self.gurobi_vars[x_idx - 1][in_chan_idx][in_row_idx][
                                            in_col_idx]

                            v = self.model.addVar(lb=self.lower_bounds[x_idx][out_chan_idx][out_row_idx][out_col_idx],
                                                  ub=self.upper_bounds[x_idx][out_chan_idx][out_row_idx][out_col_idx],
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(v == lin_expr)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                print("doing relu")
                if isinstance(self.prerelu_gurobi_vars[x_idx][0], list):
                    print("doing conv relu")
                    # This is convolutional
                    for chan_idx, channel in enumerate(self.prerelu_gurobi_vars[x_idx]):
                        chan_vars = []
                        for row_idx, row in enumerate(channel):
                            row_vars = []
                            for col_idx, pre_var in enumerate(row):
                                pre_lb = self.lower_bounds[x_idx][chan_idx, row_idx, col_idx].item()
                                pre_ub = self.upper_bounds[x_idx][chan_idx, row_idx, col_idx].item()

                                v = self.model.addVar(lb=max(0, pre_lb),
                                                      ub=max(0, pre_ub),
                                                      obj=0, vtype=grb.GRB.CONTINUOUS,
                                                      name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                if pre_lb >= 0 and pre_ub >= 0:
                                    # ReLU is always passing
                                    lb = pre_lb
                                    ub = pre_ub
                                    self.model.addConstr(v == pre_var)
                                elif pre_lb <= 0 and pre_ub <= 0:
                                    lb = 0
                                    ub = 0
                                else:
                                    lb = 0
                                    ub = pre_ub
                                    self.model.addConstr(v >= pre_var)
                                    slope = pre_ub / (pre_ub - pre_lb)
                                    bias = - pre_lb * slope
                                    self.model.addConstr(v <= slope * pre_var + bias)
                                self.model.update()
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    assert isinstance(self.prerelu_gurobi_vars[x_idx][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.prerelu_gurobi_vars[x_idx]):
                        pre_lb = self.lower_bounds[x_idx][neuron_idx]
                        pre_ub = self.upper_bounds[x_idx][neuron_idx]

                        v = self.model.addVar(lb=max(0, pre_lb),
                                              ub=max(0, pre_ub),
                                              obj=0, vtype=grb.GRB.CONTINUOUS,
                                              name=f'ReLU{layer_idx}_{neuron_idx}')
                        if pre_lb >= 0 and pre_ub >= 0:
                            # The ReLU is always passing
                            self.model.addConstr(v == pre_var)
                            lb = pre_lb
                            ub = pre_ub
                        elif pre_lb <= 0 and pre_ub <= 0:
                            lb = 0
                            ub = 0
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                        else:
                            lb = 0
                            ub = pre_ub
                            self.model.addConstr(v >= pre_var)

                            slope = pre_ub / (pre_ub - pre_lb)
                            bias = - pre_lb * slope
                            self.model.addConstr(v <= slope.item() * pre_var + bias.item())
                        self.model.update()
                        new_layer_gurobi_vars.append(v)
                self.gurobi_vars.append(new_layer_gurobi_vars)
                x_idx += 1
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                continue
            else:
                raise NotImplementedError

            layer_idx += 1

        self.model.update()

    def define_linear_approximation(self, input_domain, force_optim=False, time_limit_per_layer=None, n_threads=1):
        # Create the network model from scratch, computing all network bounds (intermediate and final).
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        :param n_threads: number of threads to use in the solution of each Gurobi model
        '''
        self.lower_bounds = []
        self.upper_bounds = []
        self.gurobi_vars = []
        self.prerelu_gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', n_threads)

        # List of time limits per bounds.
        time_limits = copy.copy(time_limit_per_layer)
        if time_limits is not None:
            # The only way we can impose time_limits is if we have access to
            # intermediate values so that we can extract them before
            # terminating the model (on the LP, it's not possible to get
            # intermediate result if the model hasn't gone to convergence)

            # For that, we need to enforce that Gurobi uses the dual simplex
            # algorithm. This line of code imposes that.
            self.model.setParam('Method', 1)  # 1 means dual simplex

        ## Do the input layer, which is a special case
        zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                     vtype=grb.GRB.CONTINUOUS,
                                     name=f'zero')

        ## Do the input layer, which is a special case
        inp_lbs, inp_ubs, inp_gurobi_vars = self.create_input_variables(input_domain)
        self.lower_bounds.append(torch.tensor(inp_lbs))
        self.upper_bounds.append(torch.tensor(inp_ubs))
        self.gurobi_vars.append(inp_gurobi_vars)
        self.prerelu_gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        for layer in self.layers:
            is_final = (layer is self.layers[-1])
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                pre_lb = self.lower_bounds[-1]
                pre_ub = self.upper_bounds[-1]
                pre_vars = self.gurobi_vars[-1]
                if pre_lb.dim() > 1:
                    pre_lb = pre_lb.view(-1)
                    pre_ub = pre_ub.view(-1)
                    pre_vars = []
                    for chan_idx in range(len(self.gurobi_vars[-1])):
                        for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                            pre_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
                if layer_idx > 1:
                    # The previous bounds are from a ReLU
                    pre_lb = torch.clamp(pre_lb, 0, None)
                    pre_ub = torch.clamp(pre_ub, 0, None)
                pos_w = torch.clamp(layer.weight, 0, None)
                neg_w = torch.clamp(layer.weight, None, 0)
                out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
                out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias

                # The first layer doesn't have any optimization, so it doesn't take any budget
                should_timelimit_layer = (layer_idx > 1) and (time_limits is not None)
                if should_timelimit_layer:
                    layer_budget = time_limits.pop(0)
                    if force_optim or is_final:
                        nb_opt = 2 * out_ubs.numel()
                    else:
                        nb_opt = 2 * ((out_lbs < 0) & (out_ubs > 0)).sum()
                    opt_model = lambda: optimize_model(self.model, layer_budget / float(nb_opt))
                    layer_start_time = time.time()
                else:
                    opt_model = lambda: optimize_model(self.model, None)

                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, pre_vars)

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    should_opt = (force_optim
                                  or is_final
                                  or ((layer_idx > 1) and (out_lb < 0) and (out_ub > 0))
                    )
                    if should_opt:
                        self.model.setObjective(v, grb.GRB.MINIMIZE)
                        out_lb = opt_model()

                        # Let's now compute an upper bound
                        self.model.setObjective(v, grb.GRB.MAXIMIZE)
                        out_ub = opt_model()

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(v)
                if should_timelimit_layer:
                    layer_end_time = time.time()
                    time_used = layer_end_time - layer_start_time
                    print(f"[GRB]Used {time_used} for layer {layer_idx}")
                    # Report the unused time for the next layer
                    if time_limits:
                        time_limits[0] += max(0, time_used - layer_budget)
                self.lower_bounds.append(torch.tensor(new_layer_lb))
                self.upper_bounds.append(torch.tensor(new_layer_ub))
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb = self.lower_bounds[-1].unsqueeze(0)
                pre_ub = self.upper_bounds[-1].unsqueeze(0)
                if layer_idx > 1:
                    # The previous bounds are from a ReLU
                    pre_lb = torch.clamp(pre_lb, 0, None)
                    pre_ub = torch.clamp(pre_ub, 0, None)
                pos_weight = torch.clamp(layer.weight, 0, None)
                neg_weight = torch.clamp(layer.weight, None, 0)

                out_lbs = (F.conv2d(pre_lb, pos_weight, layer.bias,
                                    layer.stride, layer.padding, layer.dilation, layer.groups)
                           + F.conv2d(pre_ub, neg_weight, None,
                                      layer.stride, layer.padding, layer.dilation, layer.groups))
                out_ubs = (F.conv2d(pre_ub, pos_weight, layer.bias,
                                    layer.stride, layer.padding, layer.dilation, layer.groups)
                           + F.conv2d(pre_lb, neg_weight, None,
                                      layer.stride, layer.padding, layer.dilation, layer.groups))

                # The first layer doesn't have any optimization, so it doesn't take any budget
                should_timelimit_layer = (layer_idx > 1) and (time_limits is not None)
                if should_timelimit_layer:
                    layer_budget = time_limits.pop(0)
                    if force_optim or is_final:
                        nb_opt = 2 * out_ubs.numel()
                    else:
                        nb_opt = 2 * ((out_lbs < 0) & (out_ubs > 0)).sum()
                    opt_model = lambda: optimize_model(self.model, layer_budget / float(nb_opt))
                    layer_start_time = time.time()
                else:
                    opt_model = lambda: optimize_model(self.model, None)

                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_lbs = []
                    out_chan_ubs = []
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_lbs = []
                        out_row_ubs = []
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0]*out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx >= pre_lb.size(2)):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1]*out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx >= pre_lb.size(3)):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        if abs(coeff) > 1e-6:
                                            lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()

                            v = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(v == lin_expr)
                            self.model.update()

                            should_opt = (force_optim
                                          or is_final
                                          or ((layer_idx > 1) and (out_lb < 0) and (out_ub > 0))
                                          )
                            if should_opt:
                                # Let's now compute a lower bound
                                self.model.setObjective(v, grb.GRB.MINIMIZE)
                                out_lb = opt_model()
                                # Let's now compute an upper bound
                                self.model.setObjective(v, grb.GRB.MAXIMIZE)
                                out_ub = opt_model()

                            out_row_vars.append(v)
                            out_row_lbs.append(out_lb)
                            out_row_ubs.append(out_ub)
                        out_chan_vars.append(out_row_vars)
                        out_chan_lbs.append(out_row_lbs)
                        out_chan_ubs.append(out_row_ubs)
                    new_layer_gurobi_vars.append(out_chan_vars)
                    new_layer_lb.append(out_chan_lbs)
                    new_layer_ub.append(out_chan_ubs)

                if should_timelimit_layer:
                    layer_end_time = time.time()
                    time_used = layer_end_time - layer_start_time
                    # Report the unused time for the next layer if there is one
                    print(f"[GRB] Used {time_used} for layer {layer_idx}")
                    if time_limits:
                        time_limits[0] += max(0, time_used - layer_budget)
                self.lower_bounds.append(torch.tensor(new_layer_lb))
                self.upper_bounds.append(torch.tensor(new_layer_ub))
                self.prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional
                    pre_lbs = torch.Tensor(self.lower_bounds[-1])
                    pre_ubs = torch.Tensor(self.upper_bounds[-1])
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
                        chan_vars = []
                        chan_lbs = []
                        chan_ubs = []
                        for row_idx, row in enumerate(channel):
                            row_vars = []
                            row_lbs = []
                            row_ubs = []
                            for col_idx, pre_var in enumerate(row):
                                pre_lb = pre_lbs[chan_idx, row_idx, col_idx].item()
                                pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()

                                if pre_lb >= 0 and pre_ub >= 0:
                                    # ReLU is always passing
                                    lb = pre_lb
                                    ub = pre_ub
                                    v = pre_var
                                elif pre_lb <= 0 and pre_ub <= 0:
                                    lb = 0
                                    ub = 0
                                    v = zero_var
                                else:
                                    lb = 0
                                    ub = pre_ub
                                    v = self.model.addVar(lb=lb, ub=ub,
                                                          obj=0, vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    self.model.addConstr(v >= pre_var)
                                    slope = pre_ub / (pre_ub - pre_lb)
                                    bias = - pre_lb * slope
                                    self.model.addConstr(v <= slope*pre_var + bias)
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                        pre_lb = self.lower_bounds[-1][neuron_idx]
                        pre_ub = self.upper_bounds[-1][neuron_idx]

                        v = self.model.addVar(lb=max(0, pre_lb),
                                              ub=max(0, pre_ub),
                                              obj=0, vtype=grb.GRB.CONTINUOUS,
                                              name=f'ReLU{layer_idx}_{neuron_idx}')
                        if pre_lb >= 0 and pre_ub >= 0:
                            # The ReLU is always passing
                            self.model.addConstr(v == pre_var)
                            lb = pre_lb
                            ub = pre_ub
                        elif pre_lb <= 0 and pre_ub <= 0:
                            lb = 0
                            ub = 0
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                        else:
                            lb = 0
                            ub = pre_ub
                            self.model.addConstr(v >= pre_var)

                            slope = pre_ub / (pre_ub - pre_lb)
                            bias = - pre_lb * slope
                            self.model.addConstr(v <= slope.item() * pre_var + bias.item())

                        new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                continue
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        self.model.update()

    def create_input_variables(self, input_domain):
        """
        Function to create, given its domain, the Gurobi variables for the network input. These are added to the model.
        :param input_domain: Tensor containing in each row the lower and upper bound for the corresponding input dimension
        :return: input lower bounds (list), input upper bounds (list), input Gurobi vars (list)
        the dimensionality of the output list depends on whether the input layer is convolutional or linear
        """

        inp_lbs = []
        inp_ubs = []
        inp_gurobi_vars = []
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
                inp_lbs.append(lb)
                inp_ubs.append(ub)
        else:
            assert input_domain.dim() == 4
            for chan in range(input_domain.size(0)):
                chan_vars = []
                chan_lbs = []
                chan_ubs = []
                for row in range(input_domain.size(1)):
                    row_vars = []
                    row_lbs = []
                    row_ubs = []
                    for col in range(input_domain.size(2)):
                        lb = input_domain[chan, row, col, 0]
                        ub = input_domain[chan, row, col, 1]
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        row_vars.append(v)
                        row_lbs.append(lb.item())
                        row_ubs.append(ub.item())
                    chan_vars.append(row_vars)
                    chan_lbs.append(row_lbs)
                    chan_ubs.append(row_ubs)
                inp_gurobi_vars.append(chan_vars)
                inp_lbs.append(chan_lbs)
                inp_ubs.append(chan_ubs)
        self.model.update()

        return inp_lbs, inp_ubs, inp_gurobi_vars


def optimize_model(model, time_budget):
    model.update()
    model.reset()
    opt_val = None
    attempt = 0
    if time_budget is not None:
        start = time.time()
        end = start + time_budget
        stop_callback, interm_bound = stop_before(end)
        while attempt <= 1:
            model.optimize(stop_callback)
            if model.status == grb.GRB.OPTIMAL:
                # Model was optimally solved
                opt_val = model.objVal
                break
            elif model.status == grb.GRB.INTERRUPTED:
                opt_val = interm_bound.value
                break
            elif model.status == grb.GRB.INF_OR_UNBD:
                assert attempt == 0
                model.setParam('DualReductions', 0)
                attempt += 1
                continue
            else:
                raise Exception(f"Unexpected Status code: {model.status}")
    else:
        while attempt <= 1:
            model.optimize()
            if model.status == grb.GRB.OPTIMAL:
                opt_val = model.objVal
                break
            elif model.status == grb.GRB.INF_OR_UNBD:
                assert attempt == 0
                model.setParam('DualReductions', 0)
                attempt += 1
                continue
            else:
                raise Exception(f"Unexpected Status code: {model.status}")
    if attempt == 1:
        model.setParam('DualReductions', 1)
    assert opt_val is not None
    return opt_val


class BestOptResult:
    def __init__(self):
        self.value = None


def stop_before(end_time, all_print=False):
    opt_bound = BestOptResult

    def timelimit_callback(model, where):
        if where == grb.GRB.Callback.SIMPLEX:
            new_time = time.time()
            obj = model.cbGet(grb.GRB.Callback.SPX_OBJVAL)
            # should always be >0 because gurobi maintain a dual feasible solution,
            # if it's not, this means we're just getting placeholders.
            # Given that the algo operates on the dual, if we have pinf=0, that means
            # the point is primal feasible and dual feasible. Can only be the optimum
            pinf = model.cbGet(grb.GRB.Callback.SPX_PRIMINF)
            # dinf = model.cbGet(grb.GRB.Callback.SPX_DUALINF) # Should always be 0, dual is feasible
            # print(f"It:{iter_count}\tpinf:{pinf}\tdinf{dinf}\tobj:{obj}")
            opt_bound.value = obj

            if (pinf > 0) and (new_time > end_time):
                model.terminate()
        # if where == grb.GRB.Callback.BARRIER:
        #     new_time = time.time()

        #     iter_count = model.cbGet(grb.GRB.Callback.BARRIER_ITRCNT)
        #     prim_obj = model.cbGet(grb.GRB.Callback.BARRIER_PRIMOBJ)
        #     dual_obj = model.cbGet(grb.GRB.Callback.BARRIER_DUALOBJ)
        #     prim_inf = model.cbGet(grb.GRB.Callback.BARRIER_PRIMINF)
        #     dual_inf = model.cbGet(grb.GRB.Callback.BARRIER_DUALINF)

        #     print(f"IT:{iter_count}\tprim_obj:{prim_obj}\tprim_inf:{prim_inf}\t"
        #           + f"dual_obj:{dual_obj}\tdual_inf:{dual_inf}")
        #     if new_time > end_budget:
        #         model.terminate()

    return timelimit_callback, opt_bound

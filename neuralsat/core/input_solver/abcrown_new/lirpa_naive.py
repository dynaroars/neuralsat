from collections import defaultdict
import gurobipy as grb
import torch.nn as nn
import numpy as np
import torch
import copy
import time


from .auto_LiRPA import BoundedModule, BoundedTensor
from .auto_LiRPA.perturbations import *
from .auto_LiRPA.utils import *


def copy_model(model):
    """
    deep copy a gurobi model together with variable historical results
    """
    model_split = model.copy()
    model_split.update()
    return model_split


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


class LiRPANaive:

    def __init__(self, model_ori, input_shape, c=None, rhs=None, device='cpu', conv_mode='patches'):

        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.c = c
        self.rhs = rhs
        self.model_ori = model_ori
        self.layers = layers
        self.input_shape = input_shape
        self.device = device
        
        self.net = BoundedModule(
            net, torch.zeros(input_shape, device=self.device),
            bound_opts={
                'relu': 'adaptive',
                'deterministic': False,
                'conv_mode': conv_mode,
                'sparse_features_alpha': True,
                'sparse_spec_alpha': True,
                'crown_batch_size': 1e9,
                'max_crown_size': 1e9,
                'forward_refinement': False,
                'dynamic_forward': False,
                'forward_max_dim': 1e9,
                'use_full_conv_alpha': True,
            },
            device=self.device
        )
        self.net.eval()

        # lp solver
        self.net.model = None

        # check conversion correctness
        dummy = torch.randn(input_shape, device=self.device)
        try:
            assert torch.allclose(net(dummy), self.net(dummy))
        except AssertionError:
            print(f'torch allclose failed: norm {torch.norm(net(dummy) - self.net(dummy))}')


    @torch.no_grad()
    def __call__(self, dm_l, dm_u, stop_criterion_func=None, reset_param=True):
        batch = len(dm_l)
        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=dm_l, x_U=dm_u)
        new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
        C = self.c.repeat(batch, *[1] * len(self.c.shape[1:]))

        with torch.no_grad():
            lb, _ = self.net.compute_bounds(x=(new_x,), C=C, method="backward", bound_upper=False)
            lb = lb.cpu()
        ub = [None] * (batch)

        return (lb, ub), None



    def build_solver_model(self, timeout):
        """
        m is the instance of LiRPAConvNet
        model_type ["mip", "lp", "lp_integer"]: three different types of guorbi solvers
        lp_integer using mip formulation but continuous integer variable from 0 to 1 instead of
        binary variable as mip; lp_integer should have the same results as lp but allowing us to
        estimate integer variables.
        NOTE: we build lp/mip solver from computer graph
        """

        self.net.model = grb.Model()
        self.net.model.setParam('OutputFlag', False)
        self.net.model.setParam("FeasibilityTol", 1e-7)
        # m.net.model.setParam('TimeLimit', timeout)

        # build model in auto_LiRPA
        out_vars = self.net.build_solver_module(C=self.c, final_node_name=self.net.final_name, model_type='lp')
        self.net.model.update()
        return out_vars



    def lp_solve_all_node_split(self, lower_bounds, upper_bounds, rhs):
        all_node_model = copy_model(self.net.model)
        pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in self.net.relus]
        relu_layer_names = [relu_layer.name for relu_layer in self.net.relus]

        for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
            lbs, ubs = lower_bounds[relu_idx].reshape(-1), upper_bounds[relu_idx].reshape(-1)
            for neuron_idx in range(lbs.shape[0]):
                pre_var = all_node_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
                pre_var.lb = pre_lb = lbs[neuron_idx]
                pre_var.ub = pre_ub = ubs[neuron_idx]
                var = all_node_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
                # var is None if originally stable
                if var is not None:
                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        var.lb = pre_lb
                        var.ub = pre_ub
                        all_node_model.addConstr(pre_var == var)
                    elif pre_lb <= 0 and pre_ub <= 0:
                        var.lb = 0
                        var.ub = 0
                    else:
                        continue
                        raise ValueError(f'Exists unstable neuron at index [{relu_idx}][{neuron_idx}]: lb={pre_lb} ub={pre_ub}')

        all_node_model.update()
        
        feasible = True
        adv = [1, 2, 3]
        
        orig_out_vars = self.net.final_node().solver_vars
        assert len(orig_out_vars) == len(rhs), f"out shape not matching! {len(orig_out_vars)} {len(rhs)}"
        for out_idx in range(len(orig_out_vars)):
            objVar = all_node_model.getVarByName(orig_out_vars[out_idx].VarName)
            decision_threshold = rhs[out_idx]
            all_node_model.setObjective(objVar, grb.GRB.MINIMIZE)
            all_node_model.update()
            all_node_model.optimize()

            if all_node_model.status == 2:
                glb = objVar.X
            elif all_node_model.status == 3:
                print("gurobi all node split lp model infeasible!")
                glb = float('inf')
            else:
                print(f"Warning: model status {m.all_node_model.status}!")
                glb = float('inf')

            if glb > decision_threshold:
                feasible = False
                adv = None
                break
                
            if all_node_model.status == 2:
                input_vars = [all_node_model.getVarByName(var.VarName) for var in self.net.input_vars]
                adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)

        del all_node_model
        # print(lp_status, glb)
        return feasible, adv

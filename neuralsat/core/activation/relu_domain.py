import torch
import copy
import gurobipy as grb
import numpy as np
import contextlib
import multiprocessing
import torch.nn as nn
import contextlib
import random
import time
import re
import os
from collections import defaultdict

class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision assigned to ReLUs.
    Comparison between instances is based on the values of the lower bound estimated for the instances.
    """

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, slope=None, beta=None, 
                 split_history=None, history=None, intermediate_betas=None, primals=None, assignment_mapping=None):

        self.lA = lA
        self.lower_bound = lb # output lower bound
        self.upper_bound = ub # output upper bound
        self.lower_all = lb_all
        self.upper_all = up_all
        self.history = [] if history is None else history
        self.split_history = [] if split_history is None else split_history
        self.intermediate_betas = intermediate_betas
        self.slope = slope
        self.beta = beta
        self.primals = primals
        self.assignment_mapping = assignment_mapping
        self.valid = True
        self.unsat = False
        self.next_decision = None

        # self.left = None
        # self.right = None
        # self.parent = None
        # self.split = False
        # self.depth = depth
        # self.gnn_decision = gnn_decision
        # primals {"p": primal values for input, pre_relu, and obj output primals, 
        #   "z": integer values for each relu layer}
        # z: stable relus have -1, others all unstable neuron from 0 to 1
        # self.priority = priority  # Higher priority will be more likely to be selected.

    def get_assignment(self):
        assignment = {}
        for lid, (lnodes, lsigns) in enumerate(self.history):
            assignment.update({self.assignment_mapping[(lid, lnodes[i])]: lsigns[i] > 0 for i in range(len(lnodes))})
        return assignment

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.lA = [lA.to(device='cpu', non_blocking=True) for lA in self.lA]
        self.lower_all = [lbs.to(device='cpu', non_blocking=True) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device='cpu', non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)

        if self.split_history:
            if "beta" in self.split_history:
                for lidx in range(len(self.split_history["beta"])):
                    if self.split_history["single_beta"][lidx] is not None:
                        self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                        self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device='cpu', non_blocking=True)
                        self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device='cpu', non_blocking=True)
                    if self.split_history["beta"][lidx] is not None:
                        self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device='cpu', non_blocking=True)
                        self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device='cpu', non_blocking=True)
                        self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                        self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device='cpu', non_blocking=True)
                    if self.split_history["bias"][lidx] is not None:
                        self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device='cpu', non_blocking=True)
            if "general_beta" in self.split_history:
                self.split_history["general_beta"] = self.split_history["general_beta"].to(device="cpu", non_blocking=True)

        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device='cpu', non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device='cpu', non_blocking=True)

        if self.beta is not None:
            self.beta = [b.to(device='cpu', non_blocking=True) for b in self.beta]
        
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device='cpu', non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device='cpu', non_blocking=True)
        return self

    def to_device(self, device, partial=False):
        if not partial:
            self.lA = [lA.to(device, non_blocking=True) for lA in self.lA]
            self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
            self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].to(device, non_blocking=True, dtype=torch.get_default_dtype())
        if self.split_history:
            if "beta" in self.split_history:
                for lidx in range(len(self.split_history["beta"])):
                    if self.split_history["single_beta"][lidx] is not None:
                        self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device=device, non_blocking=True)
                        self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device=device, non_blocking=True)
                        self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device=device, non_blocking=True)
                    if self.split_history["beta"][lidx] is not None:
                        self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device=device, non_blocking=True)
                        self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device=device, non_blocking=True)
                        self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device=device, non_blocking=True)
                        self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device=device, non_blocking=True)
                    if self.split_history["bias"][lidx] is not None:
                        self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device=device, non_blocking=True)
            if "general_beta" in self.split_history:
                self.split_history["general_beta"] = self.split_history["general_beta"].to(device=device, non_blocking=True)
        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device, non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device, non_blocking=True)
        if self.beta is not None:
            self.beta = [b.to(device, non_blocking=True) for b in self.beta]
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device, non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device, non_blocking=True)
        return self



def add_domain_parallel(lA, lb, ub, lb_all, up_all, selected_domains, slope, beta,
                        split_history=None, branching_decision=None, decision_thresh=0,
                        intermediate_betas=None, primals=None, priorities=None):

    domain_list = []
    batch = len(selected_domains)
    for i in range(batch):
        # first half batch: active neurons
        new_history = copy.deepcopy(selected_domains[i].history)
        if branching_decision is not None:
            new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # first half batch: active neurons
            new_history[branching_decision[i][0]][1].append(+1.0)  # first half batch: active neurons

            # sanity check repeated split
            if branching_decision[i][1] in selected_domains[i].history[branching_decision[i][0]][0]:
                print('BUG!!! repeated split!')
                print(selected_domains[i].history)
                print(branching_decision[i])
                raise RuntimeError

        left = ReLUDomain(lA[i], lb[i], ub[i], lb_all[i], up_all[i], slope[i], beta[i],
                          split_history=split_history[i],
                          history=new_history,
                          intermediate_betas=intermediate_betas[i],
                          primals=primals[i] if primals is not None else None,
                          assignment_mapping=selected_domains[i].assignment_mapping)
        
        if left.lower_bound >= decision_thresh:
            left.valid = False
            left.unsat = True

        domain_list.append(left)
        
        # second half batch: inactive neurons
        new_history = copy.deepcopy(selected_domains[i].history)
        if branching_decision is not None:
            new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # second half batch: inactive neurons
            new_history[branching_decision[i][0]][1].append(-1.0)  # second half batch: inactive neurons

        right = ReLUDomain(lA[i+batch], lb[i+batch], ub[i+batch], lb_all[i+batch], up_all[i+batch], slope[i+batch],  beta[i+batch], 
                           split_history=split_history[i+batch],
                           history=new_history,
                           intermediate_betas=intermediate_betas[i + batch],
                           primals=primals[i + batch] if primals is not None else None,
                           assignment_mapping=selected_domains[i].assignment_mapping)

        if right.lower_bound >= decision_thresh:
            right.valid = False
            right.unsat = True
        domain_list.append(right)

    return domain_list

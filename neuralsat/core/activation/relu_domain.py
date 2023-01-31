from collections import defaultdict
import torch.nn as nn
import numpy as np
import contextlib
import random
import torch
import copy
import time
import re
import os


class ReLUDomain:

    """
    Object representing a domain where the domain is specified by decision assigned to ReLUs.
    Comparison between instances is based on the values of the lower bound estimated for the instances.
    """

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, 
                 slope=None, beta=None, split_history=None, history=None, intermediate_betas=None,
                 assignment_mapping=None, pre_history=None):

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
        self.assignment_mapping = assignment_mapping
        self.valid = True
        self.unsat = False
        self.next_decision = None

        self.full = sum([((l < 0) * (u > 0)).sum() for l, u in zip(lb_all[:-1], up_all[:-1])]) == 0 if lb_all is not None else False
        self.pre_history = [] if pre_history is None else pre_history # status of neurons are set initially


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
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device, non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device, non_blocking=True)
        if self.beta is not None:
            self.beta = [b.to(device, non_blocking=True) for b in self.beta]

        return self



def add_domain_parallel(lA, lb, ub, lb_all, up_all, selected_domains, slope, beta, split_history=None, 
                        branching_decision=None, decision_thresh=0, intermediate_betas=None):

    # change returned lAs in beta_solver, so that lAs need to be rearranged
    instance_lAs = [[] for _ in range(len(lb))]
    for item in lA:
        for i in range(len(instance_lAs)):
            instance_lAs[i].append(item[i:i+1])

    domain_list = []
    batch = len(selected_domains)
    decision_thresh_cpu = decision_thresh.to('cpu')

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

        left = ReLUDomain(instance_lAs[i], 
                          lb[i], 
                          ub[i], 
                          lb_all[i], 
                          up_all[i], 
                          slope[i], 
                          beta[i],
                          split_history=split_history[i],
                          history=new_history,
                          intermediate_betas=intermediate_betas[i],
                          assignment_mapping=selected_domains[i].assignment_mapping)
        
        if (left.lower_bound > decision_thresh_cpu).any():
            left.valid = False
            left.unsat = True
        domain_list.append(left)
        
        # second half batch: inactive neurons
        new_history = copy.deepcopy(selected_domains[i].history)
        if branching_decision is not None:
            new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # second half batch: inactive neurons
            new_history[branching_decision[i][0]][1].append(-1.0)  # second half batch: inactive neurons

        right = ReLUDomain(instance_lAs[i+batch], 
                           lb[i+batch], 
                           ub[i+batch], 
                           lb_all[i+batch], 
                           up_all[i+batch], 
                           slope[i+batch],  
                           beta[i+batch], 
                           split_history=split_history[i+batch],
                           history=new_history,
                           intermediate_betas=intermediate_betas[i + batch],
                           assignment_mapping=selected_domains[i].assignment_mapping)

        if (right.lower_bound > decision_thresh_cpu).any():
            right.valid = False
            right.unsat = True
        domain_list.append(right)

    return domain_list

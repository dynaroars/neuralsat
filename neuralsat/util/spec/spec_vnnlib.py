import gurobipy as grb
import numpy as np
import itertools
import random
import torch

import arguments

class SpecVNNLIB:

    def __init__(self, spec):

        self.bounds, self.mat = spec
        self.dtype = arguments.Config['dtype']
        self.device = arguments.Config['device']

        self.prop_mat = None
        self.prop_rhs = None
        self.true_labels = None
        self.target_labels = None


    def extract(self):
        # print('preprocess vnnlib spec')
        assert len(self.mat) == 1
        prop_mat, prop_rhs = self.mat[0]

        if self.prop_mat is None:
            self.prop_mat = torch.tensor(prop_mat, dtype=self.dtype, device=self.device).unsqueeze(0)
        
        if self.prop_rhs is None:
            self.prop_rhs = torch.tensor(prop_rhs, dtype=self.dtype, device=self.device).unsqueeze(0)


        if self.true_labels is None or self.target_labels is None:
            true_labels, target_labels = [], []
            for m in prop_mat:
                true_label = np.where(m == 1)[-1]
                if len(true_label) != 0:
                    assert len(true_label) == 1
                    true_labels.append(true_label[0])
                else:
                    true_labels.append(None)

                target_label = np.where(m == -1)[-1]
                if len(target_label) != 0:
                    assert len(target_label) == 1
                    target_labels.append(target_label[0])
                else:
                    target_labels.append(None)

            self.true_labels = np.array([true_labels])
            self.target_labels = np.array([target_labels])

        # print('c   ', self.prop_mat)
        # print('rhs ', self.prop_rhs)
        # print('y   ', self.true_labels)
        # print('pidx', self.target_labels)
        # exit()

        return self.prop_mat, self.prop_rhs, self.true_labels, self.target_labels

    def get_input_property(self):
        return {
            'lbs': [b[0] for b in self.bounds],
            'ubs': [b[1] for b in self.bounds]
        }

    def get_output_property(self, output):
        dnf =  []
        for lhs, rhs in self.mat:
            cnf = []
            for l, r in zip(lhs, rhs):
                cnf.append(sum(l * output) <= r)
            dnf.append(cnf)
        return dnf


    def check_output_reachability(self, lbs, ubs):
        dnf =  []
        # p = 0.0
        for idx, (lhs, rhs) in enumerate(self.mat):
            lhs = torch.tensor(lhs, dtype=self.dtype, device=lbs.device)
            rhs = torch.tensor(rhs, dtype=self.dtype, device=lbs.device)

            cnf = []
            for l, r in zip(lhs, rhs):
                val = sum((l > 0) * lbs) - sum((l < 0) * ubs)
                cnf.append(val <= r)
            dnf.append(all(cnf))

        return any(dnf), True

    def check_solution(self, output):
        for lhs, rhs in self.mat:
            lhs = torch.tensor(lhs, dtype=self.dtype, device=output.device)
            rhs = torch.tensor(rhs, dtype=self.dtype, device=output.device)
            vec = lhs @ output.squeeze(0)
            if torch.all(vec <= rhs):
                return True
        return False

    def get_output_reachability_constraints(self, lbs_expr, ubs_expr):
        dnf =  []
        for lhs, rhs in self.mat:
            cnf = []
            obj = []
            for l, r in zip(lhs, rhs):
                cnf.append(grb.quicksum((l > 0) * lbs_expr) - grb.quicksum((l < 0) * ubs_expr) <= r)
                obj.append(grb.quicksum((l > 0) * ubs_expr) - grb.quicksum((l < 0) * lbs_expr) - r)
            dnf.append((cnf, grb.quicksum(obj)))
        return dnf


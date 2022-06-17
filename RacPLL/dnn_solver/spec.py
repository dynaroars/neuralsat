import gurobipy as grb
import itertools
import random
import torch

import settings

class SpecificationVNNLIB:

    def __init__(self, spec):

        self.bounds, self.mat = spec

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


    # def register(self, lbs, ubs):
    #     self.cac =  []
    #     for lhs, rhs in self.mat:
    #         lhs = torch.tensor(lhs, dtype=settings.DTYPE)
    #         rhs = torch.tensor(rhs, dtype=settings.DTYPE)
    #         tmp = []
    #         for l, r in zip(lhs, rhs):
    #             tmp.append(sum((l > 0) * lbs) - sum((l < 0) * ubs))

    #         self.cac.append(rhs - torch.tensor(tmp))

    def check_output_reachability(self, lbs, ubs):
        dnf =  []
        # p = 0.0
        for idx, (lhs, rhs) in enumerate(self.mat):
            lhs = torch.tensor(lhs, dtype=settings.DTYPE, device=lbs.device)
            rhs = torch.tensor(rhs, dtype=settings.DTYPE, device=lbs.device)

            cnf = []
            # vals = []
            for l, r in zip(lhs, rhs):
                val = sum((l > 0) * lbs) - sum((l < 0) * ubs)
                # vals.append(r - val)
                cnf.append(val <= r)
            dnf.append(all(cnf))

            # vals = torch.tensor(vals)
            # p += torch.mean(vals / self.cac[idx])

        return any(dnf), True

    def check_solution(self, output):
        for lhs, rhs in self.mat:
            lhs = torch.tensor(lhs, dtype=settings.DTYPE, device=output.device)
            rhs = torch.tensor(rhs, dtype=settings.DTYPE, device=output.device)
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


    # def get_output_reachability_objectives(self, lbs_expr, ubs_expr):
    #     dnf =  []
    #     for lhs, rhs in self.mat:
    #         cnf = []
    #         obj = []
    #         for l, r in zip(lhs, rhs):
    #             cnf.append(sum((l > 0) * lbs_expr) - sum((l < 0) * ubs_expr) - r)
    #             obj.append(sum((l > 0) * ubs_expr) - sum((l < 0) * lbs_expr) - r)
    #         dnf.append((cnf, sum(obj)))
    #     return dnf

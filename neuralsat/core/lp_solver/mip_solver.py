import gurobipy as grb
import torch
import time

from core.input_solver.abcrown_new.lirpa_naive import LiRPANaive
from util.misc.logger import logger
import arguments


class MIPSolver:

    def __init__(self, net, spec):

        self.net = net
        self.spec = spec

        self.dtype = arguments.Config['dtype']
        self.device = arguments.Config['device']

        c, self.rhs, _, _ = self.spec.extract()

        self.model = LiRPANaive(model_ori=self.net.layers, 
                                input_shape=self.net.input_shape, 
                                device=self.device, 
                                c=c, 
                                rhs=self.rhs)

        self.refined_bounds = None

        # self.build_general()
        # exit()

    def build_solver_model(self, lower_bounds, upper_bounds, timeout=None):
        x_range = torch.tensor(self.spec.bounds, dtype=self.dtype, device=self.device)
        input_lb = x_range[:, 0].reshape(self.net.input_shape)
        input_ub = x_range[:, 1].reshape(self.net.input_shape)
        
        # forward to initialize variables
        self.model(input_lb, input_ub)

        # build Gurobi solver
        if self.refined_bounds is None:
            self.refined_bounds = self.model.build_solver_mip(x_range, lower_bounds, upper_bounds, timeout=timeout)

        return self.refined_bounds


    def build_general(self):
        x_range = torch.tensor(self.spec.bounds, dtype=self.dtype, device=self.device)
        input_lb = x_range[:, 0].reshape(self.net.input_shape)
        input_ub = x_range[:, 1].reshape(self.net.input_shape)
        
        # forward to initialize variables
        self.model(input_lb, input_ub)

        # build Gurobi solver
        self.model.build_solver_model(model_type='mip', timeout=100)
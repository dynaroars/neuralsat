import gurobipy as grb
import typing
import torch
import copy
import time
import math

from abstractor.abstractor import NetworkAbstractor
from util.misc.result import ReturnStatus

class MIPSolver:
    
    def __init__(self, net, dnf_objective, input_shape):
        self.device = 'cpu'
        self.net = net.to(self.device)
        self.input_shape = input_shape
        self.input_split = False
        
        assert dnf_objective.cs.shape[1] == 1 # c shape: [#props, 1, #outputs]
        self.objectives = copy.deepcopy(dnf_objective)
        
        self.input_lowers = self.objectives.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        self.input_uppers = self.objectives.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        self.cs = self.objectives.cs.to(self.device)
        self.rhs = self.objectives.rhs.to(self.device)
        
        self.initialize_abstractor('backward')
                

    def initialize_abstractor(self, method: str) -> None:
        if hasattr(self, 'abstractor'):
            # del self.abstractor.net
            del self.abstractor

        self.abstractor = NetworkAbstractor(
            pytorch_model=self.net, 
            input_shape=self.input_shape, 
            method=method,
            input_split=self.input_split,
            device=self.device,
        )
        
        self.abstractor.setup(self.objectives)
        self.abstractor.net.get_split_nodes(input_split=False)
        
    def verify(self, timeout):
        # print(input_lowers.shape)
        # print(cs.shape)
        
        tic = time.time()
        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=self.input_lowers, 
            input_upper=self.input_uppers, 
            c=self.cs,
            refine=False,
            timeout=None,
        )
        print(f'Initialize new MIP model in {time.time() - tic} seconds, {timeout=}')
        mip_model = self.abstractor.net.model
        print(mip_model)
        output_names = [v.VarName for v in self.abstractor.net.final_node().solver_vars]
        assert len(output_names) == len(self.cs)
        
        print(output_names)
        print(self.rhs)
        # for var_name in output_names:
        #     print(var_name)
        feasible = False
        adv = None
        for out_idx in range(len(output_names)):
            assert len(self.rhs[out_idx]) == 1
            objective_var = mip_model.getVarByName(output_names[out_idx])
            mip_model.setObjective(objective_var, grb.GRB.MINIMIZE)
            mip_model.update()
            mip_model.optimize()
            if mip_model.status == grb.GRB.OPTIMAL:
                output_lb = objective_var.X
                print(f'Optimal! {output_lb=}')
            else:
                print(f"Infeasible! Model status {mip_model.status=}")
                # output_lb = float('inf')
                return ReturnStatus.UNKNOWN, None
                
            if output_lb < self.rhs[out_idx][0]:
                return ReturnStatus.UNKNOWN, None
                # cannot verify
                feasible = True
                input_vars = [mip_model.getVarByName(f'inp_{dim}') for dim in range(math.prod(self.input_shape))]
                adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)
                print(adv)
                print(self.net(adv))
                raise
                return ReturnStatus.SAT, adv
            
        return ReturnStatus.UNSAT, None
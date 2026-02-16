import gurobipy as grb
import torch
import time
import math

from abstractor.abstractor import NetworkAbstractor
from helper.misc.check import check_solution
from helper.misc.result import ReturnStatus

class MIPSolver:
    
    def __init__(self, net, input_shape):
        self.device = 'cpu'
        self.net = net.to(self.device)
        self.input_shape = input_shape
        self.input_split = False
        self.initialize_abstractor('backward')
                

    def initialize_abstractor(self, method: str) -> None:
        if hasattr(self, 'abstractor'):
            del self.abstractor

        self.abstractor = NetworkAbstractor(
            pytorch_model=self.net, 
            input_shape=self.input_shape, 
            method=method,
            input_split=self.input_split,
            device=self.device,
        )
        
        self.abstractor.setup(None)
        self.abstractor.net.get_split_nodes()
        
    def verify(self, dnf_objective, timeout):
        if len(dnf_objective) > 10:
            return ReturnStatus.UNKNOWN, None
        start = time.time()

        saw_unknown = False
        while len(dnf_objective):
            if time.time() - start > timeout:
                return ReturnStatus.UNKNOWN, None

            objective = dnf_objective.pop(batch=1)   #batch size = 1, not index
            status, adv = self.verify_one(objective=objective, timeout=timeout)

            if status == ReturnStatus.SAT:
                return ReturnStatus.SAT, adv
            if status == ReturnStatus.UNKNOWN:
                saw_unknown = True
            #if UNSAT, keep checking other disjuncts

        if saw_unknown:
            return ReturnStatus.UNKNOWN, None
        
        return ReturnStatus.UNSAT, None
            
        
    def verify_one(self, objective, timeout):
        # print(input_lowers.shape)
        # print(self.cs.shape)
        assert len(objective.upper_bounds) == len(objective.lower_bounds) == 1
        input_upper = objective.upper_bounds.view(self.input_shape).to(self.device)
        input_lower = objective.lower_bounds.view(self.input_shape).to(self.device)
        cs = objective.cs.to(self.device)
        rhs = objective.rhs
        
        tic = time.time()
        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=input_lower, 
            input_upper=input_upper, 
            c=cs,
            refine=False,
            timeout=None,
        )
        print(f'Initialize new MIP model in {time.time() - tic} seconds, {timeout=}')
        mip_model = self.abstractor.net.solver_model
        mip_model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
        
        print(mip_model)
        output_names = [v.VarName for v in self.abstractor.net.final_node().solver_vars]
        assert len(output_names) == cs.shape[1] == rhs.shape[1], f'{len(output_names)=} {len(cs)=} {output_names=} {cs.shape=} {rhs.shape=}'
        
        for out_idx in range(len(output_names)):
            objective_var = mip_model.getVarByName(output_names[out_idx])
            mip_model.addConstr(objective_var <= rhs[0][out_idx].item())
            
        mip_model.update()
        mip_model.optimize()
            
        if mip_model.status == grb.GRB.INFEASIBLE:
            return ReturnStatus.UNSAT, None
        
        input_vars = [mip_model.getVarByName(f'inp_{dim}') for dim in range(math.prod(self.input_shape))]
        adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)
        
        if check_solution(self.net, adv, cs, rhs, input_lower, input_upper):
            return ReturnStatus.SAT, adv
        return ReturnStatus.UNKNOWN, None
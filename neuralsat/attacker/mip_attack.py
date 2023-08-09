import gurobipy as grb
import numpy as np
import random
import torch
import time

from util.misc.check import check_solution

multiprocess_mip_attack_model = None
multiprocess_stop = False

class CounterExamplePool:
    
    def __init__(self, net, objective):
        self.net = net
        self.objective = objective
        
        self._initialize()


    def _initialize(self, n_samples=10):
        pass
        
        
    def add(self, inputs):
        assert len(inputs) > 0
        pass
        

def mip_solver_worker(candidate, n_inputs):
    global multiprocess_stop
    if multiprocess_stop:
        return None
    
    # print('[!] Attacking:', candidate)
    tmp_model = multiprocess_mip_attack_model.copy()
    
    v = tmp_model.getVarByName(candidate)
    vlb = out_lb = v.lb
    vub = out_ub = v.ub
    adv = None
    
    tmp_model.setObjective(v, grb.GRB.MINIMIZE)
    tmp_model.update()
    try:
        tmp_model.optimize()
    except grb.GurobiError as e:
        print(f'Gurobi error: {e.message}')
        return None

    vlb = max(tmp_model.objbound, out_lb)
    if tmp_model.solcount > 0:
        vub = min(tmp_model.objval, out_ub)
    if vub < 0:
        input_vars = [tmp_model.getVarByName(f'inp_{dim}') for dim in range(n_inputs)]
        adv = [var.X for var in input_vars]
        multiprocess_stop = True
    return adv    
    
    
class MIPAttacker:

    def __init__(self, net, objectives, mip_model, output_names, input_shape, device):
        self.net = net
        self.objectives = objectives
        self.input_shape = input_shape
        self.device = device
        self.output_names = output_names

        assert self.objectives.cs.shape[1] == 1 # c shape: [#labels, 1, #outputs]
        
        self.mip_model = mip_model.copy()
        self.mip_model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
        self.mip_model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
        
        
    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        
    def run(self, reference_bounds=None, timeout=1.0):
        global multiprocess_mip_attack_model, multiprocess_stop
        multiprocess_mip_attack_model = self.mip_model.copy()
        multiprocess_stop = False
        
        for var_name in self.output_names:
            adv = mip_solver_worker(var_name, np.prod(self.input_shape))
            if adv is not None:
                adv = torch.tensor(adv, device=self.device).view(self.input_shape)
                if check_solution(net=self.net, adv=adv, cs=self.objectives.cs, rhs=self.objectives.rhs, data_min=self.objectives.lower_bounds, data_max=self.objectives.upper_bounds):
                    return True, adv
                
        return False, None
    
    
    def attack_domains(self, domain_params):
        # exit()
        # print(self.run())
        if not hasattr(self, 'cex_pool'):
            self.cex_pool = CounterExamplePool(self.net, self.objectives)
            
            
        # print('attack_domains', len(domain_params.lower_bounds[0]), self.objectives.cs.shape)
        # for h in domain_params.histories:
        #     print(h)
        # exit()
        
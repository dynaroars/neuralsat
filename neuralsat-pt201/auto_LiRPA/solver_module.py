import multiprocessing
import os

from .beta_crown import SparseBeta
from .bound_ops import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

MULTIPROCESS_MODEL = None
N_REFINE_LAYER = 10
EAGER_OPTIMIZE = False
N_PROC = os.cpu_count() // 2

DEBUG = True

def build_solver_module(self: 'BoundedModule', x=None, C=None, interm_bounds=None,
                        final_node_name=None, model_type="mip", solver_pkg="gurobi",
                        timeout=None, timeout_per_neuron=None, refine=False):
    r"""build lp/mip solvers in general graph.

    Args:
        x: inputs, a list of BoundedTensor. If set to None, we reuse exisint bounds that
        were previously computed in compute_bounds().
        C (Tensor): The specification matrix that can map the output of the model with an
        additional linear layer. This is usually used for maping the logits output of the
        model to classification margins.
        interm_bounds: if specified, will replace existing intermediate layer bounds.
        Otherwise we reuse exising intermediate bounds.

        final_node_name (String): the name for the target layer to optimize

        solver_pkg (String): the backbone of the solver, default gurobi, also support scipy

    Returns:
        output vars (list): a list of final nodes to optimize
    """
    # self.root_names: list of root node name
    # self.final_name: list of output node name
    # self.final_node: output module
    # <module>.input: a list of input modules of this layer module
    # <module>.solver_vars: a list of gurobi vars of every layer module
    #       list with conv shape if conv layers, otherwise flattened
    # if last layer we need to be careful with:
    #       C: specification matrix
    #       <module>.is_input_perturbed(1)
    if (x is not None) and (interm_bounds is not None):
        # Set the model to use new intermediate layer bounds, ignore the original ones.
        self.set_input(x, interm_bounds=interm_bounds)

    roots = [self[name] for name in self.root_names]

    # create interval ranges for input and other weight parameters
    for i in range(len(roots)):
        value = roots[i].forward()
        # if isinstance(root[i], BoundInput) and not isinstance(root[i], BoundParams):
        if type(roots[i]) is BoundInput:
            # create input vars for gurobi self.model
            inp_gurobi_vars = self._build_solver_input(roots[i])
        else:
            # regular weights
            roots[i].solver_vars = value

    final = self.final_node() if final_node_name is None else self[final_node_name]

    # backward propagate every layer including last layer
    if model_type == 'lp':
        self._build_solver_general(node=final, C=C, model_type=model_type, solver_pkg=solver_pkg)
    else:
        if refine:
            assert all([isinstance(_, BoundRelu) for _ in self.perturbed_optimizable_activations]), print('Error: Support ReLU only')
        self._build_solver_refined(x=x, node=final, C=C, model_type=model_type, solver_pkg=solver_pkg, refine=refine, timeout_per_neuron=timeout_per_neuron)

    # a list of output solver vars
    return final.solver_vars


def _build_solver_general(self: 'BoundedModule', node: Bound, C=None, model_type="mip", solver_pkg="gurobi"):
    if not hasattr(node, 'solver_vars'):
        for n in node.inputs:
            self._build_solver_general(n, C=C, model_type=model_type, solver_pkg=solver_pkg)
        inp = [n_pre.solver_vars for n_pre in node.inputs]
        # print(node, node.inputs)
        if C is not None and isinstance(node, BoundLinear) and\
                not node.is_input_perturbed(1) and self.final_name == node.name:
            # when node is the last layer
            # merge the last BoundLinear node with the specification,
            # available when weights of this layer are not perturbed
            solver_vars = node.build_solver(*inp, model=self.model, C=C,
                model_type=model_type, solver_pkg=solver_pkg)
        else:
            solver_vars = node.build_solver(*inp, model=self.model, C=None,
                    model_type=model_type, solver_pkg=solver_pkg)
        # just return output node gurobi vars
        return solver_vars


def _build_solver_refined(self, x, node, C=None, model_type="mip", solver_pkg="gurobi", refine=False, timeout_per_neuron=None):
    if not hasattr(node, 'solver_vars'):
        for n in node.inputs:
            self._build_solver_refined(x=x, node=n, C=C, model_type=model_type, solver_pkg=solver_pkg, refine=refine, timeout_per_neuron=timeout_per_neuron)
        
        # refine from second relu layer
        if refine and isinstance(node, BoundRelu) and (node in self.relus[1:1+N_REFINE_LAYER]):
            global MULTIPROCESS_MODEL
            
            refine_node = node.inputs[0]
            assert len(refine_node.lower) == 1
            
            if DEBUG:
                print('[+] Refine layer:', refine_node.name)
                # print(refine_node.solver_vars)
                
            if isinstance(refine_node, (BoundConv, BoundConvTranspose)):
                # TODO: refine for Conv layers
                if DEBUG:
                    print('\t- Skip:', refine_node)
            else:
                candidates = []
                # candidate_neuron_ids = []
                unstable_to_stable = []
                    
                for neuron_idx in range(refine_node.lower.numel()):
                    v = self.model.getVarByName(f'lay{refine_node.name}_{neuron_idx}')
                    if v.ub * v.lb < 0: # unstable neuron
                        candidates.append((neuron_idx, v.VarName, timeout_per_neuron))
                        
                    # TODO: uncomment for speeding up
                    v.lb, v.ub = -np.inf, np.inf
                    
                self.model.update()
                
                # tighten bounds
                if len(candidates):
                    if DEBUG:
                        print('#candidates =', len(candidates))
                        
                    MULTIPROCESS_MODEL = self.model.copy()
                    if (N_PROC > 1) and (len(candidates) > 1):
                        with multiprocessing.Pool(min(N_PROC, len(candidates))) as pool:
                            solver_result = pool.map(mip_solver_worker, candidates, chunksize=1)
                    else:
                        solver_result = []
                        for can in candidates:
                            solver_result.append(mip_solver_worker(can))
                    MULTIPROCESS_MODEL = None
                    
                    # update bounds
                    refine_node_lower = refine_node.lower.clone().detach().cpu().flatten(1)
                    refine_node_upper = refine_node.upper.clone().detach().cpu().flatten(1)
                    for neuron_idx, vlb, vub, refined in solver_result:
                        if refined:
                            if vlb >= 0:
                                unstable_to_stable.append((neuron_idx, 1))
                            if vub <= 0:
                                unstable_to_stable.append((neuron_idx, -1))
                            refine_node_lower[0, neuron_idx] = max(vlb, refine_node_lower[0, neuron_idx])
                            refine_node_upper[0, neuron_idx] = min(vub, refine_node_upper[0, neuron_idx])
                        
                # restore bounds
                for neuron_idx in range(refine_node.lower.numel()):
                    v = self.model.getVarByName(f'lay{refine_node.name}_{neuron_idx}')
                    v.lb = refine_node_lower[0, neuron_idx]
                    v.ub = refine_node_upper[0, neuron_idx]
                
                shape = refine_node.lower.shape
                refine_node.lower = refine_node_lower.view(shape).to(self.device)
                refine_node.upper = refine_node_upper.view(shape).to(self.device)
                self.model.update()

                # compute bounds
                reference_bounds = self.get_refined_interm_bounds()
                # if 1:
                #     self.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': False}})
                #     lb_, _ = self.compute_bounds(x=x, C=C, method="CROWN-optimized", reference_bounds=reference_bounds)
                #     print('optimized  (w/o beta):', lb_)
                
                # initialize all betas
                if node == self.relus[1]:
                    for relu_layer in self.relus:
                        relu_layer.sparse_betas = [SparseBeta((1, 0), device=self.device)]

                # add beta constraints
                max_splits_per_layer = len(unstable_to_stable)
                node.sparse_betas = [SparseBeta((1, max_splits_per_layer), device=self.device)]
                
                # assign split constraints
                for neuron_idx, (refined_neuron, sign) in enumerate(unstable_to_stable):
                    node.sparse_betas[0].loc[0, neuron_idx] = refined_neuron
                    node.sparse_betas[0].sign[0, neuron_idx] = sign
                
                self.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': True}})
                lb_, _ = self.compute_bounds(x=x, C=C, method="crown-optimized", reference_bounds=reference_bounds)
                self.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': False}})
                if DEBUG:
                    print('optimized  (w/  beta):', lb_)
                    
                # TODO: check spec here?
        else:
            # TODO: refine for general activation
            pass
        
        
        inp = [n_pre.solver_vars for n_pre in node.inputs]
        if C is not None and isinstance(node, BoundLinear) and \
                not node.is_input_perturbed(1) and self.final_name == node.name:
            # when node is the last layer, merge node with the specification, 
            # available when weights of this layer are not perturbed
            solver_vars = node.build_solver(*inp, model=self.model, C=C, 
                    model_type=model_type, solver_pkg=solver_pkg)
        else:
            solver_vars = node.build_solver(*inp, model=self.model, C=None, 
                    model_type=model_type, solver_pkg=solver_pkg)
            
        return solver_vars
    



def _reset_solver_vars(self: 'BoundedModule', node: Bound):
    if hasattr(node, 'solver_vars'):
        del node.solver_vars
    for n in node.inputs:
        self._reset_solver_vars(n)


def _build_solver_input(self: 'BoundedModule', node):
    ## Do the input layer, which is a special case
    assert isinstance(node, BoundInput)
    assert node.perturbation is not None
    assert node.perturbation.norm == float("inf")
    inp_gurobi_vars = []
    # zero var will be shared within the solver model
    zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    one_var = self.model.addVar(lb=1, ub=1, obj=0, vtype=grb.GRB.CONTINUOUS, name='one')
    neg_one_var = self.model.addVar(lb=-1, ub=-1, obj=0, vtype=grb.GRB.CONTINUOUS, name='neg_one')
    x_L = node.value - node.perturbation.eps if node.perturbation.x_L is None else node.perturbation.x_L
    x_U = node.value + node.perturbation.eps if node.perturbation.x_U is None else node.perturbation.x_U
    x_L = x_L.squeeze(0)
    x_U = x_U.squeeze(0)
    # x_L, x_U = node.lower.squeeze(0), node.upper.squeeze(0)

    if x_L.ndim not in [1, 3]:
        x_L = x_L.squeeze()
        x_U = x_U.squeeze()
    
    # handle case (batch, 1, h, w) -- mnist
    if (x_L.ndim == 3) and (x_L.size(0) == 1):
        x_L = x_L.flatten()
        x_U = x_U.flatten()
        
    if x_L.ndim == 1:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(zip(x_L, x_U)):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        assert x_L.ndim == 3, f"x_L ndim  {x_L.ndim}"
        dim = 0
        for chan in range(x_L.shape[0]):
            chan_vars = []
            for row in range(x_L.shape[1]):
                row_vars = []
                for col in range(x_L.shape[2]):
                    lb = x_L[chan, row, col]
                    ub = x_U[chan, row, col]
                    v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'inp_{dim}')
                                            # name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                    dim += 1
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)

    node.solver_vars = inp_gurobi_vars
    # save the gurobi input variables so that we can later extract primal values in input space easily
    self.input_vars = inp_gurobi_vars
    self.model.update()
    return inp_gurobi_vars



def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise
    
    
def mip_solver_worker(candidate):
    """ Multiprocess worker for solving MIP models in build_the_model_mip_refine """

    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9: # Timed out. Get current bound.
            bound = bound_type(grb_model.objbound, reference)
            refined = bound != reference
        elif grb_model.status == 2: # Optimally solved.
            bound = grb_model.objbound
            refined = True
        elif grb_model.status == 15: # Found an lower bound >= 0 or upper bound <= 0, so this neuron becomes stable.
            bound = bound_type(1., -1.) * eps
            refined = True
        else:
            bound = reference
        return bound, refined, grb_model.status

    def solve_ub(model, v, out_ub, eps=1e-5):
        status_ub_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min)
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5):
        status_lb_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max)
        return vlb, refined, status_lb, status_lb_r

    init_lb, init_ub = None, None
    eps = 1e-5

    neuron_idx, var_name, timeout_per_neuron = candidate
    # print(f'Refining neuron={var_name}, timeout={timeout_per_neuron}')
    
    model = MULTIPROCESS_MODEL.copy()
    model.setParam('Threads', 1)
    if timeout_per_neuron is not None:
        model.setParam('TimeLimit', timeout_per_neuron)
    
    v = model.getVarByName(var_name)
    out_lb, out_ub = v.lb, v.ub
    refine_time = time.time()
    neuron_refined = False

    if abs(out_lb) < abs(out_ub): # lb is tighter, solve lb first.
        vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
        neuron_refined = neuron_refined or refined
        if vlb <= 0 or EAGER_OPTIMIZE: # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # lb > 0, neuron is stable, we skip solving ub.
            vub, status_ub, status_ub_r = out_ub, -1, -1
    else: # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
        neuron_refined = neuron_refined or refined
        if vub >= 0 or EAGER_OPTIMIZE: # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # ub < 0, neuron is stable, we skip solving ub.
            vlb, status_lb, status_lb_r = out_lb, -1, -1

    if DEBUG:
        # print(model)
        print(f"Solving MIP (#workers={N_PROC}, timeout={timeout_per_neuron}) for {v.VarName:<10}: [{out_lb:.6f}, {out_ub:.6f}]=>[{vlb:.6f}, {vub:.6f}] ({status_lb}, {status_ub}), time: {time.time()-refine_time:.4f}s, #vars: {model.NumVars}, #constrs: {model.NumConstrs}")
        sys.stdout.flush()

    return neuron_idx, vlb, vub, neuron_refined


from plnn.network_linear_approximation import LinearizedNetwork
from tools.custom_torch_modules import View, Flatten
from plnn.proxlp_solver.utils import prod, OptimizationTrace
from plnn.branch_and_bound.utils import ParentInit
from plnn.proxlp_solver.solver import SaddleLP

from torch import nn
import torch
from itertools import chain, combinations  # tools to generate power set
import math

"""
    References:
    [1] https://arxiv.org/pdf/1811.01988.pdf
    [2] https://www.gurobi.com/documentation/8.1/refman/py_model_cbcut.html
    [3] https://www.gurobi.com/documentation/8.1/refman/callback_codes.html
"""


class AndersonLinearizedNetwork(LinearizedNetwork):
    """
    Define class providing, through the define_linear_approximation method, a linear approximation of a neural network using
    Anderson et al.'s [1] convex hull of the composition of a linear function and a ReLu.
    It inherits from LinearizedNetwork utilities.
    """

    def __init__(self, layers, mode="lp-cut", n_cuts=100, cuts_per_neuron=False, store_bounds_progress=-1,
                 decision_boundary=0, neurons_per_layer=-1):
        """
        :param layers: A list of Pytorch layers containing only Linear/Conv2d/ReLU/
        :param mode: the relaxation can be implemented in four ways. Solve the MIP to completion: "mip-exact".
        Solve the relaxation by manually inserting cuts: "lp-cut". Insert all the exponentially many constraints:
        "lp-all". NOTE: it does not seem doable to use Gurobi's cut filtering and solve a relaxation.
        All cutting planes come from the exponential family of [1].
        :param n_cuts: in case one includes cutting planes and the mip is not solved to optimality, how many cutting
        planes passes (one pass adds a violated plane to each ReLU) Gurobi should add before termination
            (for each neuron optimisation if cuts_per_neuron).
        :param cuts_per_neuron: whether the cuts upper bound is to be interpreted per neuron of the network
            (used only with "lp-cut").
        :param decision_boundary: decision_boundary for BaB (used only with counter_example_verification)
        bounds. These are computed anyways, so no added computation.
        :param cuts_per_neuron: how many neurons per layer to use Anderson constraints on. -1 means all
        :param store_bounds_progress: which layer to store bounds progress for. -1=don't.
        """
        super().__init__(layers)
        self.mode = mode
        self.cuts_per_neuron = cuts_per_neuron
        self.check_input_format()
        self.check_options()

        self.last_relu_index = 0  # Store the last ReLU index we got to when creating the model.
        self.cut_treshold = n_cuts
        self.applied_cuts = 0  # Keeps track of how many new cuts passes have been performed
        self.old_applied_cuts = -1
        self.current_bound_value = None  # Horrible hack to retrieve relaxed bound value from Gurobi callback.
        self.pending_bound_var = None  # Horrible hack to retrieve relaxed bound value from Gurobi callback.
        self.cut_optimization_calls = 0
        self.lower_bounds = []
        self.pre_lower_bounds = []
        self.pre_upper_bounds = []
        self.upper_bounds = []
        self.gurobi_x_vars = []
        self.gurobi_z_vars = []
        self.ambiguous_relus = []  # Keep track of which ReLUs are ambiguous.
        self.neurons_per_layer = neurons_per_layer

        # Keep track of which Gurobi constraints have been added from the exponential family [only for "lp-cut"]
        self.added_exp_constraints = []

        # Keep track of whether the Gurobi variables have already been constructed and of which "fixed" (that is, not
        # related to any cut-like computation) constraints are active.
        self.model_built = False
        self.active_anderson_constraints = []
        self.decision_boundary = decision_boundary

        self.store_bounds_progress = store_bounds_progress
        self.logger = OptimizationTrace()

        self.bounds_num_tolerance = 1e-4
        self.insert_cuts = True

        # dummy Parent Init: for Gurobi we're not using any
        self.children_init = ParentInit()

    def check_input_format(self):
        """
        Check that the input format is supported.
        """
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                continue

            if type(layer) not in [nn.Conv2d, nn.Linear, nn.ReLU, Flatten, View]:
                raise ValueError("{} does not support layer type {}".format(type(self), type(layer)))

            previous_layer = self.layers[layer_idx-1]
            if type(layer) is nn.ReLU:
                if type(previous_layer) not in [nn.Conv2d, nn.Linear]:
                    raise ValueError("{} expects Linear or Conv2d before ReLU",format(type(self)))
            if type(layer) in [nn.Linear, nn.Conv2d]:
                if type(previous_layer) in [nn.Linear, nn.Conv2d]:
                    raise ValueError("{} doesn't support consecutive linear layers. Please " +
                                     "merge them into one".format(type(self)))

            if (layer is self.layers[-1]) and type(layer) not in [nn.Linear, nn.Conv2d]:
                raise ValueError("Last layer is neither linear nor convolutional.")

            if type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)  # inherited from Rudy

    def check_options(self):
        """
        Check that the chosen options are supported.
        """
        if self.mode not in ["mip-exact", "lp-cut", "lp-all"]:
            raise ValueError("Available options for mode are: mip-exact, lp-cut, lp-all")

        if self.cuts_per_neuron and self.mode != "lp-cut":
            raise NotImplementedError("cuts_per_neuron=True works only with lp-cut, right now.")

    # Method potentially useful for branch and bound.
    def build_model_using_bounds(self, input_domain, intermediate_bounds, n_threads=1):
        """
        Build the model using intermediate_bounds, containing (lower_bounds, upper_bounds)
        These should be in the same format define_linear_approximation would compute them (list of tensors).
        preactivation_bounds contains the bounds before the ReLU clipping (important to build the model)
        :param input_domain: Tensor containing in each row the lower and upper bound for the corresponding input dimension
        :param intermediate_bounds: (lower_bounds, upper_bounds) each as list of tensors (of size equal to the number
        of ReLU layers, + 1, the input bounds). IMPORTANT: these describe post-activation bounds.
        """
        self.device = input_domain[0].device
        lbs, ubs = intermediate_bounds
        self.lower_bounds = []
        self.upper_bounds = []
        self.pre_lower_bounds = []
        self.pre_upper_bounds = []
        for x_idx in range(len(lbs)):
            lb = lbs[x_idx]
            ub = ubs[x_idx]
            if x_idx == 0:
                self.lower_bounds.append(lb.clone())
                self.upper_bounds.append(ub.clone())
            else:
                self.lower_bounds.append(torch.clamp(lb, 0, None))
                self.upper_bounds.append(torch.clamp(ub, 0, None))
                self.pre_lower_bounds.append(lb.clone())
                self.pre_upper_bounds.append(ub.clone())
        x_idx_max = len(lbs) - 1
        self.define_linear_approximation(input_domain, int_bounds_provided=True, compute_final_layer=False,
                                         x_idx_max=x_idx_max, n_threads=n_threads)
        self.model_built = True

    def compute_output_from_intermediate_bounds(self, input_domain, intermediate_bounds):
        """
        Build the model using intermediate bounds and compute the bounds for the last layer only.
        :param input_domain: Tensor containing in each row the lower and upper bound for the corresponding input dimension
        :param intermediate_bounds: (lower_bounds, upper_bounds) each as list of tensors (of size equal to the number
        of ReLU layers, + 1, the input bounds). IMPORTANT: these describe post-activation bounds.
        """
        lbs, ubs = intermediate_bounds
        self.lower_bounds = []
        self.upper_bounds = []
        self.pre_lower_bounds = []
        self.pre_upper_bounds = []
        for x_idx in range(len(lbs)):
            lb = lbs[x_idx]
            ub = ubs[x_idx]
            if x_idx == 0:
                # these places will be filled in define_linear_approximation
                self.lower_bounds.append(None)
                self.upper_bounds.append(None)
            else:
                self.lower_bounds.append(torch.clamp(lb, 0, None))
                self.upper_bounds.append(torch.clamp(ub, 0, None))
                self.pre_lower_bounds.append(lb.clone())
                self.pre_upper_bounds.append(ub.clone())
        self.define_linear_approximation(input_domain, int_bounds_provided=True, compute_final_layer=True)

    # Method potentially useful for branch and bound.
    def compute_lower_bound(self, node=(-1, None), upper_bound=False, ub_only=False, counterexample_verification=False):
        """
        Compute a lower bound of the function for the given node

        :param node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        :param upper_bound: (optional) Compute an upper bound instead of a lower bound
        :pram ub_only: (optional) Compute upper bounds only, meaningful only when node[1] = None
        """

        if not self.lower_bounds:
            raise ValueError("compute_lower_bound called without any bounds stored.")

        device = self.input_domain.device
        # this piece of code assumes that the batch dimension is not there
        self.lower_bounds = [lbs.to(device).squeeze(0) for lbs in self.lower_bounds]
        self.upper_bounds = [ubs.to(device).squeeze(0) for ubs in self.upper_bounds]

        # retrieve correct relu and layer indices
        x_idx = len(self.lower_bounds) + node[0] if node[0] < 0 else node[0]
        id_count = 1
        for lay_idx, layer in enumerate(self.layers):
            if id_count == x_idx and type(layer) is not Flatten:
                break
            if type(layer) is nn.ReLU:
                id_count += 1
        layer_idx = lay_idx

        # Retrieve lower, upper bounds, and variables of previous layer.
        l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, layer_idx)

        is_batch = (node[1] is None)
        if not is_batch:

            c_b = self.compute_pre_activation_bounds(
                x_idx, self.layers[layer_idx], x_km1_vars, ub_only=ub_only,
                counterexample_verification=counterexample_verification, single_neuron=(node[1], upper_bound))

            # the rest of the code assumes the batch dimension is there
            self.lower_bounds = [lbs.to(device).unsqueeze(0) for lbs in self.lower_bounds]
            self.upper_bounds = [ubs.to(device).unsqueeze(0) for ubs in self.upper_bounds]

            return torch.tensor(c_b, device=device).unsqueeze(0)
        else:
            # Batched last layer bound computation.

            # start of logging time for optimization
            if x_idx == self.store_bounds_progress:
                self.logger.start_timing()
            # compute the layer output and optimize over it.
            preact_lower, preact_upper, _ = self.compute_pre_activation_bounds(
                x_idx, self.layers[layer_idx], x_km1_vars, ub_only=ub_only,
                counterexample_verification=counterexample_verification)

            # the rest of the code assumes the batch dimension is there
            self.lower_bounds = [lbs.to(device).unsqueeze(0) for lbs in self.lower_bounds]
            self.upper_bounds = [ubs.to(device).unsqueeze(0) for ubs in self.upper_bounds]

            return preact_lower.unsqueeze(0), preact_upper.unsqueeze(0)

    def solve_mip(self, timeout, insert_cuts=True):
        grb = self.grb
        # The MIP mode must have been set beforehand.
        assert self.mode == "mip-exact"
        self.insert_cuts = insert_cuts

        self.global_lb_upper_bound = float("inf")

        if self.lower_bounds[-1].item() > 0:
            print("Early stopping")
            return (False, None, 0)
        if timeout is not None:
            self.model.setParam('TimeLimit', timeout)

        global_lb = self.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        nb_visited_states = self.model.nodeCount

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution (what's the assumption, here? Feasible so small that there's no counterexample?)
            return (False, None, nb_visited_states)

        elif self.model.status is grb.GRB.OPTIMAL:
            # There is a feasible solution.
            return (global_lb < 0, global_lb, nb_visited_states)

        elif self.model.status is grb.GRB.INTERRUPTED:
            return (self.interrupted_sat, None, nb_visited_states)

        elif self.model.status is grb.GRB.TIME_LIMIT:
            # We timed out, return a None Status
            return (None, None, nb_visited_states)
        else:
            raise Exception("Unexpected Status code")

    # Method potentially useful for branch and bound.
    def get_lower_bound(self, domain, force_optim=False):
        raise NotImplementedError

    def define_linear_approximation(self, input_domain, int_bounds_provided=False, compute_final_layer=True,
                                    n_threads=1, x_idx_max=-1):
        # if the upper constraints are made tighter and to depend only on \hat{l} rather than l, the force_optim
        # flag could be introduced to avoid computing unnecessary bounds. This is never the case with the Anderson
        # relaxation.
        """
        Get a linear approximation of a neural network using Anderson et al.'s [1] convex hull of the composition of a
        linear function and a ReLu.

        :param input_domain: Tensor containing in each row the lower and upper bound for the corresponding input dimension
        :param int_bounds_provided: if True, bounds are not computed but taken from self.lower_bounds and self.upper_bounds.
        :param compute_final_layer: whether to compute the final layer bounds
        :param n_threads: number of threads to use in the solution of each Gurobi model
        :return: nothing (the upper/lower bounds computation results are stored in self.lower_bounds and self.upper_bounds)
        """
        grb = self.grb
        self.device = input_domain[0].device
        # TODO: introduce time limits per bounds as in planet rel (network_linear_approximation, time_limit)

        if not int_bounds_provided:
            self.lower_bounds = []
            self.upper_bounds = []
            self.pre_lower_bounds = []
            self.pre_upper_bounds = []
        elif not (self.lower_bounds and self.upper_bounds):
            raise ValueError("compute_bounds=False but bounds haven't been provided in any way.")
        self.ambiguous_relus = []  # Keep track of which ReLUs are ambiguous.
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        if not self.model_built:
            gurobi_env = grb.Env()
            self.model = grb.Model(env=gurobi_env)
            self.model.setParam('OutputFlag', False)
            self.model.setParam('Threads', n_threads)
            if self.mode == "mip-exact":
                # Shut off Gurobi's cut generation, as in Anderson's paper [1].
                # self.model.setParam('Cuts', 0)
                # self.model.setParam('CutPasses', 0)
                # [2]
                self.model.setParam('PreCrush', 1)
            self.preact_x_vars = []
            self.gurobi_x_vars = []
            self.gurobi_z_vars = []
        else:
            # Remove previously added constraints.
            for constraint in self.active_anderson_constraints:
                self.model.remove(constraint)
            for c_added_exp_constraints in self.added_exp_constraints:
                for constraint in c_added_exp_constraints:
                    self.model.remove(constraint)
            self.model.update()
            self.active_anderson_constraints = []
            self.applied_cuts = 0
            self.old_applied_cuts = -1

        self.input_domain = input_domain
        device = self.input_domain.device
        ## Do the input layer, which is a special case
        inp_lbs, inp_ubs, inp_gurobi_vars = self.create_input_variables(input_domain)
        if not int_bounds_provided:
            self.lower_bounds.append(torch.tensor(inp_lbs, dtype=torch.float, device=self.device))
            self.upper_bounds.append(torch.tensor(inp_ubs, dtype=torch.float, device=self.device))
        else:
            self.lower_bounds[0] = torch.tensor(inp_lbs, dtype=torch.float, device=self.device)
            self.upper_bounds[0] = torch.tensor(inp_ubs, dtype=torch.float, device=self.device)
        if not self.model_built:
            self.gurobi_x_vars.append(inp_gurobi_vars)
            self.preact_x_vars.append(inp_gurobi_vars)

        self.last_relu_index = 0  # Store the last ReLU index we got to when creating the model.

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        x_idx = 1
        for layer_idx, layer in enumerate(self.layers[:-1]):
            # stop if we've reached x_idx_max
            if x_idx_max > 0 and x_idx > x_idx_max:
                break
            new_layer_gurobi_x_vars = []
            new_layer_gurobi_z_vars = []
            new_layer_ambiguous_relus = []

            if type(layer) is nn.ReLU:

                # Retrieve lower, upper bounds, and variables of previous layer.
                l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, layer_idx)
                previous_layer = self.layers[layer_idx-1]
                if not int_bounds_provided:
                    # Compute upper and lower bounds of the linear part of the layer.
                    # start of logging time for optimization
                    if x_idx == self.store_bounds_progress:
                        self.logger.start_timing()
                    preact_lower, preact_upper, preact_vars = self.compute_pre_activation_bounds(
                        x_idx, previous_layer, x_km1_vars)
                else:
                    preact_lower = self.pre_lower_bounds[x_idx - 1]
                    preact_upper = self.pre_upper_bounds[x_idx - 1]
                    preact_vars = self.compute_pre_activation_bounds(x_idx, previous_layer, x_km1_vars, optimize=False)
                self.model.update()

                if type(previous_layer) is nn.Linear:

                    for neuron_idx in range(previous_layer.weight.size(0)):
                        non_zero_indices = torch.nonzero(previous_layer.weight[neuron_idx, :]).flatten().tolist()
                        non_zero_indices_set = set(non_zero_indices)
                        powerset = None
                        if self.mode == "lp-all":
                            # Generate useful variables for the exponentially many constraints. The powerset does not
                            # include the empty and the full set, whose constraints are added separately.
                            powerset = list(chain.from_iterable(
                                combinations(non_zero_indices, r) for r in range(1, len(non_zero_indices))))

                        x_var, z_var, ambiguous_relu = self.add_anderson_relu_neuron(
                            x_idx,
                            layer_idx,
                            previous_layer,
                            neuron_idx,
                            preact_lower[neuron_idx].item(),
                            preact_upper[neuron_idx].item(),
                            preact_vars[neuron_idx],
                            non_zero_indices_set,
                            powerset
                        )

                        new_layer_gurobi_x_vars.append(x_var)
                        new_layer_gurobi_z_vars.append(z_var)
                        new_layer_ambiguous_relus.append(ambiguous_relu)

                elif type(previous_layer) is nn.Conv2d:
                    # convolutional layers are dealt with separately to take their structure (and of their input) into account.

                    # Compute interval propagation upper and lower bounds. Used when adding the big-M constraints.
                    pos_weight = torch.clamp(previous_layer.weight, 0, None)
                    neg_weight = torch.clamp(previous_layer.weight, None, 0)
                    M_minus = (nn.functional.conv2d(l_km1, pos_weight, previous_layer.bias,
                                                    previous_layer.stride, previous_layer.padding,
                                                    previous_layer.dilation, previous_layer.groups)
                               + nn.functional.conv2d(u_km1, neg_weight, None,
                                                      previous_layer.stride, previous_layer.padding,
                                                      previous_layer.dilation, previous_layer.groups))

                    # Iterate over all this layer's neurons (they're arranged in 4d: batch x channel x row x column)
                    for out_chan_idx in range(M_minus.size(1)):
                        out_chan_x_vars = []
                        out_chan_z_vars = []
                        out_chan_amb_relus = []
                        non_zero_indices = torch.nonzero(previous_layer.weight[out_chan_idx, :, :, :]).tolist()  # contains list of index tuples corresponding to nonzero entries in the conv filter
                        non_zero_indices_set = set([tuple(ind_list) for ind_list in non_zero_indices])
                        powerset = None
                        if self.mode == "lp-all":
                            # Generate useful variables for the exponentially many constraints. The powerset does not
                            # include the empty and the full set, whose constraints are added separately.
                            powerset = list(chain.from_iterable(combinations(non_zero_indices, r) for r in range(1, len(non_zero_indices))))
                        for out_row_idx in range(M_minus.size(2)):
                            out_row_x_vars = []
                            out_row_z_vars = []
                            out_row_amb_relus = []
                            for out_col_idx in range(M_minus.size(3)):

                                x_var, z_var, ambiguous_relu = self.add_anderson_relu_neuron(
                                    x_idx,
                                    layer_idx,
                                    previous_layer,
                                    (out_chan_idx, out_row_idx, out_col_idx),
                                    preact_lower[out_chan_idx][out_row_idx][out_col_idx].item(),
                                    preact_upper[out_chan_idx][out_row_idx][out_col_idx].item(),
                                    preact_vars[out_chan_idx][out_row_idx][out_col_idx],
                                    non_zero_indices_set,
                                    powerset
                                )

                                out_row_x_vars.append(x_var)
                                out_row_z_vars.append(z_var)
                                out_row_amb_relus.append(ambiguous_relu)
                            out_chan_x_vars.append(out_row_x_vars)
                            out_chan_z_vars.append(out_row_z_vars)
                            out_chan_amb_relus.append(out_row_amb_relus)
                        new_layer_gurobi_x_vars.append(out_chan_x_vars)
                        new_layer_gurobi_z_vars.append(out_chan_z_vars)
                        new_layer_ambiguous_relus.append(out_chan_amb_relus)

                if not int_bounds_provided:
                    self.lower_bounds.append(torch.clamp(preact_lower, 0, None))
                    self.upper_bounds.append(torch.clamp(preact_upper, 0, None))
                    self.pre_lower_bounds.append(preact_lower)
                    self.pre_upper_bounds.append(preact_upper)
                if not self.model_built:
                    self.preact_x_vars.append(preact_vars)
                    self.gurobi_x_vars.append(new_layer_gurobi_x_vars)
                    self.gurobi_z_vars.append(new_layer_gurobi_z_vars)
                self.ambiguous_relus.append(new_layer_ambiguous_relus)
                x_idx += 1
                self.last_relu_index = layer_idx
                self.model.update()

            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                continue
            else:
                raise NotImplementedError

        if (x_idx_max < 0 or x_idx <= x_idx_max):
            if compute_final_layer:
                # start of logging time for optimization
                if x_idx == self.store_bounds_progress:
                    self.logger.start_timing()
                self.compute_final_layer_bounds()
            elif not self.model_built:
                x_idx = len(self.lower_bounds) - 1
                l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, len(self.layers)-1)
                preact_vars = self.compute_pre_activation_bounds(x_idx, self.layers[-1], x_km1_vars, optimize=False)
                self.preact_x_vars.append(preact_vars)

        # unsqueeze the bounds to comply with batched SaddleLP
        self.lower_bounds = [lbs.to(device).unsqueeze(0) for lbs in self.lower_bounds]
        self.upper_bounds = [ubs.to(device).unsqueeze(0) for ubs in self.upper_bounds]

        self.model.update()

    def add_anderson_relu_neuron(
        self,
        x_idx,
        layer_idx,
        linear_layer,
        neuron_coordinates,
        c_preact_lower,
        c_preact_upper,
        c_preact_var,
        non_zero_indices_set,
        powerset
    ):
        """
        Add all the variables and constraints corresponding to the composition of a linear function and a ReLU according
        to the Anderson MIP formulation [1].
        :param x_idx: layer number counting hidden layers (ReLUs not counted)
        :param layer_idx: layer number as in the layer list (counts ReLUs)
        :param linear_layer: layer representing the linear operation
        :param neuron_coordinates: indices related to the current neuron (one index for Linear, a triple for Conv2d)
        :param c_preact_lower: current pre-activation lower bound
        :param c_preact_upper: current pre-activation upper bound
        :param c_preact_var: current pre-activation Gurobi var
        :param c_m_minus: previous layer interval propagation lower bound
        :param c_m_plus: previous layer interval propagation upper bound
        :param non_zero_indices_set: set of nonzero indices (possibly a set of tuples for cnn)
        :param powerset: all possible subsets of non_zero_indices_set
        :return: neuron variable, z neuron variable, neuron dual vars, flag for
        whether the ReLU is ambiguous.
        """
        grb = self.grb
        if type(linear_layer) == nn.Linear:
            neuron_idx = neuron_coordinates
            out_gurobi_format = f"{x_idx}_{neuron_idx}"
        else:
            # Conv2d layer.
            out_chan_idx = neuron_coordinates[0]
            out_row_idx = neuron_coordinates[1]
            out_col_idx = neuron_coordinates[2]
            out_gurobi_format = f"{x_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]"

        # Initialize dual variables to None.
        alpha_0 = None; alpha_1 = None; beta = None; gamma_l = None; gamma_u = None; delta_0 = None; delta_1 = None

        l_k = max(0, c_preact_lower)
        u_k = max(0, c_preact_upper)
        z_l = 0 if (c_preact_lower < 0) or (c_preact_lower == 0 and c_preact_upper == 0) else 1
        z_u = 0 if (c_preact_upper <= 0) else 1
        if not self.model_built:
            x_var = self.model.addVar(
                lb=l_k,
                ub=u_k,
                obj=0,
                vtype=grb.GRB.CONTINUOUS,
                name=f'x_lay{out_gurobi_format}'
            )
            z_var = self.model.addVar(
                lb=z_l,
                ub=z_u,
                obj=0,
                vtype=grb.GRB.CONTINUOUS if self.mode in ["lp-cut", "lp-all"] else grb.GRB.BINARY,
                name=f'z_lay{out_gurobi_format}'
            )
        else:
            x_var = self.gurobi_x_vars[x_idx][neuron_idx] if type(linear_layer) == nn.Linear else \
                self.gurobi_x_vars[x_idx][out_chan_idx][out_row_idx][out_col_idx]
            x_var.lb = l_k
            x_var.ub = u_k
            z_var = self.gurobi_z_vars[x_idx-1][neuron_idx] if type(linear_layer) == nn.Linear else \
                self.gurobi_z_vars[x_idx-1][out_chan_idx][out_row_idx][out_col_idx]
            z_var.lb = z_l
            z_var.ub = z_u

        if c_preact_lower >= 0 and c_preact_upper >= 0:
            # The ReLU is always passing, set upper and lower bound to the linear layer's.
            # New x variable equal to the output of the linear layer.
            alpha_0 = self.model.addConstr(x_var == c_preact_var)
            self.active_anderson_constraints.append(alpha_0)
            # Create new z variable always 1.
            ambiguous_relu = False
        elif c_preact_lower <= 0 and c_preact_upper <= 0:
            # ReLU always blocking.
            # Create new z variable always 0.
            ambiguous_relu = False
        else:
            # Ambiguous ReLU. We need to use [1]'s convex hull.
            alpha_1 = self.model.addConstr(x_var >= c_preact_var)
            self.active_anderson_constraints.append(alpha_1)
            ambiguous_relu = True

            beta = []
            # The constraints corresponding to the PLANET relaxation. Further constraints added via callback (if not
            # done below).
            b1 = self.model.addConstr(x_var <= c_preact_var - c_preact_lower * (1 - z_var))
            b2 = self.model.addConstr(x_var <= c_preact_upper * z_var)
            beta.extend([b1, b2])

            if self.mode == "lp-all":
                # Add all the exponentially many tightening constraints.
                for index_list in powerset:
                    if type(linear_layer) is nn.Linear:
                        negative_list = non_zero_indices_set - set(index_list)
                    else:
                        negative_list = non_zero_indices_set - set([tuple(ind_list) for ind_list in index_list])
                    tighten_expression = self.get_tighten_constraint(x_idx, layer_idx, neuron_coordinates,
                                                                     index_list, negative_list, z_var)
                    c_beta = self.model.addConstr(x_var <= tighten_expression)
                    beta.append(c_beta)
            self.active_anderson_constraints.extend(beta)

        return x_var, z_var, ambiguous_relu

    def compute_final_layer_bounds(self, ub_only=False):
        """
        Compute the bounds for the last network layer. Assumes all previous bounds have been computed (or provided)
        and stored in self.lower_bounds and self.upper_bounds.
        :param ub_only: Compute upper bounds only
        """

        if not self.lower_bounds:
            raise ValueError("compute_final_layer_bounds called without any bounds stored.")

        x_idx = len(self.lower_bounds) - 1
        layer_idx = len(self.layers)-1

        # Retrieve lower, upper bounds, and variables of previous layer.
        l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, layer_idx)

        # If we're on the final level, compute the layer output and optimize over it.
        preact_lower, preact_upper, preact_vars = self.compute_pre_activation_bounds(x_idx, self.layers[-1], x_km1_vars,
                                                                                     ub_only=ub_only)

        self.lower_bounds.append(preact_lower)
        self.upper_bounds.append(preact_upper)
        self.preact_x_vars.append(preact_vars)

    def compute_pre_activation_bounds(self, x_idx, layer, x_km1_vars, ub_only=False, counterexample_verification=False,
                                      single_neuron=False, optimize=True):
        """
        Compute the upper and lower bounds corresponding to "pre-activation" variables (i.e., before the ReLU, after
        a linear operation).
        :param x_idx: x variable number (for naming  purposes)
        :param layer: linear part of layer whose output we compute the bounds for
        :param x_km1_vars: the Gurobi x_{k-1} variables, in a list
        :param ub_only: Compute upper bounds only
        :param single_neuron: False means compute a batch of bounds (sequentially). Else expects
            (neuron_coordinate, upper_bound), upper_bound is whether to compute UB instead of LB.
        :param optimize: whether to perform the optimization (else it only adds the vars)
        :return: pre-activation lower bounds (torch.tensor), pre-activation upper bounds (torch.tensor), list of gurobi
                    variables associated to the linear operation output
        """
        grb = self.grb
        # store bounds in the format of proxlp/explp for logging purposes.
        out_shape = layer(self.lower_bounds[x_idx-1].unsqueeze(0)).squeeze(0).shape if type(layer) is nn.Conv2d else \
            layer.weight.shape[0]
        # no. of neurons over which we're optimizing
        self.n_opt_neurons = 2 * prod(out_shape) if not single_neuron else 2

        if len(self.lower_bounds) >= (x_idx + 1) and not single_neuron:
            # if they are available, initialize c_bounds with whatever bounds we built the model with
            self.c_bounds = torch.cat([-self.upper_bounds[x_idx], self.lower_bounds[x_idx]])
        else:
            self.c_bounds = -float("inf") * torch.ones(self.n_opt_neurons, dtype=torch.float, device=self.device)
        self.added_exp_constraints = [[]] * self.n_opt_neurons

        do_upper_bound = single_neuron[1] if single_neuron else True
        do_lower_bound = not single_neuron[1] if single_neuron else (not ub_only)

        # We will now compute the requested lower bound
        allowed_statuses = [grb.GRB.OPTIMAL]
        if counterexample_verification:
            allowed_statuses.extend([grb.GRB.INFEASIBLE, grb.GRB.INF_OR_UNBD])

        # store the primal solution in the format of explp for initialization purposes. Those associated to ub and lb
        # are stored separately for ordering purposes
        x_solution_ub = [None] * x_idx
        x_solution_lb = [None] * x_idx
        z_solution_ub = [None] * (x_idx - 1)
        z_solution_lb = [None] * (x_idx - 1)
        self.c_x_idx = x_idx

        preact_lower = []
        preact_upper = []
        preact_vars = []

        gurobi_callback = lambda model, where: self.insert_tighten_cut(
            model, where, counterexample_verification=counterexample_verification)

        if self.neurons_per_layer != -1 and optimize and x_idx > 1:
            # Determine neuron set to apply the Anderson relaxation to.
            intercept_scores = self.compute_crown_intercept(x_idx)

        if type(layer) is nn.Linear:
            neurons = list(range(layer.weight.size(0))) if not single_neuron else [single_neuron[0]]
            # for lp-cut, this computes big-M bounds only, here (and collect them)
            for neuron_idx in neurons:
                # Avoid numerical instability: on Wide one neuron has ub-lb = 1e-7
                c_lb = self.pre_lower_bounds[x_idx - 1][neuron_idx]
                c_ub = self.pre_upper_bounds[x_idx - 1][neuron_idx]
                bound_diff = c_ub - c_lb
                if bound_diff < self.bounds_num_tolerance:
                    c_ub += self.bounds_num_tolerance
                    if c_ub - self.bounds_num_tolerance <= 0:
                        c_ub = min(0, c_ub)
                    c_lb -= self.bounds_num_tolerance
                    if c_lb + self.bounds_num_tolerance >= 0:
                        c_lb = max(0, c_lb)
                if not self.model_built:
                    lin_expr = self.get_layer_linear_expression(layer, x_km1_vars, neuron_idx, grb)
                    x_var = self.model.addVar(lb=c_lb,
                                              ub=c_ub,
                                              obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'x_preact_lay{x_idx}_{neuron_idx}')
                    self.model.addConstr(x_var == lin_expr)
                else:
                    x_var = self.preact_x_vars[x_idx][neuron_idx]
                    x_var.lb = c_lb
                    x_var.ub = c_ub

                preact_vars.append(x_var)
                if not optimize:
                    continue

                # Reset count for applied cuts to allow cuts to be added to the last layers as well (otherwise, we "run out of cuts" too soon)
                self.pending_bound_var = x_var
                self.pending_lower_bound = True
                # Optimize for the last layer's lower and upper bounds.
                self.c_neuron = int(self.n_opt_neurons / 2) + neuron_idx
                if do_lower_bound:
                    self.model.setObjective(x_var, grb.GRB.MINIMIZE)
                    l_k = self.optimize_var(x_var, gurobi_callback, allowed_statuses, store_input=True)
                    if not math.isinf(l_k):
                        self.collect_primal_solution(x_solution_lb, z_solution_lb)  # store primal solution in x_solution and z_solution
                else:
                    # if ub_only, avoid computing lower bounds, and retrieve them from what's available (in case it is)
                    if len(self.lower_bounds) > x_idx:
                        l_k = self.lower_bounds[x_idx][neuron_idx]
                    else:
                        l_k = -float('inf')
                self.c_bounds[self.c_neuron] = l_k

                # Let's now compute an upper bound
                self.c_neuron = neuron_idx
                self.pending_lower_bound = False
                if do_upper_bound:
                    self.model.setObjective(x_var, grb.GRB.MAXIMIZE)
                    # We have computed the upper bounds.
                    u_k = self.optimize_var(x_var, gurobi_callback, allowed_statuses, store_input=False)
                    self.c_bounds[self.c_neuron] = -u_k
                    if not math.isinf(u_k):
                        self.collect_primal_solution(x_solution_ub, z_solution_ub)  # store primal solution in x_solution and z_solution
                else:
                    if len(self.lower_bounds) > x_idx:
                        u_k = self.upper_bounds[x_idx][neuron_idx]
                    else:
                        u_k = float('inf')

                preact_lower.append(l_k)
                preact_upper.append(u_k)

            if not optimize:
                return preact_vars

            self.x_solution, self.z_solution = AndersonLinearizedNetwork.combine_primal_solutions(
                x_solution_ub, x_solution_lb, z_solution_ub, z_solution_lb)
            x_solution_ub = [None] * x_idx
            x_solution_lb = [None] * x_idx
            z_solution_ub = [None] * (x_idx - 1)
            z_solution_lb = [None] * (x_idx - 1)

            if self.c_x_idx == self.store_bounds_progress:
                self.logger.add_point(self.c_x_idx, self.c_bounds.clone())

            if self.mode == "lp-cut" and self.applied_cuts < self.cut_treshold:
                # insert cuts over the neurons whose bounds we are computing.
                for neuron_idx in neurons:
                    self.pending_bound_var = preact_vars[neuron_idx]
                    # Optimize for the last layer's lower and upper bounds.
                    self.c_neuron = int(self.n_opt_neurons / 2) + neuron_idx
                    if do_lower_bound and (not math.isinf(preact_lower[neuron_idx])):
                        if preact_lower[neuron_idx] > self.decision_boundary and counterexample_verification:
                            # for BaB, no need to go further if the constraint is over the BaB decision boundary
                            continue
                        if self.cuts_per_neuron:
                            self.applied_cuts = 0
                        if self.neurons_per_layer != -1:
                            self.tight_set = self.get_tight_neuron_set(self.c_neuron, self.neurons_per_layer, intercept_scores,
                                                                       self.lower_bounds, self.upper_bounds)
                        self.pending_lower_bound = True
                        self.model.setObjective(self.pending_bound_var, grb.GRB.MINIMIZE)
                        # We compute a lower bound
                        l_k = self.insert_tighten_cut(self.model, "lp-cut", first_call=True,
                                                      counterexample_verification=counterexample_verification)
                        self.collect_primal_solution(x_solution_lb,
                                                     z_solution_lb)  # store primal solution in x_solution and z_solution
                        preact_lower[neuron_idx] = l_k
                        self.c_bounds[self.c_neuron] = l_k
                    # IMPORTANT: remove the added cuts before changing bound computation
                    for added_constraint in self.added_exp_constraints[self.c_neuron]:
                        self.model.remove(added_constraint)
                    self.added_exp_constraints[self.c_neuron] = []

                    # Let's now compute an upper bound
                    if self.cuts_per_neuron:
                        self.applied_cuts = 0
                    self.c_neuron = neuron_idx
                    if math.isinf(preact_upper[neuron_idx]):
                        continue
                    self.pending_lower_bound = False
                    if do_upper_bound:
                        if self.neurons_per_layer != -1:
                            self.tight_set = self.get_tight_neuron_set(self.c_neuron, self.neurons_per_layer, intercept_scores,
                                                                       self.lower_bounds, self.upper_bounds)
                        self.model.setObjective(self.pending_bound_var, grb.GRB.MAXIMIZE)
                        # We compute the upper bounds.
                        u_k = self.insert_tighten_cut(self.model, "lp-cut", first_call=True,
                                                      counterexample_verification=counterexample_verification)
                        preact_upper[neuron_idx] = u_k
                        self.c_bounds[self.c_neuron] = -u_k
                        self.collect_primal_solution(x_solution_ub,
                                                     z_solution_ub)  # store primal solution in x_solution and z_solution
                        # IMPORTANT: remove the added cuts before changing bound computation
                        for added_constraint in self.added_exp_constraints[self.c_neuron]:
                            self.model.remove(added_constraint)
                        self.added_exp_constraints[self.c_neuron] = []

        else:
            # Convolutional layer.
            # Compute sample output to easily get output dimensions (a forward pass is done elsewhere anyways, in Python loops).
            sample_out = layer(torch.ones((len(x_km1_vars), len(x_km1_vars[0]), len(x_km1_vars[0][0])),
                                          dtype=torch.float, device=self.device).unsqueeze(0))

            chan_neurons = list(range(sample_out.size(1)) if not single_neuron else [single_neuron[0][0]])
            row_neurons = list(range(sample_out.size(2)) if not single_neuron else [single_neuron[0][1]])
            col_neurons = list(range(sample_out.size(3)) if not single_neuron else [single_neuron[0][2]])

            # Iterate over all this layer's neurons (they're arranged in 4d: batch x channel x row x column)
            for out_chan_idx in chan_neurons:
                out_chan_lbs = []
                out_chan_ubs = []
                out_chan_x_vars = []
                for out_row_idx in row_neurons:
                    out_row_lbs = []
                    out_row_ubs = []
                    out_row_x_vars = []
                    for out_col_idx in col_neurons:
                        c_neur_idx = out_chan_idx * sample_out.size(2) * sample_out.size(3) + out_row_idx * \
                                     sample_out.size(3) + out_col_idx

                        # Avoid numerical instability: on Wide one neuron has ub-lb = 1e-7
                        c_lb = self.pre_lower_bounds[x_idx - 1][out_chan_idx][out_row_idx][out_col_idx]
                        c_ub = self.pre_upper_bounds[x_idx - 1][out_chan_idx][out_row_idx][out_col_idx]
                        bound_diff = c_ub - c_lb
                        if bound_diff < self.bounds_num_tolerance:
                            c_ub += self.bounds_num_tolerance
                            if c_ub - self.bounds_num_tolerance <= 0:
                                c_ub = min(0, c_ub)
                            c_lb -= self.bounds_num_tolerance
                            if c_lb + self.bounds_num_tolerance >= 0:
                                c_lb = max(0, c_lb)
                        if not self.model_built:
                            # Compute W_k * x_{k-1} + b_k, as a Gurobi linear expression.
                            lin_expr = self.get_layer_linear_expression(layer, x_km1_vars,
                                                                        (out_chan_idx, out_row_idx, out_col_idx), grb)
                            x_var = self.model.addVar(lb=c_lb,
                                                      ub=c_ub,
                                                      obj=0,
                                                      vtype=grb.GRB.CONTINUOUS,
                                                      name=f'x_preact_lay{x_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(x_var == lin_expr)
                        else:
                            x_var = self.preact_x_vars[x_idx][out_chan_idx][out_row_idx][out_col_idx]
                            x_var.lb = c_lb
                            x_var.ub = c_ub

                        out_row_x_vars.append(x_var)
                        if not optimize:
                            continue

                        # Reset count for applied cuts to allow cuts to be added to the last layers as well (otherwise, we "run out of cuts" too soon)
                        self.pending_bound_var = x_var
                        self.pending_lower_bound = True
                        # Optimize for the last layer's lower and upper bounds.
                        self.c_neuron = int(self.n_opt_neurons / 2) + c_neur_idx
                        if do_lower_bound:
                            if self.neurons_per_layer != -1:
                                self.tight_set = self.get_tight_neuron_set(self.c_neuron, self.neurons_per_layer,
                                                                           intercept_scores, self.lower_bounds, self.upper_bounds)
                            self.model.setObjective(x_var, grb.GRB.MINIMIZE)
                            l_k = self.optimize_var(x_var, gurobi_callback, allowed_statuses, store_input=True)
                            if not math.isinf(l_k):
                                self.collect_primal_solution(x_solution_lb, z_solution_lb)  # store primal solution in x_solution and z_solution
                        else:
                            # if ub_only, avoid computing lower bounds, and retrieve them from what's available (in case it is)
                            if len(self.lower_bounds) > x_idx:
                                l_k = self.lower_bounds[x_idx][out_chan_idx][out_row_idx][out_col_idx]
                            else:
                                l_k = -float('inf')
                        self.c_bounds[self.c_neuron] = l_k

                        # Let's now compute an upper bound
                        self.c_neuron = c_neur_idx
                        self.pending_lower_bound = False
                        if do_upper_bound:
                            if self.neurons_per_layer != -1:
                                self.tight_set = self.get_tight_neuron_set(self.c_neuron, self.neurons_per_layer,
                                                                           intercept_scores, self.lower_bounds, self.upper_bounds)
                            self.model.setObjective(x_var, grb.GRB.MAXIMIZE)
                            u_k = self.optimize_var(x_var, gurobi_callback, allowed_statuses, store_input=False)
                            self.c_bounds[self.c_neuron] = -u_k
                            if not math.isinf(u_k):
                                self.collect_primal_solution(x_solution_ub, z_solution_ub)  # store primal solution in x_solution and z_solution
                        else:
                            if len(self.lower_bounds) > x_idx:
                                u_k = self.upper_bounds[x_idx][out_chan_idx][out_row_idx][out_col_idx]
                            else:
                                u_k = float('inf')

                        out_row_lbs.append(l_k)
                        out_row_ubs.append(u_k)
                    out_chan_x_vars.append(out_row_x_vars)
                    out_chan_lbs.append(out_row_lbs)
                    out_chan_ubs.append(out_row_ubs)

                preact_lower.append(out_chan_lbs)
                preact_upper.append(out_chan_ubs)
                preact_vars.append(out_chan_x_vars)

            if not optimize:
                return preact_vars

            self.x_solution, self.z_solution = AndersonLinearizedNetwork.combine_primal_solutions(
                x_solution_ub, x_solution_lb, z_solution_ub, z_solution_lb)
            x_solution_ub = [None] * x_idx
            x_solution_lb = [None] * x_idx
            z_solution_ub = [None] * (x_idx - 1)
            z_solution_lb = [None] * (x_idx - 1)

            if self.c_x_idx == self.store_bounds_progress:
                self.logger.add_point(self.c_x_idx, self.c_bounds.clone())

            if self.mode == "lp-cut" and self.applied_cuts < self.cut_treshold:
                # insert cuts over the neurons whose bounds we are computing.
                for out_chan_idx in chan_neurons:
                    for out_row_idx in row_neurons:
                        for out_col_idx in col_neurons:
                            c_neur_idx = out_chan_idx * sample_out.size(2) * sample_out.size(3) + out_row_idx * \
                                         sample_out.size(3) + out_col_idx

                            if self.cuts_per_neuron:
                                self.applied_cuts = 0
                            self.pending_bound_var = preact_vars[out_chan_idx][out_row_idx][out_col_idx]
                            # Optimize for the last layer's lower and upper bounds.
                            self.c_neuron = int(self.n_opt_neurons / 2) + c_neur_idx
                            self.pending_lower_bound = True
                            if do_lower_bound and (not math.isinf(preact_lower[out_chan_idx][out_row_idx][out_col_idx])):
                                if preact_lower[out_chan_idx][out_row_idx][out_col_idx] > self.decision_boundary and counterexample_verification:
                                    # for BaB, no need to go further if the constraint is over the BaB decision boundary
                                    continue
                                self.model.setObjective(self.pending_bound_var, grb.GRB.MINIMIZE)
                                # We compute a lower bound
                                l_k = self.insert_tighten_cut(self.model, "lp-cut", first_call=True,
                                                              counterexample_verification=counterexample_verification)
                                self.collect_primal_solution(x_solution_lb,
                                                             z_solution_lb)  # store primal solution in x_solution and z_solution
                                preact_lower[out_chan_idx][out_row_idx][out_col_idx] = l_k
                                self.c_bounds[self.c_neuron] = l_k
                            # IMPORTANT: remove the added cuts before changing bound computation
                            for added_constraint in self.added_exp_constraints[self.c_neuron]:
                                self.model.remove(added_constraint)
                            self.added_exp_constraints[self.c_neuron] = []

                            # Let's now compute an upper bound
                            self.c_neuron = c_neur_idx
                            if self.cuts_per_neuron:
                                self.applied_cuts = 0

                            if math.isinf(preact_upper[out_chan_idx][out_row_idx][out_col_idx]):
                                continue

                            self.pending_lower_bound = False
                            if do_upper_bound:
                                self.model.setObjective(self.pending_bound_var, grb.GRB.MAXIMIZE)
                                # We compute an upper bound
                                u_k = self.insert_tighten_cut(self.model, "lp-cut", first_call=True,
                                                        counterexample_verification=counterexample_verification)
                                preact_upper[out_chan_idx][out_row_idx][out_col_idx] = u_k
                                self.c_bounds[self.c_neuron] = -u_k
                                self.collect_primal_solution(x_solution_ub,
                                                             z_solution_ub)  # store primal solution in x_solution and z_solution
                                # IMPORTANT: remove the added cuts before changing bound computation
                                for added_constraint in self.added_exp_constraints[self.c_neuron]:
                                    self.model.remove(added_constraint)
                                self.added_exp_constraints[self.c_neuron] = []

        if self.mode != "lp-cut" or self.cut_treshold > 0:
            self.x_solution, self.z_solution = AndersonLinearizedNetwork.combine_primal_solutions(
                x_solution_ub, x_solution_lb, z_solution_ub, z_solution_lb)

        if not single_neuron:
            return torch.tensor(preact_lower, dtype=torch.float, device=self.device), \
                   torch.tensor(preact_upper, dtype=torch.float, device=self.device), preact_vars
        else:
            return torch.tensor(u_k, dtype=torch.float, device=self.device) if single_neuron[1] else \
                torch.tensor(l_k, dtype=torch.float, device=self.device)

    def optimize_var(self, var, gurobi_callback, allowed_statuses, store_input=False):
        # Assuming the objective has already been set, optimize over a variable (providing a gurobi callback for the
        #  MIP case, and the allowed optimization statuses). Returns the var's value at the end of the opt.
        grb = self.grb
        if self.mode in ["lp-cut", "lp-all"]:
            self.model.optimize()
            assert self.model.status in allowed_statuses
        else:
            self.model.optimize(gurobi_callback)
            if self.model.status not in allowed_statuses + [grb.GRB.NODE_LIMIT, grb.GRB.INTERRUPTED, grb.GRB.TIME_LIMIT]:
                raise RuntimeError(f"MIP crashed with status code {self.model.status}")

        if self.model.status in [grb.GRB.INFEASIBLE, grb.GRB.INF_OR_UNBD, grb.GRB.INTERRUPTED, grb.GRB.TIME_LIMIT]:  # (infeasible, interrupted)
            out = float('inf')
            if store_input:
                self.lb_input = self.lower_bounds[0].clone().unsqueeze(0)  # dummy lower bound input
        else:
            # We have computed a lower bound
            out = var.X
            if store_input:
                self.lb_input = self.get_input_list()
        return out

    @staticmethod
    def get_layer_linear_expression(layer, x_km1_vars, out_coordinates, grb, custom_bias=None, custom_forward=False):
        """
        Compute Gurobi linear expression for a layer with a linear operation (Linear or Conv2d), for the neuron identified
        by the out_coordinates.
        :param layer: the layer to compute it on
        :param x_km1_vars: the previous layer variables (triple lit for Conv2d, list for Linear)
            :param out_coordinates: indices related to the current neuron (one index for Linear, a triple for Conv2d)
        :param custom_bias: custom bias instead of b. Not used in this relaxation (i.e., set to 1), but
        :param custom_forward: custom forward pass (passes list of expressions instead of list of variables)
        employed for instance in the L1 relaxation.
        :return: the Gurobi linear expression
        """

        if type(layer) is nn.Linear:
            neuron_idx = out_coordinates
            lin_expr = layer.bias[neuron_idx].item() if custom_bias is None else custom_bias
            if not custom_forward:
                lin_expr += grb.LinExpr(layer.weight[neuron_idx, :], x_km1_vars)
            else:
                for idx, carg in enumerate(x_km1_vars):
                    coeff = layer.weight[neuron_idx, idx]
                    if abs(coeff) > 1e-6:
                        lin_expr += coeff * carg

        elif type(layer) is nn.Conv2d:
            out_chan_idx = out_coordinates[0]
            out_row_idx = out_coordinates[1]
            out_col_idx = out_coordinates[2]
            # Compute W_k * x_{k-1} + b_k, as a Gurobi linear expression.
            lin_expr = layer.bias[out_chan_idx].item() if custom_bias is None else custom_bias
            for in_chan_idx in range(layer.weight.shape[1]):
                for ker_row_idx in range(layer.weight.shape[2]):
                    in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                    if (in_row_idx < 0) or (in_row_idx >= len(x_km1_vars[0])):
                        # This is padding -> value of 0
                        continue
                    for ker_col_idx in range(layer.weight.shape[3]):
                        in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                        if (in_col_idx < 0) or (in_col_idx >= len(x_km1_vars[0][0])):
                            # This is padding -> value of 0
                            continue
                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                        if abs(coeff) > 1e-6:
                            lin_expr += coeff * x_km1_vars[in_chan_idx][in_row_idx][in_col_idx]
        else:
            ValueError("Linear expression computation implemented only for linear or convolutional layers")

        return lin_expr

    def get_previous_layer_info(self, x_idx, layer_idx):
        """
        Get lower bounds, upper bounds, and Gurobi variables relative to the last layer.
        :param x_idx: layer number counting hidden layers (ReLUs not counted)
        :param layer_idx: layer number as in the layer list (counts ReLUs)
        :return: last layer lower bounds (torch.tensor), upper bounds (torch.tensor), list of gurobi variables
        """

        c_layer = self.layers[layer_idx]
        if layer_idx == 0 and type(c_layer) is nn.ReLU:
            raise ValueError("Error: ReLU as very first layer.")
        linear_layer = self.layers[layer_idx-1] if type(c_layer) is nn.ReLU else c_layer

        if type(linear_layer) is nn.Linear:
            l_km1 = self.lower_bounds[x_idx - 1]
            u_km1 = self.upper_bounds[x_idx - 1]
            if l_km1.dim() > 1:
                l_km1 = l_km1.flatten()
                u_km1 = u_km1.flatten()
                x_km1_vars = []
                for chan_idx in range(len(self.gurobi_x_vars[x_idx-1])):
                    for row_idx in range(len(self.gurobi_x_vars[x_idx-1][chan_idx])):
                        x_km1_vars.extend(self.gurobi_x_vars[x_idx-1][chan_idx][row_idx])
            else:
                x_km1_vars = self.gurobi_x_vars[x_idx-1]
        elif type(linear_layer) is nn.Conv2d:
            l_km1 = self.lower_bounds[x_idx-1].unsqueeze(0)  # these bounds are 3d: they are images (input to conv), plus the first dimension is the batch size
            u_km1 = self.upper_bounds[x_idx-1].unsqueeze(0)  # unsqueeze makes them 4d to add batch size (1)
            x_km1_vars = self.gurobi_x_vars[x_idx-1]
        else:
            raise ValueError(f"get_x_km1 not implemented for {type(linear_layer)} layer")

        return l_km1, u_km1, x_km1_vars

    def insert_tighten_cut(self, model, where, first_call=False, counterexample_verification=False):
        """
        Gurobi callback function that inserts a cut for the most violated constraint for all the ambiguous ReLU neurons
        that we already added to the model.
        :param model: gurobi callback specifications, returns the model (already available in self.model)
        :param where: Gurobi code indicating where callback function is called from
        """
        grb = self.grb
        allowed_statuses = [grb.GRB.OPTIMAL]
        if counterexample_verification:
            allowed_statuses.extend([grb.GRB.INFEASIBLE, grb.GRB.INF_OR_UNBD])

        # Callback cuts are to be inserted when reaching a new node [2].
        if where == grb.GRB.Callback.MIPNODE or where == "lp-cut":
            # Called when performing operations within a MIP node. At the root, it's called once per cut application [3]
            # and the number of applied cut passes is controlled by parameter CutPasses.
            # At all the other nodes, cuts are applied once https://groups.google.com/forum/#!topic/gurobi/_JUHVqdNy7s

            if self.mode == "mip-exact":
                nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                if (nodeCount % 100) == 0:
                    print(f"Running Nb states visited: {nodeCount}")

            # Get current relaxed solution, set it as the model solution.
            if where == "lp-cut" or model.cbGet(grb.GRB.Callback.MIPNODE_STATUS) == grb.GRB.Status.OPTIMAL:

                if self.mode == "lp-cut":
                    self.old_applied_cuts = self.applied_cuts

                if self.mode == "mip-exact":
                    # check if this LP solution leads to a negative UB on the global LB (terminate, in case)
                    with torch.no_grad():
                        out = self.net(self.get_input_list()).squeeze().item()
                        out = out
                    if out < 0:
                        self.interrupted_sat = True  # SAT
                        model.terminate()

                # Add the most violated constraint for all the ReLU neurons we already added to the model.
                # For each layer, neurons are added in decreasing order of violation amount.
                x_idx = 1
                # If the threshold has been reached already, no need to add more constraints.
                if (self.mode == "mip-exact" and self.insert_cuts) or \
                        (self.mode == "lp-cut" and self.applied_cuts < self.cut_treshold):
                    for layer_idx in range(self.last_relu_index+1):
                        # Nothing to cut a the last layer.
                        if layer_idx == len(self.layers)-1:
                            continue
                        layer = self.layers[layer_idx]

                        if type(layer) is nn.ReLU:
                            previous_layer = self.layers[layer_idx - 1]

                            violation_list = []
                            if type(previous_layer) is nn.Linear:
                                # possibly limit the constraint addition to a subset
                                neuron_range = self.tight_set[x_idx] if self.neurons_per_layer != -1 else \
                                    range(previous_layer.weight.size(0))
                                for neuron_idx in neuron_range:
                                    if self.ambiguous_relus[x_idx-1][neuron_idx]:
                                        is_violated, violated_set, violation = self.most_violated_tighten(
                                            x_idx, layer_idx, neuron_idx, first_call=first_call)
                                        if is_violated:
                                            non_zero_indices = torch.nonzero(
                                                previous_layer.weight[neuron_idx, :]).flatten().tolist()
                                            non_zero_indices_set = set(non_zero_indices)
                                            violation_list.append(
                                                (violated_set, violation, neuron_idx, non_zero_indices_set))

                            elif type(previous_layer) is nn.Conv2d:

                                if self.neurons_per_layer != -1:
                                    # limit the addition to a subset
                                    neuron_range = self.tight_set[x_idx]
                                else:
                                    # Add an Anderson constraint for all neurons of the layer.
                                    # Compute sample output to easily get output dimensions (a forward pass is done elsewhere
                                    # anyways, in Python loops).
                                    l_km1 = self.lower_bounds[x_idx - 1].unsqueeze(0)
                                    sample_out = previous_layer(l_km1)
                                    out_shape = (sample_out.shape[1], sample_out.shape[2], sample_out.shape[3])
                                    neuron_range = range(prod(out_shape))

                                non_zero_indices_sets = {}
                                for c_neuron in neuron_range:
                                    out_chan_idx, out_row_idx, out_col_idx = \
                                        lin_index(self.ambiguous_relus[x_idx-1], c_neuron, get_idx_only=True)
                                    if out_chan_idx not in non_zero_indices_sets:
                                        non_zero_indices = torch.nonzero(
                                            previous_layer.weight[out_chan_idx, :, :, :]).tolist()
                                        non_zero_indices_sets[out_chan_idx] = \
                                            set([tuple(ind_list) for ind_list in non_zero_indices])
                                    if self.ambiguous_relus[x_idx-1][out_chan_idx][out_row_idx][out_col_idx]:
                                        is_violated, violated_set, violation = self.most_violated_tighten(
                                            x_idx, layer_idx, (out_chan_idx, out_row_idx, out_col_idx),
                                            first_call=first_call)
                                        if is_violated:
                                            violation_list.append((violated_set, violation,
                                                                   (out_chan_idx, out_row_idx, out_col_idx),
                                                                   non_zero_indices_sets[out_chan_idx]))

                            violation_list = sorted(violation_list, key=lambda entry: entry[1], reverse=True)
                            for violated_set, violation, neuron_coordinates, non_zero_indices_set in violation_list:
                                if self.mode == "lp-cut" and self.applied_cuts >= self.cut_treshold:
                                    break
                                if isinstance(neuron_coordinates, int):
                                    z_var = self.gurobi_z_vars[x_idx - 1][neuron_coordinates]
                                    x_var = self.gurobi_x_vars[x_idx][neuron_coordinates]
                                else:
                                    out_chan_idx, out_row_idx, out_col_idx = neuron_coordinates
                                    z_var = self.gurobi_z_vars[x_idx-1][out_chan_idx][out_row_idx][out_col_idx]
                                    x_var = self.gurobi_x_vars[x_idx][out_chan_idx][out_row_idx][out_col_idx]
                                negative_list = non_zero_indices_set - set(violated_set)
                                tighten_expression = self.get_tighten_constraint(
                                    x_idx, layer_idx, neuron_coordinates, violated_set, negative_list, z_var)
                                if self.mode == "lp-cut":
                                    cbeta = model.addConstr(x_var <= tighten_expression)
                                    self.added_exp_constraints[self.c_neuron].append(cbeta)
                                else:
                                    # TODO: adding lazyconstraints from MIPSOL seems better but is not original Anderson
                                    #  approach
                                    model.cbCut(x_var <= tighten_expression)
                            x_idx += 1

                self.applied_cuts += 1

                # If the cutting planes are manually implemented
                if self.mode == "lp-cut":
                    if self.last_relu_index > 0 and self.applied_cuts > self.old_applied_cuts:
                        bound = self.optimize_var(self.pending_bound_var, None, allowed_statuses,
                                                  store_input=self.pending_lower_bound)
                        if self.c_neuron <= self.n_opt_neurons/2:
                            self.c_bounds[self.c_neuron] = -bound
                        else:
                            self.c_bounds[self.c_neuron] = bound
                        self.cut_optimization_calls += 1
                        if self.c_x_idx == self.store_bounds_progress:
                            self.logger.add_point(self.c_x_idx, self.c_bounds.clone())

                        continue_adding = self.applied_cuts < self.cut_treshold
                        if counterexample_verification and self.pending_lower_bound:
                            continue_adding = continue_adding and (bound < self.decision_boundary)
                        if continue_adding:
                            self.insert_tighten_cut(self.model, "lp-cut", counterexample_verification=counterexample_verification)
                        return bound

        # Handle early terminations if solving the MIP.
        if self.mode == "mip-exact":
            if where == grb.GRB.Callback.MIP:
                best_bound = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                self.global_lb_upper_bound = best_bound
                if best_bound > 0:
                    self.interrupted_sat = False  # UNSAT
                    model.terminate()

            if where == grb.GRB.Callback.MIPSOL:
                obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                if obj < 0:
                    # Does it have a chance at being a valid
                    # counter-example?

                    # Check it with the network
                    input_vals = model.cbGetSolution(self.preact_x_vars[0])

                    with torch.no_grad():
                        if isinstance(input_vals, list):
                            inps = torch.Tensor(input_vals).view(1, -1)
                        else:
                            assert isinstance(input_vals, grb.tupledict)
                            inps = torch.Tensor([val for val in input_vals.values()])
                            inps = inps.view((1,) + self.lower_bounds[0].shape)
                        out = self.net(inps).squeeze()
                        out = out.item()

                    if out < 0:
                        self.interrupted_sat = True  # SAT
                        model.terminate()

    def most_violated_tighten(self, x_idx, layer_idx, neuron_coordinates, first_call=False):
        """
        Compute the most violated constraint from the exponential family of [1] for the neuron specified by
        neuron_coordinates. Designed to be called from within the gurobi callback "insert_tighten_cut"
        :param x_idx: layer number counting hidden layers (ReLUs not counted)
        :param layer_idx: layer number as in the layer list (counts ReLUs)
        :param neuron_coordinates: indices related to the current neuron (one index for Linear, a triple for Conv2d)
        :param first_call: in the first call, it takes the current variable assignment from the collected_primals
        :return: bool indicating whether the constraint is violated, list of indices making up the I set for the most
        violated tighten constraint.
        """

        layer = self.layers[layer_idx-1]
        # Retrieve lower, upper bounds, and variables of previous layer.
        l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, layer_idx)

        if type(layer) is nn.Linear:

            neuron_idx = neuron_coordinates

            coeffs = layer.weight[neuron_idx, :]
            neg_w_indices = coeffs < 0
            l_breve_km1 = l_km1.clone()
            l_breve_km1[neg_w_indices] = u_km1[neg_w_indices]
            u_breve_km1 = u_km1.clone()
            u_breve_km1[neg_w_indices] = l_km1[neg_w_indices]
            if self.mode == "lp-cut":
                if first_call:
                    x_km1_values = self.x_solution[x_idx - 1][self.c_neuron]
                    if x_km1_values.dim() > 1:
                        x_km1_values = x_km1_values.view(-1)
                    z_value = self.z_solution[x_idx - 1][self.c_neuron][neuron_idx]
                else:
                    x_km1_values = torch.tensor([c_x_km1.X for c_x_km1 in x_km1_vars], dtype=torch.float,
                                                device=self.device)
                    z_value = self.gurobi_z_vars[x_idx - 1][neuron_idx].X
            else:
                z_value = self.model.cbGetNodeRel(self.gurobi_z_vars[x_idx-1][neuron_idx])
                x_km1_values = torch.tensor([self.model.cbGetNodeRel(c_x_km1) for c_x_km1 in x_km1_vars],
                                            dtype=torch.float, device=self.device)

            # Linear oracle from [1] to obtain the most violated set for a neuron.
            left_vector = coeffs * x_km1_values
            right_vector = coeffs * (l_breve_km1*(1 - z_value) + u_breve_km1*z_value)
            most_violated_indices_mask = left_vector - right_vector < 1e-3
            I_mask = torch.zeros(coeffs.shape, dtype=torch.float, device=self.device)
            I_mask[most_violated_indices_mask] = 1

            # Compute if the constraint corresponding to most_violated_indices is violated and by how much.
            if self.mode == "lp-cut":
                if first_call:
                    y_value = self.x_solution[x_idx][self.c_neuron][neuron_idx]
                else:
                    y_value = self.gurobi_x_vars[x_idx][neuron_idx].X
            else:
                y_value = self.model.cbGetNodeRel(self.gurobi_x_vars[x_idx][neuron_idx])
            constraint_value = layer.bias[neuron_idx]*z_value + torch.dot(coeffs * I_mask, x_km1_values - l_breve_km1*(1 - z_value)) + \
                               torch.dot(coeffs * (1 - I_mask), u_breve_km1 * z_value)

            # Convert the mask to a list of indices.
            most_violated_indices = torch.nonzero(I_mask).flatten().tolist()

        elif type(layer) is nn.Conv2d:
            out_chan_idx = neuron_coordinates[0]
            out_row_idx = neuron_coordinates[1]
            out_col_idx = neuron_coordinates[2]
            if self.mode == "lp-cut":
                if first_call:
                    z_value = self.z_solution[x_idx - 1][self.c_neuron][out_chan_idx][out_row_idx][out_col_idx]
                    y_value = self.x_solution[x_idx][self.c_neuron][out_chan_idx][out_row_idx][out_col_idx]
                    x_km1_values = self.x_solution[x_idx-1][self.c_neuron]
                else:
                    z_value = self.gurobi_z_vars[x_idx - 1][out_chan_idx][out_row_idx][out_col_idx].X
                    y_value = self.gurobi_x_vars[x_idx][out_chan_idx][out_row_idx][out_col_idx].X
                    x_km1_values = torch.tensor([[[cx.X for cx in cx_col] for cx_col in cx_row] for cx_row in x_km1_vars],
                                                dtype=torch.float, device=self.device)
            else:
                z_value = self.model.cbGetNodeRel(self.gurobi_z_vars[x_idx-1][out_chan_idx][out_row_idx][out_col_idx])
                y_value = self.model.cbGetNodeRel(self.gurobi_x_vars[x_idx][out_chan_idx][out_row_idx][out_col_idx])
                x_km1_values = torch.tensor([[[self.model.cbGetNodeRel(cx) for cx in cx_col] for cx_col in cx_row]
                                             for cx_row in x_km1_vars], dtype=torch.float, device=self.device)
            # No need to use tensors as getting x_km1_values is the bottleneck anyways.

            # Linear oracle from [1] to obtain the most violated set for a neuron.
            most_violated_indices = []  # List of index lists, representing the most violated set I.
            for in_chan_idx in range(layer.weight.shape[1]):
                for ker_row_idx in range(layer.weight.shape[2]):
                    in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                    if (in_row_idx < 0) or (in_row_idx >= len(x_km1_values[0])):
                        # This is padding -> value of 0
                        continue
                    for ker_col_idx in range(layer.weight.shape[3]):
                        in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                        if (in_col_idx < 0) or (in_col_idx >= len(x_km1_values[0][0])):
                            # This is padding -> value of 0
                            continue
                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                        if abs(coeff) > 1e-6:
                            c_l_breve_km1 = l_km1[0, in_chan_idx, in_row_idx, in_col_idx].item() if coeff >= 0 \
                                else u_km1[0, in_chan_idx, in_row_idx, in_col_idx].item()
                            c_u_breve_km1 = u_km1[0, in_chan_idx, in_row_idx, in_col_idx].item() if coeff >= 0 \
                                else l_km1[0, in_chan_idx, in_row_idx, in_col_idx].item()
                            if self.mode == "lp-cut":
                                left_coeff = coeff * x_km1_values[in_chan_idx][in_row_idx][in_col_idx]
                            else:
                                left_coeff = coeff * self.model.cbGetNodeRel(x_km1_vars[in_chan_idx][in_row_idx][in_col_idx])
                            right_coeff = coeff * (c_l_breve_km1*(1 - z_value) + c_u_breve_km1*z_value)
                            if left_coeff - right_coeff < 1e-3:
                                most_violated_indices.append((in_chan_idx, ker_row_idx, ker_col_idx))

            # Compute the extent to which the constraint is violated.
            non_zero_indices = torch.nonzero(layer.weight[out_chan_idx, :, :, :]).tolist()  # contains list of index tuples corresponding to nonzero entries in the conv filter
            non_zero_indices_set = set([tuple(ind_list) for ind_list in non_zero_indices])
            negative_list = non_zero_indices_set - set(most_violated_indices)
            constraint_value = layer.bias[out_chan_idx].item() * z_value
            for j_list in most_violated_indices:
                in_chan_idx = j_list[0]
                ker_row_idx = j_list[1]
                ker_col_idx = j_list[2]
                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                if (in_row_idx < 0) or (in_row_idx >= l_km1.size(2)) or (in_col_idx < 0) or (
                        in_col_idx >= l_km1.size(3)):
                    # This is padding -> value of 0
                    continue
                coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                c_l_breve_km1 = l_km1[0, in_chan_idx, in_row_idx, in_col_idx].item() if coeff >= 0 \
                    else u_km1[0, in_chan_idx, in_row_idx, in_col_idx].item()
                if abs(coeff) > 1e-6:
                    if self.mode == "lp-cut":
                        c_x = x_km1_values[in_chan_idx][in_row_idx][in_col_idx]
                    else:
                        c_x = self.model.cbGetNodeRel(x_km1_vars[in_chan_idx][in_row_idx][in_col_idx])
                    constraint_value += coeff * (c_x - c_l_breve_km1 * (1 - z_value))

            for in_chan_idx, ker_row_idx, ker_col_idx in negative_list:
                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                if (in_row_idx < 0) or (in_row_idx >= l_km1.size(2)) or (in_col_idx < 0) or (
                        in_col_idx >= l_km1.size(3)):
                    # This is padding -> value of 0
                    continue
                coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                c_u_breve_km1 = u_km1[0, in_chan_idx, in_row_idx, in_col_idx].item() if coeff >= 0 \
                    else l_km1[0, in_chan_idx, in_row_idx, in_col_idx].item()
                if abs(coeff) > 1e-6:
                    constraint_value += coeff * c_u_breve_km1 * z_value

        else:
            raise ValueError(f"most_violated_tighten not implemented for {type(layer)} layer")

        # Full and empty index list constraints are already included (but this shouldn't change anything).
        is_violated = y_value - constraint_value > 1e-3 and len(most_violated_indices) > 0 and len(most_violated_indices) < l_km1.nelement()

        return is_violated, most_violated_indices, y_value - constraint_value

    def get_tighten_constraint(self, x_idx, layer_idx, neuron_coordinates, index_list, negative_list, z_var):
        """
        Return a Gurobi linear expression for the current tighten constraint defined by index_list (and its complement
        negative_list).
        :param x_idx: layer number counting hidden layers (ReLUs not counted)
        :param layer_idx: layer number as in the layer list (counts ReLUs)
        :param neuron_coordinates: indices related to the current neuron (one index for Linear, a triple for Conv2d)
        :param index_list: list of indices making up the I set for the tighten constraint
        :param negative_list: list of (non-zero) indices not in the I set for the tighten constraint
        :param z_var: the Gurobi z variable to use in the constraint
        :return: Gurobi linear expression for the current tighten constraint
        """

        layer = self.layers[layer_idx - 1]
        # Retrieve lower, upper bounds, and variables of previous layer.
        l_km1, u_km1, x_km1_vars = self.get_previous_layer_info(x_idx, layer_idx)
        tighten_expression = 0

        if type(layer) is nn.Linear:

            neuron_idx = neuron_coordinates

            for j in index_list:
                c_l_breve_km1 = l_km1[j].item() if layer.weight[neuron_idx, j] >= 0 else u_km1[j].item()
                tighten_expression += layer.weight[neuron_idx, j].item() * (x_km1_vars[j] - c_l_breve_km1 * (1 - z_var))
            u_breve_sum = 0
            for j in negative_list:
                c_u_breve_km1 = u_km1[j].item() if layer.weight[neuron_idx, j] >= 0 else l_km1[j].item()
                u_breve_sum += layer.weight[neuron_idx, j].item() * c_u_breve_km1
            tighten_expression += z_var * (layer.bias[neuron_idx].item() + u_breve_sum)

        elif type(layer) is nn.Conv2d:

            out_chan_idx = neuron_coordinates[0]
            out_row_idx = neuron_coordinates[1]
            out_col_idx = neuron_coordinates[2]

            for j_list in index_list:
                in_chan_idx = j_list[0]
                ker_row_idx = j_list[1]
                ker_col_idx = j_list[2]
                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                if (in_row_idx < 0) or (in_row_idx >= l_km1.size(2)) or (in_col_idx < 0) or (
                        in_col_idx >= l_km1.size(3)):
                    # This is padding -> value of 0
                    continue
                coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                c_l_breve_km1 = l_km1[0, in_chan_idx, in_row_idx, in_col_idx] if coeff >= 0 else u_km1[
                    0, in_chan_idx, in_row_idx, in_col_idx]
                if abs(coeff) > 1e-6:
                    tighten_expression += coeff * (
                                x_km1_vars[in_chan_idx][in_row_idx][in_col_idx] - c_l_breve_km1.item() * (1 - z_var))

            u_breve_sum = 0
            for in_chan_idx, ker_row_idx, ker_col_idx in negative_list:
                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                if (in_row_idx < 0) or (in_row_idx >= l_km1.size(2)) or (in_col_idx < 0) or (
                        in_col_idx >= l_km1.size(3)):
                    # This is padding -> value of 0
                    continue
                coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                c_u_breve_km1 = u_km1[0, in_chan_idx, in_row_idx, in_col_idx] if coeff >= 0 else l_km1[
                    0, in_chan_idx, in_row_idx, in_col_idx]
                if abs(coeff) > 1e-6:
                    u_breve_sum += coeff * c_u_breve_km1.item()

            tighten_expression += z_var * (layer.bias[out_chan_idx].item() + u_breve_sum)

        return tighten_expression

    def collect_primal_solution(self, x_solution, z_solution):
        """
        Retrieve the primal solution tensors from Gurobi's variables. Assumes that a solution has just been computed.
        The solution is added (stacked) on self.x_solution and self.z_solution, which are lists of tensors of shape
        2 * layer_width x layer_shape
        """
        for idx, c_x_vars in enumerate(self.gurobi_x_vars):
            if type(c_x_vars[0]) is list:  # convolutional layer
                c_tensor = torch.tensor([[[cx.X for cx in cx_col] for cx_col in cx_row] for cx_row in c_x_vars],
                                        dtype=torch.float, device=self.device).unsqueeze(0)
            else:  # linear layer
                c_tensor = torch.tensor([cx.X for cx in c_x_vars], dtype=torch.float, device=self.device).unsqueeze(0)

            if x_solution[idx] is None:
                x_solution[idx] = c_tensor
            else:
                x_solution[idx] = torch.cat([x_solution[idx], c_tensor], 0)

        for idx, c_z_vars in enumerate(self.gurobi_z_vars):
            if type(c_z_vars[0]) is list:
                # convolutional layer
                c_tensor = torch.tensor([[[cz.X for cz in cz_col] for cz_col in cz_row] for cz_row in c_z_vars],
                                        dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                # linear layer
                c_tensor = torch.tensor([cz.X for cz in c_z_vars], dtype=torch.float, device=self.device).unsqueeze(0)
            if z_solution[idx] is None:
                z_solution[idx] = c_tensor
            else:
                z_solution[idx] = torch.cat([z_solution[idx], c_tensor], 0)

    @staticmethod
    def combine_primal_solutions(x_solution_ub, x_solution_lb, z_solution_ub, z_solution_lb):
        """
        Merge the primal solutions related to ub and lb computations into a single tensor in the format of explp
        :return: lists of tensors (x, z) for the combined solution
        """
        if x_solution_ub[0] is None:
            if x_solution_lb[0] is None:
                return x_solution_ub, z_solution_ub
            x_solution_ub = [float("inf") * torch.ones_like(x_solution_lb[x_idx]) for x_idx in range(len(x_solution_ub))]
            z_solution_ub = [float("inf") * torch.ones_like(z_solution_lb[x_idx]) for x_idx in range(len(z_solution_ub))]
        if x_solution_lb[0] is None:
            x_solution_lb = [float("inf") * torch.ones_like(x_solution_ub[x_idx]) for x_idx in range(len(x_solution_lb))]
            z_solution_lb = [float("inf") * torch.ones_like(z_solution_ub[x_idx]) for x_idx in range(len(z_solution_lb))]

        return [torch.cat([x_solution_ub[x_idx], x_solution_lb[x_idx]], 0) for x_idx in range(len(x_solution_ub))],\
               [torch.cat([z_solution_ub[x_idx], z_solution_lb[x_idx]], 0) for x_idx in range(len(z_solution_ub))]

    def get_input_list(self):
        inp_size = self.lower_bounds[0].size()
        mini_inp = torch.zeros_like(self.lower_bounds[0])

        if len(inp_size) == 1:
            # This is a linear input.
            for i in range(inp_size[0]):
                if self.mode != "mip-exact":
                    mini_inp[i] = self.gurobi_x_vars[0][i].x
                else:
                    mini_inp[i] = self.model.cbGetNodeRel(self.gurobi_x_vars[0][i])

        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        if self.mode != "mip-exact":
                            mini_inp[i, j, k] = self.gurobi_x_vars[0][i][j][k].x
                        else:
                            mini_inp[i, j, k] = self.model.cbGetNodeRel(self.gurobi_x_vars[0][i][j][k])
        return mini_inp.unsqueeze(0)

    def compute_crown_intercept(self, x_idx):
        # Compute CROWN bounds at layer x_idx (both lower and upper bounds) and return intercept scores at layer x_idx.
        # This is executed on CPU.
        l0_net = SaddleLP(self.layers, store_bounds_primal=True)
        l0_net.set_decomposition('pairs', 'crown')
        l0_net.set_solution_optimizer('init', None)
        domain = torch.stack([self.lower_bounds[0], self.upper_bounds[0]], dim=-1).unsqueeze(0)
        l0_net.build_model_using_bounds(
            domain, ([self.lower_bounds[0]] + [clbs.unsqueeze(0) for clbs in self.pre_lower_bounds],
                     [self.upper_bounds[0]] + [cubs.unsqueeze(0) for cubs in self.pre_upper_bounds]), build_limit=x_idx)
        _, _ = l0_net.compute_lower_bound(node=(x_idx, None))
        return l0_net.get_l0_intercept_scores()

    @staticmethod
    def get_tight_neuron_set(c_neuron, k, intercept_scores, intermediate_lbs, intermediate_ubs):
        """
        As the rest of Gurobi-based code, assumes the domain batch_size is 1.
        Computes the set of neurons for which we want to use a tighter ReLU relaxation (Anderson, L1, etc).
        """
        # Compute L1 sets from intercept scores.
        L1_set = [[] for lay in range(len(intermediate_lbs))]
        for lay in range(1, len(intermediate_lbs) - 1):
            clist = []
            cscores = intercept_scores[lay][0][c_neuron]
            ck = min(len(cscores), k)
            for cidx in torch.topk(cscores, ck)[1].tolist():
                if intermediate_lbs[lay].view(-1)[cidx] <= 0 and intermediate_ubs[lay].view(-1)[cidx] >= 0:
                    clist.append(cidx)
            L1_set[lay].extend(clist)
        return L1_set


class DualAndersonGurobiVar:
    """
    Class storing the dual variables associated to a single neuron of the network.
    """

    def __init__(self, alpha_0, alpha_1, beta, gamma_l, gamma_u, delta_0, delta_1):
        """
        Class constructor. Any dual variable can be None if the constraint is not enforced (if, for instance, the ReLU
        is blocking).
        All the variables are represented as constraints. The dual variable value is retrieved by calling constr.Pi.
        For MILP, constr.Pi is not available https://www.gurobi.com/documentation/8.1/refman/matlab_gurobi.html

        :param alpha_0: lagrangian multiplier (LM) for non-negativity constraint if the ReLU is ambiguous, otherwise
         it is associated to the equality constraint x=0 or x=Wx + b. Single Gurobi constraint.
        :param alpha_1: LM for greater than Wx + b, single Gurobi constraint
        :param beta: LM for the exponentially many constraints, list of Gurobi constraint
        :param gamma_l: LM for x lower bound, single Gurobi constraint
        :param gamma_u: LM for x upper bound, single Gurobi constraint
        :param delta_0: LM for z lower bound, single Gurobi constraint
        :param delta_1: LM for z upper bound, single Gurobi constraint
        """

        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.beta = beta
        self.gamma_l = gamma_l
        self.gamma_u = gamma_u
        self.delta_0 = delta_0
        self.delta_1 = delta_1


def lin_index(input, idx, get_idx_only=False):
    # Given 3D input from convolutional layer, return its entry indexed with respect to its linearized version.
    # if get_idx_only, return index as tuple and not entry
    if isinstance(input[0], list):
        conv_shape = (len(input), len(input[0]), len(input[0][0]))
        out_chan_idx = idx // prod(conv_shape[1:])
        out_row_idx = (idx % prod(conv_shape[1:])) // conv_shape[2]
        out_col_idx = (idx % prod(conv_shape[1:])) % conv_shape[2]
        if not get_idx_only:
            return input[out_chan_idx][out_row_idx][out_col_idx]
        else:
            return out_chan_idx, out_row_idx, out_col_idx
    else:
        if not get_idx_only:
            return input[idx]
        else:
            return idx
import torch
from torch import nn
from plnn.dual_bounding import DualBounding
from plnn.proxlp_solver import utils
from plnn.branch_and_bound.utils import ParentInit


default_params = {
    'nb_steps': 5,
    'initial_step_size': 1e0,
    'step_size_decay': 0.98,
    'betas': (0.9, 0.999),
    'best_among': None,  # Used only if type is "best_prop"
    'joint_ib': False  # Whether to do joint optimization over IBs.
}


class Propagation(DualBounding):
    """
    Class implementing propagation-based bounding in the dual space.
    NOTE: does not support being last layer bound within BaB.
    """
    def __init__(self, layers, params=None, type="alpha-crown", store_bounds_primal=False, max_batch=2000):
        """
        :param type: which propagation-based bounding to use. Options available: ['naive', 'KW', 'crown']
        """
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.max_batch = max_batch
        self.params = dict(default_params, **params) if params is not None else default_params
        self.store_bounds_primal = store_bounds_primal
        self.bounds_primal = None
        self.external_init = None

        assert type in ["naive", "KW", "crown", "best_prop", "alpha-crown", "gamma-crown", "beta-crown"]
        if type == "best_prop":
            assert self.params["best_among"] is not None, "Must provide list of prop types to choose best from"
            self.optimize = self.best_prop_optimizers(self.params["best_among"])
        else:
            self.type = type
            self.optimize = self.propagation_optimizer

    def propagation_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):

        add_coeff = next(iter(additional_coeffs.values()))
        if self.type == "naive" or len(weights) == 1:
            # 'naive'
            lay_n = len(weights) - 1
            xn_coeff = weights[lay_n].backward(add_coeff)

            argmin_ibs = torch.where(xn_coeff >= 0, lower_bounds[lay_n].unsqueeze(1), upper_bounds[lay_n].unsqueeze(1))
            # The bounds of layers after the first need clamping (as these are post-activation bounds)
            if lay_n > 0:
                argmin_ibs = argmin_ibs.clamp(0, None)

            bounding_out = utils.bdot(argmin_ibs, xn_coeff)
            bounding_out += utils.bdot(add_coeff, weights[-1].get_bias())

        elif self.type in ["crown", "KW"]:
            # Compute dual variables.
            dual_vars = PropDualVars.get_duals_from(
                weights, additional_coeffs, lower_bounds, upper_bounds, init_type=self.type)

            # Compute objective.
            bounding_out = self.compute_bounds(weights, add_coeff, dual_vars, lower_bounds, upper_bounds)

        elif self.type in ["alpha-crown", "gamma-crown", "beta-crown"]:
            if self.params["joint_ib"] and add_coeff.shape[1] != 1:
                raise ValueError("Can do joint IB optimization only in case of single output NN, w/o UB computations")

            if self.external_init is not None and type(self.external_init) is PropInit and self.params["nb_steps"] > 0:
                # Take the dual variables we optimize over from parent.
                alphas, gammas, betas = self.external_init.alphas, self.external_init.gammas, self.external_init.betas
                # Handle alpha-crown initialization of gamma-crown.
                if gammas is None and self.type == "gamma-crown":
                    gammas = {"l": [], "u": []}
                    for lay_idx in range(len(alphas)):
                        gammas["l"].append(torch.zeros_like(alphas[lay_idx]))
                        gammas["u"].append(torch.zeros_like(alphas[lay_idx]))
                # Handle alpha-crown initialization of beta-crown.
                if betas is None and self.type == "beta-crown":
                    betas = []
                    for lay_idx in range(len(alphas)):
                        betas.append(torch.zeros_like(alphas[lay_idx]))
            else:
                # Assign crown lb_slope to alphas, and gammas if doing gamma-crown
                alphas = []
                gammas = {"l": [], "u": []} if self.type == "gamma-crown" else None
                betas = [] if self.type == "beta-crown" else None
                for lay_idx, (lbs, ubs) in enumerate(zip(lower_bounds, upper_bounds)):
                    if lay_idx > 0 and lay_idx < len(weights):
                        neuron_batch_size = add_coeff.shape[1]
                        crown_lb_slope = (ubs >= torch.abs(lbs)).type(lbs.dtype).unsqueeze(1).repeat(
                            (1, neuron_batch_size) + ((1,) * (lbs.dim() - 1)))
                        alphas.append(crown_lb_slope)
                        if self.type == "gamma-crown":
                            gammas["l"].append(torch.zeros_like(crown_lb_slope))
                            gammas["u"].append(torch.zeros_like(crown_lb_slope))
                        elif self.type == "beta-crown":
                            betas.append(torch.zeros_like(crown_lb_slope))

            # TODO: can't reproduce Beta-CROWN results for joint IB within BaB. Probably need to exclude non-ambiguous
            #  bounds from the computation
            if self.params["joint_ib"]:
                # Define two parameter blocks per layer (one for LBs, the other for UBs).
                # Dict of lists of alphas and betas for each LB/UB computation. Key format: "<LB/UB>-<x_idx>"
                alphas_ib = {}
                betas_ib = {}
                # Dict of dict of lists of gammas for each LB/UB computation. Key format: "<LB/UB>-<x_idx>"
                gammas_ib = {}
                for lay_idx in range(2, len(weights)):
                    alphas_ib[f"LB-{lay_idx}"] = [alpha.clone() for alpha in alphas[:lay_idx-1]]
                    alphas_ib[f"UB-{lay_idx}"] = [alpha.clone() for alpha in alphas[:lay_idx-1]]
                    betas_ib[f"LB-{lay_idx}"] = None
                    gammas_ib[f"LB-{lay_idx}"], gammas_ib[f"UB-{lay_idx}"] = None, None
                    if self.type == "gamma-crown":
                        gammas_ib[f"LB-{lay_idx}"] = {["l"]: [gamma.clone() for gamma in gammas["l"][:lay_idx-1]],
                                                      ["u"]: [gamma.clone() for gamma in gammas["u"][:lay_idx-1]]}
                        gammas_ib[f"UB-{lay_idx}"] = {["l"]: [gamma.clone() for gamma in gammas["l"][:lay_idx-1]],
                                                      ["u"]: [gamma.clone() for gamma in gammas["u"][:lay_idx-1]]}
                    elif self.type == "beta-crown":
                        betas_ib[f"LB-{lay_idx}"] = [beta.clone() for beta in betas[:lay_idx-1]]
                        betas_ib[f"UB-{lay_idx}"] = [beta.clone() for beta in betas[:lay_idx-1]]

                def obj(alphas, alphas_ib, gammas=None, gammas_ib=None, betas=None, betas_ib=None,
                        store_bounds_primal=False):

                    joint_lbs, joint_ubs = [lower_bounds[0], lower_bounds[1]], [upper_bounds[0], upper_bounds[1]]
                    for lay_idx in range(2, len(weights)):
                        # With this implementation one has to store batches of mus and lambdas as when computing IBs
                        # normally, not sure there is an efficient solution.
                        # Quite expensive, compare to optimizing them the normal way
                        layer_shape = lower_bounds[lay_idx].shape[1:]
                        layer_prod = utils.prod(layer_shape)

                        # LB computation
                        ib_add_coeffs = {lay_idx: torch.eye(layer_prod,  device=alphas[0].device).
                            view((layer_prod, *layer_shape)).
                            expand((lower_bounds[lay_idx].shape[0], layer_prod, *layer_shape))}
                        prop_vars = PropDualVars.get_duals_from(
                            weights[:lay_idx], ib_add_coeffs, joint_lbs[:lay_idx], joint_ubs[:lay_idx],
                            alphas=alphas_ib[f"LB-{lay_idx}"], gammas=gammas_ib[f"LB-{lay_idx}"],
                            betas=betas_ib[f"LB-{lay_idx}"])
                        clbs = self.compute_bounds(
                            weights[:lay_idx], ib_add_coeffs[lay_idx], prop_vars, joint_lbs[:lay_idx],
                            joint_ubs[:lay_idx], gammas=gammas_ib[f"LB-{lay_idx}"], store_primal=store_bounds_primal)
                        joint_lbs.append(clbs.view_as(lower_bounds[lay_idx]))
                        # UB computation
                        ib_add_coeffs = {lay_idx: -torch.eye(layer_prod,  device=alphas[0].device).
                            view((layer_prod, *layer_shape)).
                            expand((lower_bounds[lay_idx].shape[0], layer_prod, *layer_shape))}
                        prop_vars = PropDualVars.get_duals_from(
                            weights[:lay_idx], ib_add_coeffs, joint_lbs[:lay_idx], joint_ubs[:lay_idx],
                            alphas=alphas_ib[f"UB-{lay_idx}"], gammas=gammas_ib[f"UB-{lay_idx}"],
                            betas=betas_ib[f"UB-{lay_idx}"])
                        cubs = self.compute_bounds(
                            weights[:lay_idx], ib_add_coeffs[lay_idx], prop_vars, joint_lbs[:lay_idx],
                            joint_ubs[:lay_idx], gammas=gammas_ib[f"UB-{lay_idx}"], store_primal=store_bounds_primal)
                        joint_ubs.append(-cubs.view_as(upper_bounds[lay_idx]))

                    prop_vars = PropDualVars.get_duals_from(
                        weights, additional_coeffs, joint_lbs, joint_ubs, alphas=alphas, gammas=gammas, betas=betas)
                    bound = self.compute_bounds(weights, add_coeff, prop_vars, joint_lbs, joint_ubs,
                                                gammas=gammas, store_primal=store_bounds_primal)
                    return bound, joint_lbs, joint_ubs
            else:
                alphas_ib, gammas_ib, betas_ib = None, None, None

                # define objective function
                def obj(alphas, gammas=None, betas=None, store_bounds_primal=False):
                    prop_vars = PropDualVars.get_duals_from(
                        weights, additional_coeffs, lower_bounds, upper_bounds, alphas=alphas, gammas=gammas,
                        betas=betas)
                    bound = self.compute_bounds(weights, add_coeff, prop_vars, lower_bounds, upper_bounds,
                                                gammas=gammas, store_primal=store_bounds_primal)
                    return bound

            with torch.enable_grad():

                # Mark which variables we are optimizing over...
                gammas_list = gammas["l"] + gammas["u"] if self.type == "gamma-crown" else []
                betas_list = betas if self.type == "beta-crown" else []
                optvars = alphas + gammas_list + betas_list
                for cvar in optvars:
                    cvar.requires_grad = True
                if self.params["joint_ib"]:
                    for key in alphas_ib:
                        for cvar in alphas_ib[key]:
                            cvar.requires_grad = True
                    if self.type == "gamma-crown":
                        for key in gammas_ib:
                            for cvar in gammas_ib[key]["l"] + gammas_ib[key]["u"]:
                                cvar.requires_grad = True
                    elif self.type == "beta-crown":
                        for key in betas_ib:
                            for cvar in betas_ib[key]:
                                cvar.requires_grad = True

                # ...and pass them as a list to the optimizer.
                if self.params["joint_ib"]:
                    for key in alphas_ib:
                        optvars += alphas_ib[key]
                    if self.type == "gamma-crown":
                        for key in gammas_ib:
                            optvars += gammas_ib[key]["l"] + gammas_ib[key]["u"]
                    elif self.type == "beta-crown":
                        for key in betas_ib:
                            optvars += betas_ib[key]
                optimizer = torch.optim.Adam(optvars, lr=self.params["initial_step_size"], betas=self.params["betas"])
                # Decay step size.
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.params["step_size_decay"])

                # do autograd-adam
                for step in range(self.params["nb_steps"]):
                    optimizer.zero_grad()
                    if not self.params["joint_ib"]:
                        obj_value = -obj(alphas, gammas=gammas, betas=betas)
                    else:
                        obj_value = -obj(alphas, alphas_ib, gammas=gammas, gammas_ib=gammas_ib, betas=betas,
                                         betas_ib=betas_ib)[0]
                    obj_value.mean().backward()
                    optimizer.step()
                    scheduler.step()

                    # Project into [0,1], R+
                    with torch.no_grad():
                        for alpha in alphas:
                            alpha.clamp_(0, 1)
                        for gamma in gammas_list:
                            gamma.clamp_(0, None)
                        for beta in betas_list:
                            beta.clamp_(0, None)
                        if self.params["joint_ib"]:
                            for key in alphas_ib:
                                for alpha in alphas_ib[key]:
                                    alpha.clamp_(0, 1)
                            if self.type == "gamma-crown":
                                for key in gammas_ib:
                                    for gamma in gammas_ib[key]["l"] + gammas_ib[key]["u"]:
                                        gamma.clamp_(0, None)
                            elif self.type == "beta-crown":
                                for key in betas_ib:
                                    for beta in betas_ib[key]:
                                        beta.clamp_(0, 1)

                # Detach variables before returning them.
                alphas = [alpha.detach() for alpha in alphas]
                if self.type == "gamma-crown":
                    for key in gammas:
                        gammas[key] = [gamma.detach() for gamma in gammas[key]]
                elif self.type == "beta-crown":
                    betas = [beta.detach() for beta in betas]
                if self.params["joint_ib"]:
                    for key in alphas_ib:
                        alphas_ib[key] = [alpha.detach() for alpha in alphas_ib[key]]
                    if self.type == "gamma-crown":
                        for key in gammas_ib:
                            for inner_key in gammas_ib[key]:
                                gammas_ib[key][inner_key] = [gamma.detach() for gamma in gammas_ib[key][inner_key]]
                    elif self.type == "beta-crown":
                        for key in betas_ib:
                            betas_ib[key] = [beta.detach() for beta in betas_ib[key]]

                # store last dual solution for future usage
                self.children_init = PropInit(alphas, parent_gammas=gammas, parent_betas=betas)

                # End of the optimization
                if self.params["joint_ib"]:
                    bounding_out = obj(alphas, alphas_ib, gammas=gammas, gammas_ib=gammas_ib, betas=betas,
                                       betas_ib=betas_ib, store_bounds_primal=self.store_bounds_primal)
                else:
                    bounding_out = obj(alphas, gammas=gammas, betas=betas, store_bounds_primal=self.store_bounds_primal)

        return bounding_out

    def best_prop_optimizers(self, best_among):
        # We return the best amongst the "best_among' types of propagation-based methods
        c_fun = self.propagation_optimizer

        def optimize(*args, **kwargs):
            self.type = best_among[0]
            best_bounds = c_fun(*args, **kwargs)
            for method in best_among[1:]:
                self.type = method
                c_bounds = c_fun(*args, **kwargs)
                best_bounds = torch.max(c_bounds, best_bounds)
            return best_bounds

        return optimize

    def get_closedform_lastlayer_lb_duals(self):
        # Returns dual variables (instance of dj_relaxation.DualVarSet) corresponding to closed-form bounding
        # computations for the lower bounds of the network output. Assumes the network has only one output.
        weights = self.weights
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds
        # Create additional_coeffs corresponding to the last layer lower bounding (the batch must fit in memory)
        additional_coeffs = {len(weights): torch.ones_like(lower_bounds[-1].unsqueeze(1))}

        if self.type in ["crown", "KW"]:
            # Compute dual variables.
            dual_vars = PropDualVars.get_duals_from(
                weights, additional_coeffs, lower_bounds, upper_bounds, init_type=self.type)
        elif "-crown" in self.type:
            # Compute dual variables using external init.
            assert self.external_init is not None
            alphas, gammas, betas = self.external_init.alphas, self.external_init.gammas, self.external_init.betas
            dual_vars = PropDualVars.get_duals_from(
                weights, additional_coeffs, lower_bounds, upper_bounds, init_type=self.type,
                alphas=alphas, gammas=gammas, betas=betas)
        else:
            raise ValueError("get_closedform_duals callable only for crown or KW bounds.")
        return dual_vars

    def compute_bounds(self, weights, add_coeff, dual_vars, lower_bounds, upper_bounds, gammas=None,
                       store_primal=False):
        """
        Compute the value of the (batch of) network bound in the propagation-based formulation, corresponding to
         eq. (5) of https://arxiv.org/abs/2104.06718.
        Given the network layers, pre-activation bounds as lists of tensors, and dual variables
        (and functions thereof) as PropDualVars.
        :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
        upper bound of each neuron, the second the lower bound.
        """
        x0_coeff = -weights[0].backward(dual_vars.mus[0])
        x0 = torch.where(x0_coeff >= 0, lower_bounds[0].unsqueeze(1), upper_bounds[0].unsqueeze(1))
        bound = utils.bdot(x0, x0_coeff)
        if store_primal:
            self.bounds_primal = x0
        else:
            del x0
        del x0_coeff

        for lay_idx in range(1, len(weights)):
            lbs = lower_bounds[lay_idx].unsqueeze(1).clamp(None, 0)
            ubs = upper_bounds[lay_idx].unsqueeze(1).clamp(0, None)
            neg_bias = ((lbs * ubs) / (ubs - lbs))
            neg_bias.masked_fill_(ubs == lbs, 0)  # cover case in which ubs & lbs coincide
            bound += utils.bdot(dual_vars.lambdas[lay_idx - 1].clamp(0, None), neg_bias)
            bound -= utils.bdot(dual_vars.mus[lay_idx - 1], weights[lay_idx - 1].get_bias())

            if gammas is not None:
                bound += utils.bdot(gammas["l"][lay_idx - 1], lower_bounds[lay_idx].unsqueeze(1))
                bound -= utils.bdot(gammas["u"][lay_idx - 1], upper_bounds[lay_idx].unsqueeze(1))

        bound += utils.bdot(add_coeff, weights[-1].get_bias())
        return bound

    def get_lower_bound_network_input(self):
        """
        Return the input of the network that was used in the last bounds computation.
        Converts back from the conditioned input domain to the original one.
        Assumes that the last layer is a single neuron.
        """
        assert self.store_bounds_primal
        assert self.bounds_primal.shape[1] in [1, 2], "the last layer must have a single neuron"
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        net_input = (1/2) * (u_0 - l_0) * self.bounds_primal.select(1, self.bounds_primal.shape[1]-1) +\
                    (1/2) * (u_0 + l_0)
        return utils.apply_transforms(self.input_transforms, net_input, inverse=True)

    def initialize_from(self, external_init):
        # setter to initialise from an external list of dual/primal variables (instance of PropInit)
        self.external_init = external_init

    def internal_init(self):
        self.external_init = None

    # BaB-related method to implement automatic no. of iters.
    def set_iters(self, iters):
        self.params["nb_steps"] = iters
        self.steps = iters

    # BaB-related method to implement automatic no. of iters.
    def default_iters(self, set_min=False):
        # Set no. of iters to default for the algorithm this class represents.
        if set_min and self.get_iters() != -1:
            self.min_steps = self.get_iters()
        else:
            self.min_steps = 0
        self.max_steps = 100
        self.step_increase = 5
        self.set_iters(self.min_steps)

    # BaB-related method to implement automatic no. of iters.
    def get_iters(self):
        return self.params["nb_steps"]


def handle_propagation_add_coeff(weights, additional_coeffs, lower_bounds):
    # Go backwards and set to 0 all mus that are after additional coefficients.
    # Returns list of zero mus, first non-zero mu, index of additional coeffs.
    mus = []
    final_lay_idx = len(weights)
    if final_lay_idx in additional_coeffs:
        # There is a coefficient on the output of the network
        mu = -additional_coeffs[final_lay_idx]
        lay_idx = final_lay_idx
    else:
        # There is none. Just identify the shape from the additional coeffs
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        device = lower_bounds[-1].device

        lay_idx = final_lay_idx - 1
        while lay_idx not in additional_coeffs:
            lay_shape = lower_bounds[lay_idx].shape[1:]
            mus.append(torch.zeros((*batch_size,) + lay_shape,
                                    device=device))
            lay_idx -= 1
        # We now reached the time where lay_idx has an additional coefficient
        mu = -additional_coeffs[lay_idx]
        mus.append(torch.zeros_like(mu))
    lay_idx -= 1
    return mus, mu, lay_idx


class PropDualVars:
    """
    Class defining dual variables for the Propagation-based formulation (lambda, mu). They might be a function of other
    variables over which to optimize (alpha in the auto-LiRPA formulation, gamma in gamma-CROWN...)
    If the ub/lb slopes are not optimized over, they are set to CROWN's (or KW, depending on init_type).
    """
    def __init__(self, lambdas, mus):
        self.lambdas = lambdas  # from relu 0 to n-1
        self.mus = mus  # from relu 0 to n-1

    @staticmethod
    def get_duals_from(weights, additional_coeffs, lower_bounds, upper_bounds, init_type="crown", alphas=None,
                       gammas=None, betas=None):

        mus, mu, lay_idx = handle_propagation_add_coeff(weights, additional_coeffs, lower_bounds)
        do_kw = (init_type == "KW" and alphas is None)
        do_crown = (init_type == "crown" and alphas is None)

        lbdas = []
        while lay_idx > 0:
            lay = weights[lay_idx]
            lbda = lay.backward(mu)
            lbdas.append(lbda)

            lbs = lower_bounds[lay_idx].unsqueeze(1)
            ubs = upper_bounds[lay_idx].unsqueeze(1)

            ub_slope = (ubs / (ubs - lbs))
            ub_slope.masked_fill_(lbs >= 0, 1)
            ub_slope.masked_fill_(ubs <= 0, 0)

            if do_crown:
                # Use CROWN slopes assignment.
                lb_slope = (ubs >= torch.abs(lbs)).type(lbs.dtype)
                lb_slope.masked_fill_(lbs >= 0, 1)
                lb_slope.masked_fill_(ubs <= 0, 0)
            elif alphas is None:
                # KW slopes assignment
                lb_slope = ub_slope
            else:
                # lb_slope for ambiguous neurons passed via alpha.
                lb_slope = torch.where((lbs < 0) & (ubs > 0), alphas[lay_idx-1], ub_slope)

            if not do_kw:
                mu = torch.where(lbda >= 0, ub_slope, lb_slope) * lbda
            else:
                mu = lbda * ub_slope
            if gammas is not None:
                mu += (gammas["l"][lay_idx-1] - gammas["u"][lay_idx-1])
            elif betas is not None:
                mu += ((lbs >= 0).type(lbs.dtype) - (ubs <= 0).type(ubs.dtype)) * betas[lay_idx-1]
            mus.append(mu)
            lay_idx -= 1

        mus.reverse()
        lbdas.reverse()

        return PropDualVars(lbdas, mus)


class PropInit(ParentInit):
    """
    Parent Init class for alpha-crown/gamma-crown.
    """
    def __init__(self, parent_alphas, parent_gammas=None, parent_betas=None):
        # parent_alphas/parent_gammas are the dual values (list of tensors and dict of list of tensors) at parent
        # termination
        self.alphas = parent_alphas
        self.gammas = parent_gammas
        self.betas = parent_betas

    def to_cpu(self):
        # Move content to cpu.
        self.alphas = [cvar.cpu() for cvar in self.alphas]
        if self.gammas is not None:
            self.gammas = {"l": [cvar.cpu() for cvar in self.gammas["l"]],
                           "u": [cvar.cpu() for cvar in self.gammas["u"]]}
        if self.betas is not None:
            self.betas = [cvar.cpu() for cvar in self.betas]

    def to_device(self, device):
        # Move content to device "device"
        self.alphas = [cvar.to(device) for cvar in self.alphas]
        if self.gammas is not None:
            self.gammas = {"l": [cvar.to(device) for cvar in self.gammas["l"]],
                           "u": [cvar.to(device) for cvar in self.gammas["u"]]}
        if self.betas is not None:
            self.betas = [cvar.to(device) for cvar in self.betas]

    def as_stack(self, stack_size):
        # Repeat (copies) the content of this parent init to form a stack of size "stack_size"
        stacked_alphas = self.do_stack_list(self.alphas, stack_size)
        stacked_gammas = None
        stacked_betas = None
        if self.gammas is not None:
            stacked_gammas = {"l": self.do_stack_list(self.gammas["l"], stack_size),
                              "u": self.do_stack_list(self.gammas["u"], stack_size)}
        if self.betas is not None:
            stacked_betas = self.do_stack_list(self.betas, stack_size)
        return PropInit(stacked_alphas, parent_gammas=stacked_gammas, parent_betas=stacked_betas)

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        for x_idx in range(len(self.alphas)):
            self.set_parent_entries(self.alphas[x_idx], parent_solution.alphas[x_idx], batch_idx)
            if self.gammas is not None:
                for key in self.gammas:
                    self.set_parent_entries(self.gammas[key][x_idx], parent_solution.gammas[key][x_idx], batch_idx)
            if self.betas is not None:
                self.set_parent_entries(self.betas[x_idx], parent_solution.betas[x_idx], batch_idx)

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        alphas = self.get_entry_list(self.alphas, batch_idx)
        gammas = None
        betas = None
        if self.gammas is not None:
            gammas = {"l": self.get_entry_list(self.gammas["l"], batch_idx),
                      "u": self.get_entry_list(self.gammas["u"], batch_idx)}
        if self.betas is not None:
            betas = self.get_entry_list(self.betas, batch_idx)
        return PropInit(alphas, parent_gammas=gammas, parent_betas=betas)

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        # this operation makes sense only in the BaB context (single output neuron), when both lb and ub where computed.
        assert self.alphas[0].shape[1] == 2
        alphas = self.lb_only_list(self.alphas)
        gammas = None
        betas = None
        if self.gammas is not None:
            gammas = {"l": self.lb_only_list(self.gammas["l"]),
                      "u": self.lb_only_list(self.gammas["u"])}
        if self.betas is not None:
            betas = self.lb_only_list(self.betas)
        return PropInit(alphas, parent_gammas=gammas, parent_betas=betas)

    def get_presplit_parents(self):
        # Before bounding, the parent init class contains two copies of each initializer (one per BaB children).
        # Return only one of them.
        def halve(xlist):
            return [x.view((x.shape[0]//2, 2, *x.shape[1:])).select(1, 0) for x in xlist]
        alphas = halve(self.alphas)
        gammas = {"l": halve(self.gammas["l"]), "u": halve(self.gammas["u"])} if self.gammas is not None else None
        betas = halve(self.betas) if self.betas is not None else None
        return PropInit(alphas, parent_gammas=gammas, parent_betas=betas)

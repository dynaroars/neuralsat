import torch
from torch import nn
from src.dual_bounding import DualBounding
from src import utils


default_params = {
    'nb_steps': 5,
    'initial_step_size': 1e0,
    'step_size_decay': 0.98,
    'betas': (0.9, 0.999),
    'best_among': None,  # Used only if type is "best_prop"
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

        assert type in ["naive", "KW", "crown", "best_prop", "alpha-crown"]
        if type == "best_prop":
            assert self.params["best_among"] is not None, "Must provide list of prop types to choose best from"
            self.optimize = self.best_prop_optimizers(self.params["best_among"])
        else:
            self.type = type
            self.optimize = self.propagation_optimizer

    def propagation_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):

        add_coeff = next(iter(additional_coeffs.values()))
        if self.type in ["crown", "KW"]:
            # Compute dual variables.
            dual_vars = PropDualVars.get_duals_from(
                weights, additional_coeffs, lower_bounds, upper_bounds, init_type=self.type)

            # Compute objective.
            bounding_out = self.compute_bounds(weights, add_coeff, dual_vars, lower_bounds, upper_bounds)

        elif self.type == "naive":
            # 'naive'
            lay_n = len(weights) - 1
            xn_coeff = weights[lay_n].backward(add_coeff)

            argmin_ibs = torch.where(xn_coeff >= 0, lower_bounds[lay_n].unsqueeze(1), upper_bounds[lay_n].unsqueeze(1))
            # The bounds of layers after the first need clamping (as these are post-activation bounds)
            if lay_n > 0:
                argmin_ibs = argmin_ibs.clamp(0, None)

            bounding_out = utils.bdot(argmin_ibs, xn_coeff)
            bounding_out += utils.bdot(add_coeff, weights[-1].get_bias())

        elif self.type in ["alpha-crown"]:

            # Assign crown lb_slope to alphas
            alphas = []
            for lay_idx, (lbs, ubs) in enumerate(zip(lower_bounds, upper_bounds)):
                if lay_idx > 0 and lay_idx < len(weights):
                    neuron_batch_size = add_coeff.shape[1]
                    crown_lb_slope = (ubs >= torch.abs(lbs)).type(lbs.dtype).unsqueeze(1).repeat(
                        (1, neuron_batch_size) + ((1,) * (lbs.dim() - 1)))
                    alphas.append(crown_lb_slope)

            # define objective function
            def obj(alphas, store_bounds_primal=False):
                prop_vars = PropDualVars.get_duals_from(
                    weights, additional_coeffs, lower_bounds, upper_bounds, alphas=alphas)
                bound = self.compute_bounds(weights, add_coeff, prop_vars, lower_bounds, upper_bounds,
                                            store_primal=store_bounds_primal)
                return bound

            with torch.enable_grad():

                # Mark which variables we are optimizing over...
                optvars = alphas
                for cvar in optvars:
                    cvar.requires_grad = True

                # ...and pass them as a list to the optimizer.
                optimizer = torch.optim.Adam(optvars, lr=self.params["initial_step_size"], betas=self.params["betas"])
                # Decay step size.
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.params["step_size_decay"])

                # do autograd-adam
                for step in range(self.params["nb_steps"]):
                    optimizer.zero_grad()
                    obj_value = -obj(alphas)
                    obj_value.mean().backward()
                    optimizer.step()
                    scheduler.step()

                    # Project into [0,1], R+
                    with torch.no_grad():
                        for alpha in alphas:
                            alpha.clamp_(0, 1)

                # Detach variables before returning them.
                alphas = [alpha.detach() for alpha in alphas]

                # End of the optimization
                bounding_out = obj(alphas, store_bounds_primal=self.store_bounds_primal)

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

    def compute_bounds(self, weights, add_coeff, dual_vars, lower_bounds, upper_bounds, store_primal=False):
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
        return net_input

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
    variables over which to optimize (alpha in the auto-LiRPA formulation)
    If the ub/lb slopes are not optimized over, they are set to CROWN's (or KW, depending on init_type).
    """
    def __init__(self, lambdas, mus):
        self.lambdas = lambdas  # from relu 0 to n-1
        self.mus = mus  # from relu 0 to n-1

    @staticmethod
    def get_duals_from(weights, additional_coeffs, lower_bounds, upper_bounds, init_type="crown", alphas=None):

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
            mus.append(mu)
            lay_idx -= 1

        mus.reverse()
        lbdas.reverse()

        return PropDualVars(lbdas, mus)


import copy
import time

try:
    import gurobipy as grb
except ImportError:
    pass
from torch.nn import ZeroPad2d

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from modules import Flatten


def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
    return new_all_layers


def add_single_prop(layers, gt, cls):
    """
    gt: ground truth lablel
    cls: class we want to verify against
    """
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt.detach()] = 1

    final_layers = [layers[-1], additional_lin_layer]
    final_layer = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


class InfeasibleMaskException(Exception):
    pass


class LiRPAConvNet:

    def __init__(self, model_ori, pred, test, solve_slope=False, device='cuda', simplify=True, in_size=(1, 3, 32, 32)):
        """
        convert pytorch model to auto_LiRPA module
        """
        assert type(model_ori[-1]) is torch.nn.Linear
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        if simplify:
            added_prop_layers = add_single_prop(layers, pred, test)
            self.layers = added_prop_layers
        else:
            self.layers = layers
        # net = nn.Sequential(*self.layers)
        net[-1] = self.layers[-1]
        self.solve_slope = solve_slope
        if solve_slope:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'random_evaluation', 'conv_mode': 'patches', 'ob_get_heuristic': False},
                                     device=device)
        else:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'same-slope', 'conv_mode': 'patches'}, device=device)
        self.net.eval()

    def get_lower_bound(self, relu_mask, pre_lbs, pre_ubs, decision, choice, parallel=False, no_LP=False, slopes=None,
                        warm=True, pre_split=False, use_gnn=False, history=None, decision_thresh=0, layer_set_bound=True,
                        lr_alpha=0.1):
        try:
            start = time.time()
            ret = self.update_the_model_warm(relu_mask, pre_lbs, pre_ubs, decision, choice, parallel=parallel,
                                             no_LP=no_LP, slopes=slopes, pre_split=pre_split, history=history,
                                             decision_thresh=decision_thresh, layer_set_bound=layer_set_bound, lr_alpha=lr_alpha)
            end = time.time()
            print('bounding time: ', end - start)
            return ret
        except InfeasibleMaskException:
            # The model is infeasible, so this mask is wrong.
            # We just return an infinite lower bound
            if use_gnn:
                return float('inf'), float('inf'), None, None, None, None, relu_mask, None, None, None
            else:
                return float('inf'), float('inf'), None, relu_mask, None, None, None

    def check_optimization_success(self, introduced_constrs_all=None, model=None):
        if model is None:
            model = self.model
        if model.status == 2:
            # Optimization successful, nothing to complain about
            pass
        elif model.status == 3:
            for introduced_cons_layer in introduced_constrs_all:
                model.remove(introduced_cons_layer)
            # The model is infeasible. We have made incompatible
            # assumptions, so this subdomain doesn't exist.
            raise InfeasibleMaskException()
        else:
            print('\n')
            print(f'model.status: {model.status}\n')
            raise NotImplementedError

    def get_relu(self, model, idx):
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer

    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        if self.input_domain.ndim == 2:
            lower_bounds = [self.input_domain[:, 0].squeeze(-1)]
            upper_bounds = [self.input_domain[:, 1].squeeze(-1)]
        else:
            lower_bounds = [self.input_domain[:, :, :, 0].squeeze(-1)]
            upper_bounds = [self.input_domain[:, :, :, 1].squeeze(-1)]
        self.pre_relu_indices = []
        idx, i, model_i = 0, 0, 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {0: model.root_name[0]}
        model_names = list(model._modules)

        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                i += 1
                this_relu = self.get_relu(model, i)
                lower_bounds[-1] = this_relu.inputs[0].lower.squeeze().detach()
                upper_bounds[-1] = this_relu.inputs[0].upper.squeeze().detach()
                lower_bounds.append(F.relu(lower_bounds[-1]).detach())
                upper_bounds.append(F.relu(upper_bounds[-1]).detach())
                self.pre_relu_indices.append(idx)
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 1
            elif isinstance(layer, Flatten):
                lower_bounds.append(lower_bounds[-1].reshape(-1).detach())
                upper_bounds.append(upper_bounds[-1].reshape(-1).detach())
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 8  # Flatten is split to 8 ops in BoundedModule
            elif isinstance(layer, ZeroPad2d):
                lower_bounds.append(F.pad(lower_bounds[-1], layer.padding))
                upper_bounds.append(F.pad(upper_bounds[-1], layer.padding))
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 24
            else:
                self.name_dict[idx + 1] = model_names[model_i]
                lower_bounds.append([])
                upper_bounds.append([])
                model_i += 1
            idx += 1

        # Also add the bounds on the final thing
        lower_bounds[-1] = (lb.view(-1).detach())
        upper_bounds[-1] = (ub.view(-1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices

    def get_candidate_parallel(self, model, lb, ub, batch):
        # get the intermediate bounds in the current model
        lower_bounds = [self.input_domain[:, :, :, 0].squeeze(-1).repeat(batch, 1, 1, 1)]
        upper_bounds = [self.input_domain[:, :, :, 1].squeeze(-1).repeat(batch, 1, 1, 1)]
        idx, i, = 0, 0
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                i += 1
                this_relu = self.get_relu(model, i)
                lower_bounds[-1] = this_relu.inputs[0].lower.detach()
                upper_bounds[-1] = this_relu.inputs[0].upper.detach()
                lower_bounds.append(
                    F.relu(lower_bounds[-1]).detach())  # TODO we actually do not need the bounds after ReLU
                upper_bounds.append(F.relu(upper_bounds[-1]).detach())
            elif isinstance(layer, Flatten):
                lower_bounds.append(lower_bounds[-1].reshape(batch, -1).detach())
                upper_bounds.append(upper_bounds[-1].reshape(batch, -1).detach())
            elif isinstance(layer, nn.ZeroPad2d):
                lower_bounds.append(F.pad(lower_bounds[-1], layer.padding).detach())
                upper_bounds.append(F.pad(upper_bounds[-1], layer.padding).detach())

            else:
                lower_bounds.append([])
                upper_bounds.append([])
            idx += 1

        # Also add the bounds on the final thing
        lower_bounds[-1] = (lb.view(batch, -1).detach())
        upper_bounds[-1] = (ub.view(batch, -1).detach())

        return lower_bounds, upper_bounds

    def get_mask_parallel(self, model):
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons
        mask = []
        for this_relu in model.relus:
            mask_tmp = torch.zeros_like(this_relu.inputs[0].lower)
            unstable = ((this_relu.inputs[0].lower < 0) & (this_relu.inputs[0].upper > 0))
            mask_tmp[unstable] = -1
            active = (this_relu.inputs[0].lower >= 0)
            mask_tmp[active] = 1
            # otherwise 0, for inactive neurons

            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))

        ret = []
        for i in range(mask[0].size(0)):
            ret.append([j[i] for j in mask])

        return ret

    def get_slope(self, model):
        s = []
        for m in model.relus:
            s.append(m.slope.transpose(0, 1).clone().detach())

        ret = []
        for i in range(s[0].size(0)):
            ret.append([j[i] for j in s])
        return ret

    def set_slope(self, model, slope, repeat=2):
        idx = 0
        for m in model.relus:
            # m.slope = slope[idx].repeat(2, *([1] * (slope[idx].ndim - 1))).requires_grad_(True)
            m.slope = slope[idx].repeat(repeat, *([1] * (slope[idx].ndim - 1))).transpose(0, 1).requires_grad_(True)
            idx += 1

    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, decision=None, slopes=None,
                               beta=False, early_stop=True, opt_choice="default", iteration=20, history=None,
                               layer_set_bound=True, lr_alpha=0.1):
        # update optimize-CROWN bounds in a parallel way
        total_batch = len(decision)
        decision = np.array(decision)

        layers_need_change = np.unique(decision[:, 0])
        layers_need_change.sort()

        self.replacing_bd_index = min(self.replacing_bd_index, self.pre_relu_indices[layers_need_change[0]])

        # initial results with empty list
        ret_l = [[] for _ in range(len(decision) * 2)]
        ret_u = [[] for _ in range(len(decision) * 2)]
        masks = [[] for _ in range(len(decision) * 2)]
        ret_s = [[] for _ in range(len(decision) * 2)]

        pre_lb_all_cp = copy.deepcopy(pre_lb_all)
        pre_ub_all_cp = copy.deepcopy(pre_ub_all)

        for idx in layers_need_change:
            # iteratively change upper and lower bound from former to later layer
            tmp_d = np.argwhere(decision[:, 0] == idx)  # .squeeze()
            # idx is the index of relu layers, change_idx is the index of all layers
            change_idx = self.pre_relu_indices[idx]
            batch = len(tmp_d)

            # TODO ADD ARGUMENT
            for m in self.net.relus:
                m.beta = None

            slope_select = [i[tmp_d.squeeze()].clone() for i in slopes]

            pre_lb_all = [i[tmp_d.squeeze()].clone() for i in pre_lb_all_cp]
            pre_ub_all = [i[tmp_d.squeeze()].clone() for i in pre_ub_all_cp]

            if batch == 1:
                pre_lb_all = [i.clone().unsqueeze(0) for i in pre_lb_all]
                pre_ub_all = [i.clone().unsqueeze(0) for i in pre_ub_all]
                slope_select = [i.clone().unsqueeze(0) for i in slope_select]

            upper_bounds = [i.clone() for i in pre_ub_all[:change_idx + 1]]
            lower_bounds = [i.clone() for i in pre_lb_all[:change_idx + 1]]
            upper_bounds_cp = copy.deepcopy(upper_bounds)
            lower_bounds_cp = copy.deepcopy(lower_bounds)

            for i in range(batch):
                d = tmp_d[i][0]
                upper_bounds[change_idx].view(batch, -1)[i][decision[d][1]] = 0
                lower_bounds[change_idx].view(batch, -1)[i][decision[d][1]] = 0

            pre_lb_all = [torch.cat(2 * [i]) for i in pre_lb_all]
            pre_ub_all = [torch.cat(2 * [i]) for i in pre_ub_all]

            # merge the inactive and active splits together
            new_candidate = {}
            for i, (l, uc, lc, u) in enumerate(zip(lower_bounds, upper_bounds_cp, lower_bounds_cp, upper_bounds)):
                # we set lower = 0 in first half batch, and upper = 0 in second half batch
                new_candidate[self.name_dict[i]] = [torch.cat((l, lc), dim=0), torch.cat((uc, u), dim=0)]

            if not layer_set_bound:
                new_candidate_p = {}
                for i, (l, u) in enumerate(zip(pre_lb_all[:-2], pre_ub_all[:-2])):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    new_candidate_p[self.name_dict[i]] = [l, u]

            # create new_x here since batch may change
            ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                     x_L=self.x.ptb.x_L.repeat(batch * 2, 1, 1, 1),
                                     x_U=self.x.ptb.x_U.repeat(batch * 2, 1, 1, 1))
            new_x = BoundedTensor(self.x.data.repeat(batch * 2, 1, 1, 1), ptb)
            self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

            if len(slope_select) > 0:
                # set slope here again
                self.set_slope(self.net, slope_select)

            if layer_set_bound:
                # we fix the intermediate bounds before change_idx-th layer by using normal CROWN
                if self.solve_slope and change_idx >= self.pre_relu_indices[-1]:
                    # we split the ReLU at last layer, directly use Optimized CROWN
                    self.net.set_bound_opts({'optimize_bound_args':
                                                 {'ob_start_idx': sum(change_idx <= x for x in self.pre_relu_indices),
                                                  'ob_beta': beta,
                                                  'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                                  'ob_lr': lr_alpha}})
                    lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                      new_interval=new_candidate, return_A=False, bound_upper=False)
                else:
                    # we split the ReLU before the last layer, calculate intermediate bounds by using normal CROWN
                    self.net.set_relu_used_count(sum(change_idx <= x for x in self.pre_relu_indices))
                    # big overhead here, this step takes 2/3 running time,
                    with torch.no_grad():
                        lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                          new_interval=new_candidate, bound_upper=False, return_A=False)

                # we don't care about the upper bound of the last layer
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)

                if change_idx < self.pre_relu_indices[-1]:
                    # check whether we have a better bounds before, and preset all intermediate bounds
                    for i, (l, u) in enumerate(
                            zip(lower_bounds_new[change_idx + 2:-1], upper_bounds_new[change_idx + 2:-1])):
                        new_candidate[self.name_dict[i + change_idx + 2]] = [
                            torch.max(l, pre_lb_all[i + change_idx + 2]), torch.min(u, pre_ub_all[i + change_idx + 2])]

                    if self.solve_slope:
                        # new_candidate already set so we start CROWN from the last layer
                        self.net.set_bound_opts({'optimize_bound_args': {'ob_start_idx': 1, 'ob_beta': beta,
                                                                         'ob_update_by_layer': layer_set_bound,
                                                                         'ob_iteration': iteration,
                                                                         'ob_lr': lr_alpha}})
                        lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                          new_interval=new_candidate, return_A=False, bound_upper=False)
                    else:
                        self.net.set_relu_used_count(sum(change_idx <= x for x in self.pre_relu_indices))
                        with torch.no_grad():
                            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                              new_interval=new_candidate, bound_upper=False,
                                                              return_A=False)
                    lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)
            else:
                # all intermediate bounds are re-calculated by optimized CROWN
                self.net.set_bound_opts(
                    {'optimize_bound_args': {'ob_start_idx': 99, 'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                                             'ob_iteration': iteration, 'ob_lr': lr_alpha}})
                lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                  new_interval=new_candidate_p, return_A=False, bound_upper=False)
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)

            # print('best results of parent nodes', pre_lb_all[-1].repeat(2, 1))
            # print('finally, after optimization:', lower_bounds_new[-1])

            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1])
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1])

            mask = self.get_mask_parallel(self.net)
            if len(slope_select) > 0:
                slope = self.get_slope(self.net)

            # reshape the results
            for i in range(len(tmp_d)):
                ret_l[int(tmp_d[i])] = [j[i] for j in lower_bounds_new]
                ret_l[int(tmp_d[i] + total_batch)] = [j[i + batch] for j in lower_bounds_new]

                ret_u[int(tmp_d[i])] = [j[i] for j in upper_bounds_new]
                ret_u[int(tmp_d[i] + total_batch)] = [j[i + batch] for j in upper_bounds_new]

                masks[int(tmp_d[i])] = mask[i]
                masks[int(tmp_d[i] + total_batch)] = mask[i + batch]
                if len(slope_select) > 0:
                    ret_s[int(tmp_d[i])] = slope[i]
                    ret_s[int(tmp_d[i] + total_batch)] = slope[i + batch]

        return ret_l, ret_u, masks, ret_s

    def copy_model(self, model, basis=True, use_basis_warm_start=True, remove_constr_list=[]):
        model_split = model.copy()

        # print(model_split.printStats())
        for rc in remove_constr_list:
            rcs = model_split.getConstrByName(rc.ConstrName)
            model_split.remove(rcs)
        model_split.update()

        if not basis:
            return model_split

        xvars = model.getVars()
        svars = model_split.getVars()
        # print(len(xvars), len(svars))
        for x, s in zip(xvars, svars):
            if use_basis_warm_start:
                s.VBasis = x.VBasis
            else:
                s.PStart = x.X

        sconstrs = model_split.getConstrs()
        # xconstrs = model.getConstrs()
        # print(len(xconstrs), len(sconstrs))

        for s in sconstrs:
            x = model.getConstrByName(s.ConstrName)
            if use_basis_warm_start:
                s.CBasis = x.CBasis
            else:
                s.DStart = x.Pi
        model_split.update()
        return model_split


    def fake_forward(self, x):
        for layer in self.layers:
            if type(layer) is nn.Linear:
                x = F.linear(x, layer.weight, layer.bias)
            elif type(layer) is nn.Conv2d:
                x = F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
            elif type(layer) == nn.ReLU:
                x = F.relu(x)
            elif type(layer) == Flatten:
                x = x.reshape(x.shape[0], -1)
            elif type(layer) == nn.ZeroPad2d:
                x = F.pad(x, layer.padding)
            else:
                print(type(layer))
                raise NotImplementedError

        return x

    def get_primals(self, A, return_x=False):
        input_A_lower = A[self.layer_names[-1]][self.net.input_name[0]][0]
        batch = input_A_lower.shape[1]
        l = self.input_domain[:, :, :, 0].repeat(batch, 1, 1, 1)
        u = self.input_domain[:, :, :, 1].repeat(batch, 1, 1, 1)
        diff = 0.5 * (l - u)  # already flip the sign by using lower - upper
        net_input = diff * torch.sign(input_A_lower.squeeze(0)) + self.x
        if return_x: return net_input

        primals = [net_input]
        for layer in self.layers:
            if type(layer) is nn.Linear:
                pre = primals[-1]
                primals.append(F.linear(pre, layer.weight, layer.bias))
            elif type(layer) is nn.Conv2d:
                pre = primals[-1]
                primals.append(F.conv2d(pre, layer.weight, layer.bias,
                                        layer.stride, layer.padding, layer.dilation, layer.groups))
            elif type(layer) == nn.ReLU:
                primals.append(F.relu(primals[-1]))
            elif type(layer) == Flatten:
                primals.append(primals[-1].reshape(primals[-1].shape[0], -1))
            else:
                print(type(layer))
                raise NotImplementedError

        # primals = primals[1:]
        primals = [i.detach().clone() for i in primals]
        # print('primals', primals[-1])

        return net_input, primals

    def build_the_model(self, input_domain, x, use_gnn=False, no_lp=False, decision_thresh=0, continue_LP=None):

        new_relu_mask = []
        self.x = x
        self.input_domain = input_domain
        slope_opt = None
        if continue_LP is None:
            # first get CROWN bounds
            if self.solve_slope:
                self.net.init_slope(self.x)
                self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': 100, 'ob_beta': False,
                                                                 'ob_alpha': True, 'ob_opt_choice': "adam",
                                                                 'ob_decision_thresh': decision_thresh,
                                                                 'ob_early_stop': False, 'ob_log': False,
                                                                 'ob_start_idx': 99, 'ob_keep_best': True,
                                                                 'ob_update_by_layer': True,
                                                                 'ob_lr': 0.5, 'ob_get_heuristic': False}})
                lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='CROWN-Optimized', return_A=False,
                                                 bound_upper=False)
                slope_opt = self.get_slope(self.net)[0]  # initial with one node only
            else:
                with torch.no_grad():
                    lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='backward', return_A=False)
            # build a complete A_dict
            # self.A_dict = A_dict
            # self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
            # self.layer_names.sort()

            # update bounds
            print('initial CROWN:', lb, ub)
            primals, mini_inp = None, None
            # mini_inp, primals = self.get_primals(self.A_dict)
            lb, ub, self.pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals can be better upper bounds
            self.replacing_bd_index = len(lb)

            return ub[-1], lb[-1], mini_inp, None, primals,\
                        self.get_mask_parallel(self.net)[0], lb, ub, self.pre_relu_indices, slope_opt
        else:
            # continue to build the LP model
            upper_bounds, lower_bounds, slope_opt = continue_LP
            slope_ = [i.clone().unsqueeze(0) for i in slope_opt]
            self.replacing_bd_index = len(lower_bounds)
            pre_relu_indices = self.pre_relu_indices

            # set slopes and retrieve intermediate bounds
            self.net(self.x)
            self.set_slope(self.net, slope_, repeat=1)
            with torch.no_grad():
                self.net.set_relu_used_count(99)
                lb, _, = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='backward',
                                                 new_interval=None, bound_upper=False, return_A=False)
            print('refine bound:', lb)
            lower_bounds, upper_bounds, self.pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)

        # Initialize the LP model
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)

        # keep a record of model's information
        self.gurobi_vars = []
        # self.feas_vars = []
        self.relu_constrs = []
        self.relu_indices_mask = []

        ## Do the input layer, which is a special case
        inp_gurobi_vars = []
        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
        else:
            assert input_domain.dim() == 4
            for chan in range(input_domain.size(0)):
                chan_vars = []
                for row in range(input_domain.size(1)):
                    row_vars = []
                    for col in range(input_domain.size(2)):
                        lb = input_domain[chan, row, col, 0]
                        ub = input_domain[chan, row, col, 1]
                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'inp_[{chan},{row},{col}]')
                        # fv = self.feas_model.addVar(lb=lb, ub=ub, obj=0,
                        #                       vtype=grb.GRB.CONTINUOUS,
                        #                       name=f'inp_[{chan},{row},{col}]')
                        # self.feas_vars.append(fv)
                        row_vars.append(v)
                    chan_vars.append(row_vars)
                inp_gurobi_vars.append(chan_vars)
        self.model.update()

        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        relu_idx = 0
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                # Get the better estimates from KW and Interval Bounds
                out_lbs = lower_bounds[layer_idx]
                out_ubs = upper_bounds[layer_idx]
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, self.gurobi_vars[-1])

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(lin_expr == v)
                    self.model.update()

                    new_layer_gurobi_vars.append(v)

            elif type(layer) is nn.Conv2d:
                assert layer.dilation == (1, 1)
                pre_lb_size = lower_bounds[layer_idx - 1].unsqueeze(0).size()
                out_lbs = lower_bounds[layer_idx].unsqueeze(0)
                out_ubs = upper_bounds[layer_idx].unsqueeze(0)

                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_vars = []
                        for out_col_idx in range(out_lbs.size(3)):
                            lin_expr = layer.bias[out_chan_idx].item()

                            for in_chan_idx in range(layer.weight.shape[1]):
                                for ker_row_idx in range(layer.weight.shape[2]):
                                    in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                    if (in_row_idx < 0) or (in_row_idx == pre_lb_size[2]):
                                        # This is padding -> value of 0
                                        continue
                                    for ker_col_idx in range(layer.weight.shape[3]):
                                        in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                        if (in_col_idx < 0) or (in_col_idx == pre_lb_size[3]):
                                            # This is padding -> value of 0
                                            continue
                                        coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                        lin_expr += coeff * self.gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

            elif isinstance(layer, nn.ZeroPad2d):
                out_lbs = lower_bounds[layer_idx].unsqueeze(0)
                out_ubs = upper_bounds[layer_idx].unsqueeze(0)
                left, right, top, bottom = layer.padding

                for out_chan_idx in range(out_lbs.size(1)):
                    out_chan_vars = []
                    for out_row_idx in range(out_lbs.size(2)):
                        out_row_vars = []
                        row_pad = out_row_idx < left or out_row_idx >= out_lbs.size(2)-right
                        for out_col_idx in range(out_lbs.size(3)):
                            col_pad = out_col_idx < top or out_col_idx >= out_lbs.size(3)-bottom
                            if row_pad or col_pad:
                                lin_expr = 0
                            else:
                                lin_expr = self.gurobi_vars[-1][out_chan_idx][out_row_idx-1][out_col_idx-1]

                            out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                            v = self.model.addVar(lb=out_lb, ub=out_ub,
                                                  obj=0, vtype=grb.GRB.CONTINUOUS,
                                                  name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                            self.model.addConstr(lin_expr == v)
                            self.model.update()

                            out_row_vars.append(v)
                        out_chan_vars.append(out_row_vars)
                    new_layer_gurobi_vars.append(out_chan_vars)

            elif type(layer) is nn.ReLU:
                new_relu_layer_constr = []
                relu_idx += 1
                this_relu = self.get_relu(self.net, relu_idx)
                if isinstance(self.gurobi_vars[-1][0], list):
                    # This is convolutional
                    pre_lbs = lower_bounds[layer_idx - 1]
                    pre_ubs = upper_bounds[layer_idx - 1]
                    new_layer_mask = []
                    ratios_all = this_relu.d
                    # bias_all = -pre_lbs * ratios_all
                    # bias_all = bias_all * this_relu.I.squeeze(0).float()
                    # bias_all = bias_all.squeeze(0)
                    temp = pre_lbs.size()
                    out_chain = temp[0]
                    out_height = temp[1]
                    out_width = temp[2]
                    for chan_idx, channel in enumerate(self.gurobi_vars[-1]):
                        chan_vars = []
                        for row_idx, row in enumerate(channel):
                            row_vars = []
                            for col_idx, pre_var in enumerate(row):
                                slope = ratios_all[0, chan_idx, row_idx, col_idx].item()
                                pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()
                                pre_lb = pre_lbs[chan_idx, row_idx, col_idx].item()
                                # bias = bias_all[chan_idx, row_idx, col_idx].item()

                                if slope == 1.0:
                                    # ReLU is always passing
                                    v = pre_var
                                    new_layer_mask.append(1)
                                elif slope == 0.0:
                                    v = zero_var
                                    new_layer_mask.append(0)
                                else:
                                    lb = 0
                                    ub = pre_ub
                                    new_layer_mask.append(-1)
                                    v = self.model.addVar(lb=pre_lb, ub=ub,
                                                          obj=0, vtype=grb.GRB.CONTINUOUS,
                                                          name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                                    out_idx = col_idx + row_idx*out_width + chan_idx*out_height*out_width

                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v>=0, name=f'ReLU{relu_idx-1}_{out_idx}_a_0'))
                                    new_relu_layer_constr.append(
                                        self.model.addConstr(v>=pre_var, name=f'ReLU{relu_idx-1}_{out_idx}_a_1'))
                                    # new_relu_layer_constr.append(self.model.addConstr(v <= slope * pre_var + bias,
                                                                                      # f'ReLU{relu_idx-1}_{out_idx}_a_2'))
                                    new_relu_layer_constr.append(self.model.addConstr(pre_ub * pre_var - (pre_ub-pre_lb) * v >= pre_ub * pre_lb,
                                                                              name=f'ReLU{relu_idx-1}_{out_idx}_a_2'))
                                row_vars.append(v)
                            chan_vars.append(row_vars)
                        new_layer_gurobi_vars.append(chan_vars)
                else:
                    pre_lbs = lower_bounds[layer_idx - 1]
                    pre_ubs = upper_bounds[layer_idx - 1]
                    new_layer_mask = []
                    # slope_all = pre_ubs/(pre_ubs-pre_lbs)
                    # bias_all = -pre_lbs*slope_all
                    # bias_all = bias_all*dual_info[0][layer_idx].I.squeeze(0).float()
                    # ratios_all = dual_info[0][layer_idx].d.squeeze(0)
                    ratios_all = this_relu.d.squeeze(0)
                    # bias_all = -pre_lbs * ratios_all
                    # bias_all = bias_all * this_relu.I.squeeze(0).float()
                    assert isinstance(self.gurobi_vars[-1][0], grb.Var)
                    for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                        pre_ub = pre_ubs[neuron_idx].item()
                        pre_lb = pre_lbs[neuron_idx].item()
                        # print(ratios_all.shape, this_relu.d.shape)
                        slope = ratios_all[neuron_idx].item()
                        # bias = bias_all[neuron_idx].item()

                        if slope == 1.0:
                            # The ReLU is always passing
                            v = pre_var
                            new_layer_mask.append(1)
                        elif slope == 0.0:
                            v = zero_var
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                            new_layer_mask.append(0)
                        else:
                            lb = 0
                            ub = pre_ub
                            v = self.model.addVar(ub=ub, lb=pre_lb,
                                                  obj=0,
                                                  vtype=grb.GRB.CONTINUOUS,
                                                  name=f'ReLU{layer_idx}_{neuron_idx}')

                            new_relu_layer_constr.append(
                                self.model.addConstr(v>=0, name=f'ReLU{relu_idx-1}_{neuron_idx}_a_0'))
                            new_relu_layer_constr.append(
                                self.model.addConstr(v>=pre_var, name=f'ReLU{relu_idx-1}_{neuron_idx}_a_1'))
                            # new_relu_layer_constr.append(self.model.addConstr(v <= slope * pre_var + bias,
                            #                                                   name=f'ReLU{relu_idx-1}_{neuron_idx}_a_2'))
                            new_relu_layer_constr.append(self.model.addConstr( pre_ub * pre_var - (pre_ub-pre_lb) * v >= pre_ub * pre_lb,
                                                                              name=f'ReLU{relu_idx-1}_{neuron_idx}_a_2'))
                            new_layer_mask.append(-1)

                        new_layer_gurobi_vars.append(v)

                new_relu_mask.append(torch.tensor(new_layer_mask, device=lower_bounds[0].device))
                self.relu_constrs.append(new_relu_layer_constr)

            # elif type(layer) == View:
            #     continue
            elif type(layer) == Flatten:
                for chan_idx in range(len(self.gurobi_vars[-1])):
                    for row_idx in range(len(self.gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(self.gurobi_vars[-1][chan_idx][row_idx])
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        self.model.update()
        print('finished building gurobi model, calling optimize function')
        # import pdb; pdb.set_trace()
        guro_start = time.time()
        # self.model.setParam("PreSolve", 0)
        # self.model.setParam("Method", 1)

        self.gurobi_vars[-1][0].LB = -10
        self.gurobi_vars[-1][0].UB = 10
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        # self.model.write("save.lp")
        self.model.optimize()

        # for c in self.model.getConstrs():
        #     print('The dual value of %s : %g %g'%(c.constrName,c.pi, c.slack))

        assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success()

        guro_end = time.time()
        print('Gurobi solved the lp with ', guro_end - guro_start)

        glb = self.gurobi_vars[-1][0].X
        lower_bounds[-1] = torch.tensor([glb], device=lower_bounds[0].device)
        print("gurobi glb:", glb)

        inp_size = lower_bounds[0].size()
        mini_inp = torch.zeros(inp_size, device=lower_bounds[0].device)

        if len(inp_size) == 1:
            # This is a linear input.
            for i in range(inp_size[0]):
                mini_inp[i] = self.gurobi_vars[0][i].x
        elif len(inp_size) == 0:
            mini_inp.data = torch.tensor(self.gurobi_vars[0][0].x).cuda()
        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        mini_inp[i, j, k] = self.gurobi_vars[0][i][j][k].x


        gub = self.net(mini_inp.unsqueeze(0))
        print("gub:", gub)

        # record model information
        # indices for undecided relu-nodes
        self.relu_indices_mask = [torch.nonzero(i == -1, as_tuple=False).view(-1).tolist() for i in new_relu_mask]
        # flatten high-dimensional gurobi var lists
        for l_idx, layer in enumerate(self.layers):
            if type(layer) is nn.Conv2d:
                flattened_gurobi = []
                for chan_idx in range(len(self.gurobi_vars[l_idx + 1])):
                    for row_idx in range(len(self.gurobi_vars[l_idx + 1][chan_idx])):
                        flattened_gurobi.extend(self.gurobi_vars[l_idx + 1][chan_idx][row_idx])
                self.gurobi_vars[l_idx + 1] = flattened_gurobi
                if type(self.layers[l_idx + 1]) is nn.ReLU:
                    flattened_gurobi = []
                    for chan_idx in range(len(self.gurobi_vars[l_idx + 2])):
                        for row_idx in range(len(self.gurobi_vars[l_idx + 2][chan_idx])):
                            flattened_gurobi.extend(self.gurobi_vars[l_idx + 2][chan_idx][row_idx])
                    self.gurobi_vars[l_idx + 2] = flattened_gurobi
            else:
                continue

        duals = [torch.zeros(len(i),3) for i in new_relu_mask]
        duals_other = {}

        for i in self.relu_constrs:
            if len(i) == 0:
                break
            else:
                for constr in i:
                    constr_name = constr.ConstrName.split('_')
                    #print(constr_name)
                    if constr_name[-2] == 'a':
                        layer_idx = int(constr_name[0][4:])
                        cons_idx = int(constr_name[-1])
                        node_idx = int(constr_name[1])
                        duals[layer_idx][node_idx][cons_idx] = constr.getAttr('Pi')
                        #print(layer_idx, node_idx , cons_idx)
                        #print(constr.getAttr('Pi'))
                    else:
                        duals_other[constr.ConstrName] = constr.getAttr('Pi')

        if use_gnn:
            # get all dual variables
            duals = [torch.zeros(len(i),3) for i in new_relu_mask]
            duals_other = {}

            for i in self.relu_constrs:
                if len(i) == 0:
                    break
                else:
                    for constr in i:
                        constr_name = constr.ConstrName.split('_')
                        #print(constr_name)
                        if constr_name[-2] == 'a':
                            layer_idx = int(constr_name[0][4:])
                            cons_idx = int(constr_name[-1])
                            node_idx = int(constr_name[1])
                            duals[layer_idx][node_idx][cons_idx] = constr.getAttr('Pi')
                            #print(layer_idx, node_idx, cons_idx)
                            #print(constr.getAttr('Pi'))
                        else:
                            duals_other[constr.ConstrName] = constr.getAttr('Pi')

            primals = []
            for one_layer in self.gurobi_vars[1:]:
                temp = []
                for var in one_layer:
                    temp.append(var.x)
                primals.append(temp)

            return gub, glb, mini_inp.unsqueeze(0), duals, primals, new_relu_mask, lower_bounds, upper_bounds, pre_relu_indices, slope_opt

        return gub, lower_bounds[-1], mini_inp.unsqueeze(0), None, None, new_relu_mask, lower_bounds, upper_bounds, pre_relu_indices, slope_opt

    def update_the_model_warm(self, relu_mask, pre_lb_all, pre_ub_all, decision, choice, parallel=False, no_LP=False,
                              slopes=None, pre_split=False, history=None, decision_thresh=0, layer_set_bound=True, lr_alpha=0.1):
        if not pre_split:
            lower_bounds, upper_bounds, masks, slopes = self.update_bounds_parallel(pre_lb_all, pre_ub_all, decision,
                                                                                    slopes, relu_mask, history=history, lr_alpha=lr_alpha)

            return [i[-1] for i in upper_bounds], [i[-1] for i in lower_bounds], None, masks, lower_bounds, upper_bounds, slopes

        else:
            lower_bounds, upper_bounds, relu_mask = pre_split
            self.model_split = self.copy_model(self.model)

        # reintroduce ub and lb for gurobi constraints
        introduced_constrs = []
        rep_index = self.replacing_bd_index
        for layer in self.layers[self.replacing_bd_index - 1:]:
            if type(layer) is nn.Linear:
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = upper_bounds[rep_index][idx].item()
                    svar.lb = lower_bounds[rep_index][idx].item()
                # self.model_lower_bounds[rep_index] = lower_bounds[rep_index].clone()
                # self.model_upper_bounds[rep_index] = upper_bounds[rep_index].clone()

            elif type(layer) is nn.Conv2d:
                conv_ub = upper_bounds[rep_index].reshape(-1)
                conv_lb = lower_bounds[rep_index].reshape(-1)
                for idx, var in enumerate(self.gurobi_vars[rep_index]):
                    svar = self.model_split.getVarByName(var.VarName)
                    svar.ub = conv_ub[idx].item()
                    svar.lb = conv_lb[idx].item()
                # self.model_lower_bounds[rep_index] = lower_bounds[rep_index].clone()
                # self.model_upper_bounds[rep_index] = upper_bounds[rep_index].clone()

            elif type(layer) is nn.ReLU:
                # locate relu index and remove all associated constraints
                relu_idx = self.pre_relu_indices.index(rep_index - 1)
                # remove relu constraints

                # self.model.remove(self.relu_constrs[relu_idx])
                # remove relu constraints
                # for rc in self.relu_constrs[relu_idx]:
                #     rcs = self.model_split.getConstrByName(rc.ConstrName)
                #     self.model_split.remove(rcs)
                #     break

                # self.relu_constrs[relu_idx] = []
                # reintroduce relu constraints
                pre_lbs = lower_bounds[rep_index - 1].reshape(-1)
                pre_ubs = upper_bounds[rep_index - 1].reshape(-1)
                for unmasked_idx in self.relu_indices_mask[relu_idx]:
                    pre_lb = pre_lbs[unmasked_idx].item()
                    pre_ub = pre_ubs[unmasked_idx].item()
                    var = self.gurobi_vars[rep_index][unmasked_idx]
                    svar = self.model_split.getVarByName(var.VarName)
                    pre_var = self.gurobi_vars[rep_index - 1][unmasked_idx]
                    pre_svar = self.model_split.getVarByName(pre_var.VarName)

                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        svar.lb = pre_lb
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(pre_svar == svar))
                        relu_mask[relu_idx][unmasked_idx] = 1
                    elif pre_lb <= 0 and pre_ub <= 0:
                        svar.lb = 0
                        svar.ub = 0
                        relu_mask[relu_idx][unmasked_idx] = 0
                    else:
                        svar.lb = 0
                        svar.ub = pre_ub
                        introduced_constrs.append(self.model_split.addConstr(svar >= pre_svar))
                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        introduced_constrs.append(self.model_split.addConstr(svar <= slope * pre_svar + bias))

            # elif type(layer) is View:
            #     pass
            elif type(layer) is Flatten:
                pass
            else:
                raise NotImplementedError
            self.model_split.update()
            rep_index += 1

        # compute optimum
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"
        target_var = self.gurobi_vars[-1][0]
        target_svar = self.model_split.getVarByName(target_var.VarName)

        self.model_split.update()
        # self.model.reset()
        self.model_split.setObjective(target_svar, grb.GRB.MINIMIZE)
        # self.model.setObjective(0, grb.GRB.MINIMIZE)
        self.model_split.optimize()
        # assert self.model.status == 2, "LP wasn't optimally solved"
        self.check_optimization_success([introduced_constrs], model=self.model_split)

        glb = target_svar.X
        lower_bounds[-1] = torch.tensor([glb], device=lower_bounds[0].device)

        # get input variable values at which minimum is achieved
        inp_size = lower_bounds[0].size()
        mini_inp = torch.zeros(inp_size, device=lower_bounds[0].device)
        if len(inp_size) == 1:
            # This is a linear input.
            for i in range(inp_size[0]):
                var = self.gurobi_vars[0][i]
                svar = self.model_split.getVarByName(var.VarName)
                mini_inp[i] = svar.x

        else:
            for i in range(inp_size[0]):
                for j in range(inp_size[1]):
                    for k in range(inp_size[2]):
                        var = self.gurobi_vars[0][i][j][k]
                        svar = self.model_split.getVarByName(var.VarName)
                        mini_inp[i, j, k] = svar.x
        gub = self.net(mini_inp.unsqueeze(0))
        # print(gub, glb, self.model_split.status)
        # assert gub>=glb, "wrong constraints added, not sound!"

        del self.model_split

        return gub, lower_bounds[-1], mini_inp.unsqueeze(0), relu_mask, lower_bounds, upper_bounds, slopes

import torch
from torch import nn
from torch.nn import functional as F

from plnn_bounds.modules import View, Flatten

class NaiveNetwork:
    """
    Class implementing naive interval bound propagation.
    """
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def remove_maxpools(self, domain):
        from plnn_bounds.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers

    def get_lower_bound(self, domain):
        self.do_interval_analysis(domain)
        return self.lower_bounds[-1]

    def do_interval_analysis(self, inp_domain):
        self.lower_bounds = []
        self.upper_bounds = []

        self.lower_bounds.append(inp_domain.select(-1, 0))
        self.upper_bounds.append(inp_domain.select(-1, 1))
        layer_idx = 1
        current_lb = self.lower_bounds[-1]
        current_ub = self.upper_bounds[-1]
        for layer in self.layers:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                if type(layer) is nn.Linear:
                    pos_weights = torch.clamp(layer.weight, min=0)
                    neg_weights = torch.clamp(layer.weight, max=0)

                    new_layer_lb = torch.mv(pos_weights, current_lb) + \
                                   torch.mv(neg_weights, current_ub) + \
                                   layer.bias
                    new_layer_ub = torch.mv(pos_weights, current_ub) + \
                                   torch.mv(neg_weights, current_lb) + \
                                   layer.bias
                elif type(layer) is nn.Conv2d:
                    pre_lb = torch.Tensor(current_lb).unsqueeze(0)
                    pre_ub = torch.Tensor(current_ub).unsqueeze(0)
                    pos_weight = torch.clamp(layer.weight, 0, None)
                    neg_weight = torch.clamp(layer.weight, None, 0)

                    out_lbs = (F.conv2d(pre_lb, pos_weight, layer.bias,
                                        layer.stride, layer.padding, layer.dilation, layer.groups)
                               + F.conv2d(pre_ub, neg_weight, None,
                                          layer.stride, layer.padding, layer.dilation, layer.groups))
                    out_ubs = (F.conv2d(pre_ub, pos_weight, layer.bias,
                                        layer.stride, layer.padding, layer.dilation, layer.groups)
                               + F.conv2d(pre_lb, neg_weight, None,
                                          layer.stride, layer.padding, layer.dilation, layer.groups))
                    new_layer_lb = out_lbs.squeeze(0)
                    new_layer_ub = out_ubs.squeeze(0)
                self.lower_bounds.append(new_layer_lb)
                self.upper_bounds.append(new_layer_ub)
                current_lb = new_layer_lb
                current_ub = new_layer_ub
            elif type(layer) == nn.ReLU:
                current_lb = torch.clamp(current_lb, min=0)
                current_ub = torch.clamp(current_ub, min=0)
            elif type(layer) == nn.MaxPool1d:
                new_layer_lb = []
                new_layer_ub = []
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported Maxpool option"

                nb_pre = len(self.lower_bounds[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    lb = max(current_lb[pre_start_idx:pre_window_end])
                    ub = max(current_ub[pre_start_idx:pre_window_end])

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)

                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                current_lb = torch.Tensor(new_layer_lb)
                current_ub = torch.Tensor(new_layer_ub)
                self.lower_bounds.append(current_lb)
                self.upper_bounds.append(current_ub)
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                current_lb = current_lb.view(-1)
                current_ub = current_ub.view(-1)
            else:
                raise NotImplementedError

    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.shape[:-1]
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        sp_shape = (nb_samples, ) + nb_inp
        rand_samples = torch.Tensor(*sp_shape)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.unsqueeze(0).expand(*sp_shape)
        domain_width = domain_width.unsqueeze(0).expand(*sp_shape)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_random

import torch.nn as nn
import torch

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

class BatchDeepPoly(nn.Module):

    def __init__(self, net, back_sub_steps=100):
        super(BatchDeepPoly, self).__init__()

        self.net = net
        self.back_sub_steps = back_sub_steps
        self.device = net.device
        self._build_network_transformer()
        self._build_subnetwork_transformer()

    def _build_network_transformer(self):
        last = BatchInputTransformer(self.net.input_shape).to(self.device)
        layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                last = BatchLinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx).to(self.device)
            elif isinstance(layer, nn.ReLU):
                last = BatchReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, kwargs=self.net.layers_mapping).to(self.device)
                idx += 1
            else:
                print(layer)
                raise NotImplementedError
            layers += [last]
            # self._modules[str(last)] = last

        self.layers = nn.Sequential(*layers)


    def _build_subnetwork_transformer(self):
        self.forward_from_layer = {k: [BatchInputHiddenTransformer(sub_id=k)] for k in self.net.layers_mapping.keys()}
        for k in self.forward_from_layer:
            idx = 0
            for layer in self.net.layers:
                last = self.forward_from_layer[k][-1]
                if isinstance(layer, nn.Linear):
                    last = BatchLinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx, sub_id=k).to(self.device)
                elif isinstance(layer, nn.ReLU):
                    last = BatchReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, sub_id=k, kwargs=self.net.layers_mapping).to(self.device)
                    idx += 1
                else:
                    print(layer)
                    raise NotImplementedError

                if k < idx:
                    self.forward_from_layer[k] += [last]

        # for k in self.forward_from_layer:
        #     print(k, [str(l) for l in self.forward_from_layer[k]])

    # @torch.no_grad()
    def __call__(self, lower, upper, assignment=None, return_hidden_bounds=False):
        bounds = (lower, upper)
        hidden_bounds = []
        for layer in self.layers:
            # print('[+] processing:', layer)
            if isinstance(layer, BatchReLUTransformer):
                hidden_bounds.append(bounds.permute(1, 2, 0)) # B x 2 x H
            bounds, params = layer(bounds, assignment)
            assert torch.all(bounds[..., 0] <= bounds[..., 1])
        if return_hidden_bounds:
            return (bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)), hidden_bounds
        return bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)


    @torch.no_grad()
    def forward_layer(self, lower, upper, layer_id, assignment=None, return_hidden_bounds=False):
        bounds = (lower, upper)
        hidden_bounds = []
        for layer in self.forward_from_layer[layer_id]:
            if isinstance(layer, BatchReLUTransformer):
                hidden_bounds.append(bounds.permute(1, 2, 0)) # B x 2 x H
            bounds, _ = layer(bounds, assignment)
            assert torch.all(bounds[..., 0] <= bounds[..., 1])
        if return_hidden_bounds:
            return (bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)), hidden_bounds
        return bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)


class BatchInputHiddenTransformer(nn.Module):

    def __init__(self, sub_id=None, last=None):
        super(BatchInputHiddenTransformer, self).__init__()
        self.last = last
        self.sub_id = sub_id

    def forward(self, bounds, assignment):
        self.bounds = torch.stack([bounds[0], bounds[1]], dim=2).transpose(0, 1) # H x B x 2
        return self.bounds, None
    

    def __str__(self):
        return f'[{self.sub_id}] Input Hidden'


class BatchInputTransformer(nn.Module):

    def __init__(self, input_shape, last=None):
        super(BatchInputTransformer, self).__init__()
        self.last = last
        self.input_shape = input_shape

    def forward(self, bounds, assignment):
        # print('\t- Forward:', self)
        self.bounds = torch.stack([bounds[0], bounds[1]], dim=2).transpose(0, 1) # H x B x 2
        return self.bounds, None
    

    def __str__(self):
        return 'Input'


class BatchLinearTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None, sub_id=None):
        super(BatchLinearTransformer, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.sub_id = sub_id
        self.idx = idx
        self.params = None

    def forward(self, bounds, assignment):
        # print('\t- Forward:', self)
        # print(self, bounds.shape) # H x B x 2
        upper = self.W_plus @ bounds[..., 1] + self.W_minus @ bounds[..., 0] # H x B
        lower = self.W_plus @ bounds[..., 0] + self.W_minus @ bounds[..., 1] # H x B

        self.bounds  = torch.stack([lower, upper], dim=2) + self.bias[:, None, None] # H x B x 2
        # print(self, self.bounds.shape)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, self.params
    
    def back_sub(self, max_steps):
        # print('\t- Backward:', self)
        new_bounds, new_params = self._back_sub(max_steps) # H x B x 2
        # new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[..., 0] > self.bounds[..., 0]
        indu = new_bounds[..., 1] < self.bounds[..., 1]
        self.bounds[..., 0][indl] = new_bounds[..., 0][indl]
        self.bounds[..., 1][indu] = new_bounds[..., 1][indu]
        self.params = new_params
        
    def _back_sub(self, max_steps, params=None):
        # print('\t\t-', self, '--->', self.last)
        # print(params)
        if params is None:
            params = self.weight.data, self.weight.data, self.bias.data, self.bias.data

        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            if self.last.beta is not None:
                H, B = self.last.beta.shape
                if len(Ml.shape) == 2:
                    # print('Ml', Ml.shape)
                    Ml = Ml.repeat(B, 1, 1).permute(1, 2, 0) # H, H2, B
                    Mu = Mu.repeat(B, 1, 1).permute(1, 2, 0) # H, H2, B
                    # print('Ml after', Ml.shape)
                    # print('self.last.beta', self.last.beta.shape)
                    # print('self.lmbda.beta', self.last.lmbda.shape)
                    Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda # H, H2, B
                    Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta # H, H2, B
                    # print('Mlnew', Mlnew.shape)
                    # print('bl', bl.shape)
                    # print('self.last.mu', self.last.mu.shape)

                    blnew = torch.bmm(torch.clamp(Ml.permute(2, 0, 1), max=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                    bunew = torch.bmm(torch.clamp(Mu.permute(2, 0, 1), min=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H
                    # print(blnew.shape)
                else:
                    # print('Ml', Ml.shape)
                    # print('bl', bl.shape)
                    # print('self.last.beta', self.last.beta.shape)
                    Ml2 = Ml.permute(1, 2, 0) # H, H2, B
                    Mu2 = Mu.permute(1, 2, 0) # H, H2, B
                    Mlnew = torch.clamp(Ml2, min=0) * self.last.beta + torch.clamp(Ml2, max=0) * self.last.lmbda # H, H2, B
                    Munew = torch.clamp(Mu2, min=0) * self.last.lmbda + torch.clamp(Mu2, max=0) * self.last.beta # H, H2, B
                    # print('Mlnew', Mlnew.shape)
                    # print('self.last.mu', self.last.mu.shape)
                    blnew = torch.bmm(torch.clamp(Ml, max=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                    bunew = torch.bmm(torch.clamp(Mu, min=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H
                    # print('blnew', blnew.shape)

                Mlnew = Mlnew.permute(2, 0, 1)
                Munew = Munew.permute(2, 0, 1)
                blnew = blnew.permute(1, 0)
                bunew = bunew.permute(1, 0)

                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                return self.last._back_sub(max_steps-1, params=params)
        else:
            # print('Ml', Ml.shape)
            # print('self.last.bounds', self.last.bounds.shape)
            if len(Ml.shape) == 2: # first layer
                lower = torch.clamp(Ml, min=0) @ self.last.bounds[..., 0] + torch.clamp(Ml, max=0) @ self.last.bounds[..., 1] + bl[:, None] # H x B
                upper = torch.clamp(Mu, min=0) @ self.last.bounds[..., 1] + torch.clamp(Mu, max=0) @ self.last.bounds[..., 0] + bu[:, None] # H x B
                return torch.stack([lower, upper], dim=2), params # H x B x 2
            else:
                # print(Ml.shape)
                # print(bl.shape)
                lower = torch.bmm(torch.clamp(Ml, min=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Ml, max=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                upper = torch.bmm(torch.clamp(Mu, min=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Mu, max=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H
                # print(lower.shape)
                output = torch.stack([lower.transpose(0, 1), upper.transpose(0, 1)], dim=2) # H x B x 2
                # print(output.shape)
                return output, params # H x B x 2

    def __str__(self):
        return f'[{self.sub_id}] Linear {self.idx}'



class BatchReLUTransformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None, sub_id=None, kwargs=None):
        super(BatchReLUTransformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps

        self.sub_id = sub_id
        self.idx = idx

        self.layers_mapping = kwargs
        self.params = None

        # self.beta = None
        self.register_parameter('beta', None)

    def init_parameters(self, x):
        self.beta = nn.Parameter(torch.zeros(x.shape[0], x.shape[1], device=x.device), requires_grad=True)
        # nn.init.uniform_(self.beta)
        # print('init beta', self, self.beta.shape, self.beta)
        # print(self.beta.requires_grad)

    def forward(self, bounds, assignment):
        if self.beta is None:
            self.init_parameters(bounds)

        # print('\tbeta:', self.beta.flatten().detach().numpy().tolist())
        # print('\t- Forward:', self)
        # H x B x 2
        device = bounds.device
        ind1 = bounds[..., 1] <= 0 # H x B
        ind2 = bounds[..., 0] > 0 # H x B
        ind3 = (bounds[..., 1] > 0) * (bounds[..., 0] < 0) # H x B
        # ind4 = (bounds[1] > -bounds[0]) * ind3

        self.bounds = torch.zeros_like(bounds, device=device) # H x B x 2
        self.bounds[..., 1][ind3] = bounds[...,1][ind3]
        # self.bounds[:, ind4] = bounds[:, ind4]

        self.lmbda = torch.zeros_like(bounds[..., 1], device=device) # H x B
        self.mu = torch.zeros_like(bounds[..., 1], device=device) # H x B
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2], device=device)

        self.beta.data[ind1] = torch.zeros_like(self.beta.data[ind1], device=device) # note: inactive indices should be set to 0
        self.beta.data[ind2] = torch.ones_like(self.beta.data[ind2], device=device) # note: active indices should be set to 1

        diff = bounds[..., 1][ind3] - bounds[..., 0][ind3] 
        self.lmbda[ind3] = torch.div(bounds[..., 1][ind3], diff)
        # self.beta[ind4] = torch.ones_like(self.beta[ind4])
        self.mu[ind3] = torch.div(-bounds[..., 0][ind3] * bounds[..., 1][ind3], diff)
        self.bounds[..., :][ind2] = bounds[..., :][ind2]

        # if (self.beta[ind1].sum() != 0):
        #     print(self.beta.detach()[ind1].numpy())
        #     raise

        self.zero_grad_indices = (ind1, ind2)

        # print( self.beta[ind2])
        # print(self.beta[ind2].shape)
        if assignment is not None:
            assert len(assignment) == bounds.shape[1]
            la = torch.stack([torch.Tensor([a.get(i, 2) for i in self.layers_mapping[self.idx]]) for a in assignment], dim=-1)
            active_ind = la==True
            inactive_ind = la==False

            self.lmbda[active_ind] = torch.ones_like(self.lmbda[active_ind], device=device)
            self.beta.data[active_ind] = torch.zeros_like(self.beta.data[active_ind], device=device)
            self.mu[active_ind] = torch.zeros_like(self.mu[active_ind], device=device)

            self.lmbda[inactive_ind] = torch.zeros_like(self.lmbda[inactive_ind], device=device)
            self.beta.data[inactive_ind] = torch.zeros_like(self.beta.data[inactive_ind], device=device)
            self.mu[inactive_ind] = torch.zeros_like(self.mu[inactive_ind], device=device)

            self.bounds[..., :][inactive_ind] = torch.zeros_like(self.bounds[..., :][inactive_ind], device=device)
            self.zero_grad_indices += (active_ind, inactive_ind)

        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, None

    def __str__(self):
        return f'[{self.sub_id}] Relu {self.idx}'

    def back_sub(self, max_steps):
        # print('\t- Backward:', self)
        # new_bounds, new_params = self._back_sub(max_steps)
        # new_bounds = new_bounds.reshape(self.bounds.shape)
        # indl = new_bounds[0] > self.bounds[0] + 1e-6
        # indu = new_bounds[1] < self.bounds[1] - 1e-6
        # print(indl.sum(), indu.sum())
        # print(self.bounds[0, indl], new_bounds[0, indl])
        # self.bounds[0, indl] = new_bounds[0, indl]
        # self.bounds[1, indu] = new_bounds[1, indu]
        # self.params = new_params

        new_bounds, new_params = self._back_sub(max_steps) # H x B x 2
        # new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[..., 0] > self.bounds[..., 0]
        indu = new_bounds[..., 1] < self.bounds[..., 1]
        self.bounds[..., 0][indl] = new_bounds[..., 0][indl]
        self.bounds[..., 1][indu] = new_bounds[..., 1][indu]
        self.params = new_params

    @staticmethod
    @torch.jit.script
    def clamp_(self, W, b, b0, b1):
        return W.clamp(min=0) @ b0 + W.clamp(max=0) @ b1 + b
         

    def _back_sub(self, max_steps, params=None):
        # print('\t\t-', self, '--->', self.last)
    
        if params is None:
            device = self.lmbda.device
            params = torch.diag_embed(self.beta.transpose(0, 1)), torch.diag_embed(self.lmbda.transpose(0, 1)), torch.zeros_like(self.mu, device=device), self.mu
        Ml, Mu, bl, bu = params
        # print(Ml.shape)
        # print(bl.shape)
        # exit()
        if max_steps > 0 and self.last.last is not None:
            Mlnew = Ml @ self.last.weight # B x H x H2
            # print('---', Ml.shape, self.last.weight.shape, bl.shape)
            # print('---', (Ml @ self.last.bias).shape, bl.shape)
            # exit()
            Munew = Mu @ self.last.weight # B x H x H2
            # print(Munew.shape)
            blnew = bl.transpose(0, 1) + (Ml @ self.last.bias) # B x H
            bunew = bu.transpose(0, 1) + (Mu @ self.last.bias) # B x H
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            # print(self.last.bounds.shape) # H x B x 2
            # print('Ml', Ml.shape) # B x H x H
            # print('bl', bl.shape) # H x B
            # print('Mu', Mu.shape) # B x H x H
            # print('bu', bu.shape) # H x B
            # lower = self.clamp_(Ml, bl, self.last.bounds[0], self.last.bounds[1])
            # upper = self.clamp_(Mu, bu, self.last.bounds[1], self.last.bounds[0])
            lower = torch.bmm(torch.clamp(Ml, min=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Ml, max=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl.transpose(0, 1)
            # lower = torch.clamp(Ml, min=0) @ self.last.bounds[..., 0] + torch.clamp(Ml, max=0) @ self.last.bounds[..., 1] + bl
            # upper = torch.clamp(Mu, min=0) @ self.last.bounds[..., 1] + torch.clamp(Mu, max=0) @ self.last.bounds[..., 0] + bu
            upper = torch.bmm(torch.clamp(Mu, min=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Mu, max=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu.transpose(0, 1)
            # print((torch.clamp(Ml, min=0) @ self.last.bounds[..., 0]).shape)
            output = torch.stack([lower.transpose(0, 1), upper.transpose(0, 1)], dim=2) # H x B x 2
            # print(output.shape)
            return output, params



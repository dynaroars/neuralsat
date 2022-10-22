from torch.autograd.functional import jacobian
import torch.nn as nn
import torch

class DeepPoly:

    def __init__(self, net, back_sub_steps=0):
        self.net = net
        self.back_sub_steps = back_sub_steps

        self.device = net.device

        self._build_network_transformer()
        self._build_subnetwork_transformer()

    def _build_network_transformer(self):
        last = InputTransformer(self.net.input_shape).to(self.device)
        self.layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                last = LinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx).to(self.device)
            elif isinstance(layer, nn.ReLU):
                last = ReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, kwargs=self.net.layers_mapping).to(self.device)
                idx += 1
            elif isinstance(layer, nn.Flatten):
                last = FlattenTransformer(last=last, idx=idx).to(self.device)
            else:
                print(layer)
                raise NotImplementedError
            self.layers += [last]

    def _build_subnetwork_transformer(self):
        self.forward_from_layer = {k: [InputHiddenTransformer(sub_id=k)] for k in self.net.layers_mapping.keys()}
        
        for k in self.forward_from_layer:
            idx = 0
            for layer in self.net.layers:
                last = self.forward_from_layer[k][-1]
                if isinstance(layer, nn.Linear):
                    last = LinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx, sub_id=k).to(self.device)
                elif isinstance(layer, nn.ReLU):
                    last = ReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, sub_id=k, kwargs=self.net.layers_mapping).to(self.device)
                    idx += 1
                elif isinstance(layer, nn.Flatten):
                    last = FlattenTransformer(last=last, idx=idx, sub_id=k).to(self.device)
                else:
                    print(layer)
                    raise NotImplementedError

                if k < idx:
                    self.forward_from_layer[k] += [last]

        # for k, v in self.forward_from_layer.items():
        #     print(k, end=' ')
        #     for _ in v:
        #         print(_, end=' -> ')
        #     print('out')

        # exit()
    
    @torch.no_grad()
    def forward_layer(self, lower, upper, layer_id, assignment=None):
        bounds = (lower, upper)
        hidden_bounds = []
        for layer in self.forward_from_layer[layer_id]:
            # print(layer, layer.last)
            if isinstance(layer, ReLUTransformer):
                hidden_bounds.append((bounds[0].squeeze(), bounds[1].squeeze()))
            
            bounds, _ = layer(bounds, assignment)
            # print('weight', layer.weight if hasattr(layer, 'weight') else None)
            # print('\t', bounds[0])
            # print('\t', bounds[1])
        #     print(layer, end=' -> ')
        # print('out')
        return (bounds[0], bounds[1]), hidden_bounds

        # exit()

    @torch.no_grad()
    def __call__(self, lower, upper, assignment=None, return_params=False):
        bounds = (lower, upper)
        hidden_bounds = []
        hidden_params = []
        for layer in self.layers:
            # print(layer)
            if isinstance(layer, ReLUTransformer):
                hidden_bounds.append((bounds[0].squeeze(), bounds[1].squeeze()))

            bounds, params = layer(bounds, assignment)
            # print(layer)
            # print('weight', layer.weight if hasattr(layer, 'weight') else None)
            # print('\t', bounds[0])
            # print('\t', bounds[1])
            assert torch.all(bounds[0] <= bounds[1]), layer

            if params is not None:
                hidden_params.append(params)
        if return_params:
            return (bounds[0], bounds[1]), hidden_bounds, hidden_params
        return (bounds[0], bounds[1]), hidden_bounds

    def get_params(self):
        return self.layers[-1].params    


class InputTransformer(nn.Module):

    def __init__(self, input_shape, last=None):
        super(InputTransformer, self).__init__()
        self.last = last
        self.input_shape = input_shape

    def forward(self, bounds, assignment):
        self.bounds = torch.stack([bounds[0].view(self.input_shape[1:]), bounds[1].view(self.input_shape[1:])], 0).squeeze()
        return self.bounds, None
    

    def __str__(self):
        return 'Input'



class InputHiddenTransformer(nn.Module):

    def __init__(self, sub_id=None, last=None):
        super(InputHiddenTransformer, self).__init__()
        self.last = last
        self.sub_id = sub_id

    def forward(self, bounds, assignment):
        self.bounds = torch.stack([bounds[0], bounds[1]], 0).squeeze()
        return self.bounds, None
    

    def __str__(self):
        return f'[{self.sub_id}] Input Hidden'


class LinearTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None, sub_id=None):
        super(LinearTransformer, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.sub_id = sub_id
        self.idx = idx
        self.params = None

    def forward(self, bounds, assignment):
        upper = self.W_plus @ bounds[1] + self.W_minus @ bounds[0]
        lower = self.W_plus @ bounds[0] + self.W_minus @ bounds[1]
        self.bounds = torch.stack([lower, upper], 0) + self.bias.view(1, -1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, self.params
    
    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params
        
    def _back_sub(self, max_steps, params=None):
        # print('_back_sub', self, '--->', self.last)
        if params is None:
            params = self.weight.data, self.weight.data, self.bias.data, self.bias.data

        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            if self.last.beta is not None:
                Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda
                Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta
                blnew = bl + torch.clamp(Ml, max=0) @ self.last.mu
                bunew = bu + torch.clamp(Mu, min=0) @ self.last.mu
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                return self.last._back_sub(max_steps-1, params=params)
        else:
            # print(Ml.shape)
            # print(self.last.bounds[0].shape)
            lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
            upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
            return torch.stack([lower, upper], 0), params

    def __str__(self):
        return f'[{self.sub_id}] Linear {self.idx}'



class ReLUTransformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None, sub_id=None, kwargs=None):
        super(ReLUTransformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.last_conv_flag = False #isinstance(self.last, Conv2dTransformer)

        self.sub_id = sub_id
        self.idx = idx

        self.layers_mapping = kwargs
        self.params = None
    
    def forward(self, bounds, assignment):
        device = bounds.device
        ind2 = bounds[0] >= 0 
        ind3 = (bounds[1] > 0) * (bounds[0] < 0) 
        ind4 = (bounds[1] > -bounds[0]) * ind3

        self.bounds = torch.zeros_like(bounds, device=device)
        self.bounds[1, ind3] = bounds[1, ind3]
        self.bounds[:, ind4] = bounds[:, ind4]
        self.lmbda = torch.zeros_like(bounds[1], device=device)
        self.beta = torch.zeros_like(bounds[1], device=device)
        self.mu = torch.zeros_like(bounds[1], device=device)
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2], device=device)

        diff = bounds[1, ind3] - bounds[0, ind3] 
        self.lmbda[ind3] = torch.div(bounds[1, ind3], diff)
        self.beta[ind4] = torch.ones_like(self.beta[ind4])
        self.mu[ind3] = torch.div(-bounds[0, ind3] * bounds[1, ind3], diff)
        self.bounds[:, ind2] = bounds[:, ind2]
        self.beta[ind2] = torch.ones_like(self.beta[ind2], device=device)
        if assignment is not None:
            la = torch.Tensor([assignment.get(i, 2) for i in self.layers_mapping[self.idx]]).view(self.lmbda.shape)
            active_ind = la==True
            inactive_ind = la==False

            self.lmbda[active_ind] = torch.ones_like(self.lmbda[active_ind], device=device)
            self.beta[active_ind] = torch.zeros_like(self.beta[active_ind], device=device)
            self.mu[active_ind] = torch.zeros_like(self.mu[active_ind], device=device)

            self.lmbda[inactive_ind] = torch.zeros_like(self.lmbda[inactive_ind], device=device)
            self.beta[inactive_ind] = torch.zeros_like(self.beta[inactive_ind], device=device)
            self.mu[inactive_ind] = torch.zeros_like(self.mu[inactive_ind], device=device)

            self.bounds[:, inactive_ind] = torch.zeros_like(self.bounds[:, inactive_ind], device=device)

        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, None

    def __str__(self):
        return f'[{self.sub_id}] Relu {self.idx}'

    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        # self.params = new_params

    def _back_sub(self, max_steps, params=None):
        # print('_back_sub', self, '--->', self.last)
        if self.last_conv_flag:
            if params is None:
                params = torch.diag(self.beta.flatten()), torch.diag(self.lmbda.flatten()), torch.zeros_like(self.mu).flatten(), self.mu.flatten()
            Ml, Mu, bl, bu = params
            if max_steps > 0 and self.last.last is not None:
                Mlnew = Ml @ self.last.weights_backsub
                Munew = Mu @ self.last.weights_backsub
                blnew = bl + (Ml @ self.last.bias_backsub).flatten()
                bunew = bu + (Mu @ self.last.bias_backsub).flatten()
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                lower = (torch.clamp(Ml, min=0) @ self.last.bounds[0].flatten() + torch.clamp(Ml, max=0) @ self.last.bounds[1].flatten()) + bl
                upper = (torch.clamp(Mu, min=0) @ self.last.bounds[1].flatten() + torch.clamp(Mu, max=0) @ self.last.bounds[0].flatten()) + bu
                return torch.cat([lower, upper], 0), params
        else:
            if params is None:
                device = self.beta.device
                params = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu, device=device), self.mu
            Ml, Mu, bl, bu = params

            if max_steps > 0 and self.last.last is not None:
                Mlnew = Ml @ self.last.weight
                Munew = Mu @ self.last.weight 
                blnew = bl + Ml @ self.last.bias
                bunew = bu + Mu @ self.last.bias
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
                upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
                # lower = Ml @ self.last.bounds[0] + bl
                # upper = Mu @ self.last.bounds[1] + bu
                return torch.stack([lower, upper], 0), params


class FlattenTransformer(nn.Module):
    def __init__(self, last=None, idx=None, sub_id=None):
        super(FlattenTransformer, self).__init__()
        self.last = last
        self.sub_id = sub_id
        self.idx = idx

    def forward(self, bounds, assignment):
        b = torch.stack([bounds[0].flatten(), bounds[1].flatten()], 0)
        return b, None
    
    def _back_sub(self, max_steps, params=None):
        # print('_back_sub', self, '--->', self.last)
        Ml, Mu, bl, bu = params
        if self.last.last is not None:
            bounds, params = self.last._back_sub(max_steps, params=params)
        else:
            lower = (torch.clamp(Ml, min=0) @ self.last.bounds[0].flatten() + torch.clamp(Ml, max=0) @ self.last.bounds[1].flatten()).flatten() + bl
            upper = (torch.clamp(Mu, min=0) @ self.last.bounds[1].flatten() + torch.clamp(Mu, max=0) @ self.last.bounds[0].flatten()).flatten() + bu
            bounds = torch.stack([lower, upper], 0)
        return bounds, params
    
    @property
    def beta(self):
        if hasattr(self.last, 'beta'):
            return self.last.beta.flatten()
        return None


    @property
    def mu(self):
        if hasattr(self.last, 'mu'):
            return self.last.mu.flatten()
        return None

    @property
    def lmbda(self):
        if hasattr(self.last, 'lmbda'):
            return self.last.lmbda.flatten()
        return None

    @property
    def bounds(self):
        return self.last.bounds

    def __str__(self):
        return f'[{self.sub_id}] Flatten {self.idx}'

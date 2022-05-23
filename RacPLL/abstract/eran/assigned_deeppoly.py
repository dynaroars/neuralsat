from torch.autograd.functional import jacobian
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import numpy as np

from utils.read_nnet import NetworkNNET


class AssignedDeepPoly:

    def __init__(self, net, back_sub_steps=0):
        self.net = net
        self.back_sub_steps = back_sub_steps

        self._build_network_transformer()

    def _build_network_transformer(self):
        last = AssignedDeepPolyInputTransformer(self.net.input_shape)
        self.layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                last = AssignedDeepPolyAffineTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
                self.layers += [last]
            elif isinstance(layer, nn.ReLU):
                last = AssignedDeepPolyReLUTansformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, kwargs=self.net.layers_mapping)
                idx += 1
                self.layers += [last]
            elif isinstance(layer, nn.Conv2d):
                last = AssignedDeepPolyConvTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
                self.layers += [last]
            elif isinstance(layer, nn.Flatten):
                last = AssignedDeepPolyFlattenTransformer(last=last, idx=idx)
                self.layers += [last]
            else:
                raise NotImplementedError
    
    @torch.no_grad()
    def __call__(self, lower, upper, assignment=None, return_params=False):
        bounds = (lower, upper)
        hidden_bounds = []
        hidden_params = []
        for layer in self.layers:
            if isinstance(layer, AssignedDeepPolyReLUTansformer):
                hidden_bounds.append((bounds[0].squeeze(), bounds[1].squeeze()))
            bounds, params = layer(bounds, assignment)
            if params is not None:
                hidden_params.append(params)
        self.bounds = bounds
        if return_params:
            return (self.bounds[0], self.bounds[1]), hidden_bounds, hidden_params
        return (self.bounds[0], self.bounds[1]), hidden_bounds

    def get_params(self):
        return self.layers[-1].params    


class AssignedDeepPolyInputTransformer(nn.Module):
    def __init__(self, input_shape, last=None):
        super(AssignedDeepPolyInputTransformer, self).__init__()
        self.last = last
        self.input_shape = input_shape

    def forward(self, bounds, assignment):
        self.bounds = torch.stack([bounds[0].view(self.input_shape[1:]), bounds[1].view(self.input_shape[1:])], 0)
        return self.bounds, None
    
    def __str__(self):
        return 'Input'


class AssignedDeepPolyAffineTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None):
        super(AssignedDeepPolyAffineTransformer, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.idx = idx

    def forward(self, bounds, assignment):
        upper = self.W_plus @ bounds[1] + self.W_minus @ bounds[0]
        lower = self.W_plus @ bounds[0] + self.W_minus @ bounds[1]
        self.bounds = torch.stack([lower, upper], 0) + self.bias.view(1, -1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, self.params
    
    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params
        
    def _back_sub(self, max_steps, params=None):
        if params is None:
            params = self.weight.data, self.weight.data, self.bias.data, self.bias.data

        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta
            blnew = bl + torch.clamp(Ml, max=0) @ self.last.mu
            bunew = bu + torch.clamp(Mu, min=0) @ self.last.mu
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
            upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
            return torch.stack([lower, upper], 0), params

    def __str__(self):
        return f'Linear {self.idx}'



class AssignedDeepPolyReLUTansformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None, kwargs=None):
        super(AssignedDeepPolyReLUTansformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.last_conv_flag = isinstance(self.last, AssignedDeepPolyConvTransformer)

        self.idx = idx

        self.layers_mapping = kwargs
    
    def forward(self, bounds, assignment):
        ind2 = bounds[0] >= 0 
        ind3 = (bounds[1] > 0) * (bounds[0] < 0) 
        # ind4 = (bounds[1] > -bounds[0]) * ind3

        self.bounds = torch.zeros_like(bounds)
        self.bounds[1, ind3] = bounds[1, ind3]
        # self.bounds[:, ind4] = bounds[:, ind4]
        self.lmbda = torch.zeros_like(bounds[1])
        self.beta = torch.zeros_like(bounds[1])
        self.mu = torch.zeros_like(bounds[1])
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2])

        diff = bounds[1, ind3] - bounds[0, ind3] 
        self.lmbda[ind3] = torch.div(bounds[1, ind3], diff)
        # self.beta[ind4] = torch.ones_like(self.beta[ind4])
        self.mu[ind3] = torch.div(-bounds[0, ind3] * bounds[1, ind3], diff)
        self.bounds[:, ind2] = bounds[:, ind2]
        self.beta[ind2] = torch.ones_like(self.beta[ind2])
        if assignment is not None:
            la = torch.Tensor([assignment.get(i, 2) for i in self.layers_mapping[self.idx]]).view(self.lmbda.shape)
            active_ind = la==True
            inactive_ind = la==False

            self.lmbda[active_ind] = torch.ones_like(self.lmbda[active_ind])
            self.beta[active_ind] = torch.ones_like(self.beta[active_ind])
            self.mu[active_ind] = torch.zeros_like(self.mu[active_ind])

            self.lmbda[inactive_ind] = torch.zeros_like(self.lmbda[inactive_ind])
            self.beta[inactive_ind] = torch.zeros_like(self.beta[inactive_ind])
            self.mu[inactive_ind] = torch.zeros_like(self.mu[inactive_ind])

            self.bounds[:, inactive_ind] = torch.zeros_like(self.bounds[:, inactive_ind])

        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, None

    def __str__(self):
        return f'Relu {self.idx}'

    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params

    def _back_sub(self, max_steps, params=None):
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
                params = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu), self.mu
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
                return torch.stack([lower, upper], 0), params


class AssignedDeepPolyFlattenTransformer(nn.Module):
    def __init__(self, last=None, idx=None):
        super(AssignedDeepPolyFlattenTransformer, self).__init__()
        self.last = last
        self.idx = idx

    def forward(self, bounds, assignment):
        b = torch.stack([bounds[0].flatten(), bounds[1].flatten()], 0)
        return b, None
    
    def _back_sub(self, max_steps, params=None):
        bounds, _ = self.last._back_sub(max_steps, params=params)
        bounds = torch.stack([bounds[:len(bounds)//2], bounds[len(bounds)//2:]], 0)
        return bounds, None
    
    @property
    def beta(self):
        return self.last.beta.flatten()

    @property
    def mu(self):
        return self.last.mu.flatten()

    @property
    def lmbda(self):
        return self.last.lmbda.flatten()

    def __str__(self):
        return f'Flatten {self.idx}'
        
class ReshapeConv(torch.nn.Module):

    def __init__(self, in_dim_1, in_dim_2, in_channels, layer):
        super(ReshapeConv, self).__init__()

        self.in_dim_1 = in_dim_1
        self.in_dim_2 = in_dim_2
        self.in_channels = in_channels
        self.layer = layer

    def forward(self, x):
        out = self.layer(x.view(1, self.in_channels, self.in_dim_1, self.in_dim_2))
        return torch.flatten(out)

class AssignedDeepPolyConvTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None):
        super(AssignedDeepPolyConvTransformer, self).__init__()

        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.weight = layer.weight
        self.bias = layer.bias

        self.W_plus = torch.clamp(self.weight, min=0)
        self.conv_plus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_plus.weight = nn.Parameter(self.W_plus)
        self.conv_plus.bias = nn.Parameter(self.bias/2.)

        self.W_minus = torch.clamp(self.weight, max=0)
        self.conv_minus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_minus.weight = nn.Parameter(self.W_minus)
        self.conv_minus.bias = nn.Parameter(self.bias/2.)
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv.weight = nn.Parameter(self.weight)
        self.conv.bias = None

        self.weights_backsub = None
        self.bias_backsub = None

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.idx = idx
    
    def toeplitz_convmatrix2d(self):
        inputs = torch.ones_like(self.last.bounds[1].flatten())
        _, C, H, W = self.last.bounds.shape
        reshape_conv = ReshapeConv(H, W, self.in_channels, self.conv)
        ## hacky but works: find toeplitz by jacobian
        j = jacobian(reshape_conv, inputs)
        return j

    def forward(self, bounds, assignment):
        if isinstance(self.weights_backsub, type(None)):
            self.weights_backsub = self.toeplitz_convmatrix2d()
        bounds = bounds.unsqueeze(1)
        upper = self.conv_plus(bounds[1]) + self.conv_minus(bounds[0])
        lower = self.conv_plus(bounds[0]) + self.conv_minus(bounds[1])
        self.bounds = torch.stack([lower, upper], 0).squeeze(1)
        if isinstance(self.bias_backsub, type(None)):
            self.bias_backsub = self.bias.repeat_interleave(self.bounds[1].shape[1]*self.bounds[1].shape[2])
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
        if params is None:
            params = self.weights_backsub, self.weights_backsub, self.bias_backsub, self.bias_backsub
            
        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta.flatten() + torch.clamp(Ml, max=0)* self.last.lmbda.flatten()
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda.flatten() + torch.clamp(Mu, max=0)* self.last.beta.flatten()
            blnew = bl + torch.clamp(Ml, max=0) @ self.last.mu.flatten()
            bunew = bu + torch.clamp(Mu, min=0) @ self.last.mu.flatten()
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = (torch.clamp(Ml, min=0) @ self.last.bounds[0].flatten() + torch.clamp(Ml, max=0) @ self.last.bounds[1].flatten()).flatten() + bl
            upper = (torch.clamp(Mu, min=0) @ self.last.bounds[1].flatten() + torch.clamp(Mu, max=0) @ self.last.bounds[0].flatten()).flatten() + bu
            return torch.cat([lower, upper], 0), params

    def __str__(self):
        return f'Conv2d {self.idx}'

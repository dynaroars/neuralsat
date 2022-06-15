from torch.autograd.functional import jacobian
import torch.nn as nn
import onnx2pytorch
import torch
import math

class DeepPoly:

    def __init__(self, net, back_sub_steps=0):
        self.net = net
        self.back_sub_steps = back_sub_steps

        self._build_network_transformer()

    def _build_network_transformer(self):
        last = InputTransformer(self.net.input_shape)
        self.layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                last = LinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
            elif isinstance(layer, nn.ReLU):
                last = ReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, kwargs=self.net.layers_mapping)
                idx += 1
            elif isinstance(layer, nn.Conv2d):
                last = Conv2dTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
            elif isinstance(layer, nn.Flatten):
                last = FlattenTransformer(last=last, idx=idx)
            elif isinstance(layer, onnx2pytorch.operations.Reshape):
                last = ReshapeTransformer(layer.shape, last=last, idx=idx, back_sub_steps=self.back_sub_steps)
            elif isinstance(layer, onnx2pytorch.operations.Transpose):
                last = TransposeTransformer(layer.dims, last=last, idx=idx, back_sub_steps=self.back_sub_steps)
            else:
                print(layer)
                raise NotImplementedError
            self.layers += [last]
    
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
            # print('\t', bounds.shape, math.prod([*bounds.shape]))
            assert torch.all(bounds[0] <= bounds[1])

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



class TransposeTransformer(nn.Module):
    def __init__(self, dims, last=None, idx=None, back_sub_steps=None):
        super(TransposeTransformer, self).__init__()
        self.last = last
        self.idx = idx
        self.dims = dims

    def forward(self, bounds, assignment):
        self.bounds = bounds.permute(self.dims)
        return self.bounds, None
    
    def _back_sub(self, max_steps, params=None):
        Ml, Mu, bl, bu = params
        if self.last.last is not None:
            bounds, params = self.last._back_sub(max_steps, params=params)
        else:
            last_bounds = self.last.bounds.permute(self.dims)
            lower = (torch.clamp(Ml, min=0) @ last_bounds[0].flatten() + torch.clamp(Ml, max=0) @ last_bounds[1].flatten()).flatten() + bl
            upper = (torch.clamp(Mu, min=0) @ last_bounds[1].flatten() + torch.clamp(Mu, max=0) @ last_bounds[0].flatten()).flatten() + bu
            bounds = torch.cat([lower, upper], 0)
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


    def __str__(self):
        return f'Transpose {self.idx}'


class ReshapeTransformer(nn.Module):
    def __init__(self, shape, last=None, idx=None, back_sub_steps=None):
        super(ReshapeTransformer, self).__init__()
        self.last = last
        self.idx = idx
        self.shape = [i for i in shape]
        self.back_sub_steps = back_sub_steps

    def forward(self, bounds, assignment):
        self.bounds = bounds
        return self.bounds.reshape(self.shape), None
    
    def _back_sub(self, max_steps, params=None):
        # print('_back_sub', self, '--->', self.last)
        bounds, params = self.last._back_sub(max_steps, params=params)
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


    def __str__(self):
        return f'Reshape {self.idx}'


class LinearTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None):
        super(LinearTransformer, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
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
            lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
            upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
            return torch.stack([lower, upper], 0), params

    def __str__(self):
        return f'Linear {self.idx}'



class ReLUTransformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None, kwargs=None):
        super(ReLUTransformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.last_conv_flag = isinstance(self.last, Conv2dTransformer)

        self.idx = idx

        self.layers_mapping = kwargs
        self.params = None
    
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
        # self.params = new_params

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


class FlattenTransformer(nn.Module):
    def __init__(self, last=None, idx=None):
        super(FlattenTransformer, self).__init__()
        self.last = last
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

class Conv2dTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None):
        super(Conv2dTransformer, self).__init__()

        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.padding = layer.padding
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.weight = layer.weight
        self.bias = layer.bias

        self.W_plus = torch.clamp(self.weight, min=0)
        self.conv_plus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_plus.weight.data = self.W_plus
        self.conv_plus.bias.data = self.bias/2.

        self.W_minus = torch.clamp(self.weight, max=0)
        self.conv_minus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_minus.weight.data = self.W_minus
        self.conv_minus.bias.data = self.bias/2.
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv.weight.data = self.weight
        self.conv.bias = None

        self.weights_backsub = None
        self.bias_backsub = None

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.idx = idx
        self.params = None
    
    def toeplitz_convmatrix2d(self):
        inputs = torch.ones_like(self.last.bounds[1].flatten())
        _, C, H, W = self.last.bounds.shape
        reshape_conv = ReshapeConv(H, W, self.in_channels, self.conv)
        ## hacky but works: find toeplitz by jacobian
        j = jacobian(reshape_conv, inputs)
        return j

    def forward(self, bounds, assignment):
        if isinstance(self.weights_backsub, type(None)) and self.back_sub_steps > 0:
            self.weights_backsub = self.toeplitz_convmatrix2d()
        bounds = bounds.unsqueeze(1)
        upper = self.conv_plus(bounds[1]) + self.conv_minus(bounds[0])
        lower = self.conv_plus(bounds[0]) + self.conv_minus(bounds[1])
        self.bounds = torch.stack([lower, upper], 0).squeeze(1)
        if isinstance(self.bias_backsub, type(None)) and self.back_sub_steps > 0:
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
        # print('_back_sub', self, '--->', self.last)
        if params is None:
            params = self.weights_backsub, self.weights_backsub, self.bias_backsub, self.bias_backsub
            
        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            if self.last.beta is not None:
                Mlnew = torch.clamp(Ml, min=0) * self.last.beta.flatten() + torch.clamp(Ml, max=0)* self.last.lmbda.flatten()
                Munew = torch.clamp(Mu, min=0) * self.last.lmbda.flatten() + torch.clamp(Mu, max=0)* self.last.beta.flatten()
                blnew = bl + torch.clamp(Ml, max=0) @ self.last.mu.flatten()
                bunew = bu + torch.clamp(Mu, min=0) @ self.last.mu.flatten()
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                return self.last._back_sub(max_steps-1, params=params)

        else:
            lower = (torch.clamp(Ml, min=0) @ self.last.bounds[0].flatten() + torch.clamp(Ml, max=0) @ self.last.bounds[1].flatten()).flatten() + bl
            upper = (torch.clamp(Mu, min=0) @ self.last.bounds[1].flatten() + torch.clamp(Mu, max=0) @ self.last.bounds[0].flatten()).flatten() + bu
            return torch.cat([lower, upper], 0), params

    def __str__(self):
        return f'Conv2d {self.idx}'

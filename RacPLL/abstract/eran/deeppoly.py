import torch.nn.functional as F
import torch.nn as nn
import torch

from typing import Tuple
from utils.read_nnet import NetworkNNET

class CorinaNet(nn.Module):

    def __init__(self):
        super().__init__()

        fc1 = nn.Linear(2, 2)
        fc1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, 1]]))
        fc1.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        fc2 = nn.Linear(2, 2)
        fc2.weight = torch.nn.Parameter(torch.Tensor([[0.5, -0.2], [-0.5, 0.1]]))
        fc2.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        fc3 = nn.Linear(2, 2)
        fc3.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [-1, 1]]))
        fc3.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        self.layers = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)


    def forward(self, x):
        return self.layers(x)


class FC(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        layers = []
        prev_size = input_size
        for idx, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            if idx < len(hidden_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeepPoly:

    def __init__(self, net, back_sub_steps=0):
        self.net = net
        self.back_sub_steps = back_sub_steps
        self.transformer = self._build_network_transformer()

    def _build_network_transformer(self):
        last = DeepPolyInputTransformer()
        layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                idx += 1
                last = DeepPolyAffineTransformer(layer.weight, layer.bias, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
                layers += [last]
            elif isinstance(layer, torch.nn.ReLU):
                last = DeepPolyReLUTansformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx)
                layers += [last]
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def __call__(self, lower, upper):
        self.bounds = self.transformer((lower, upper))
        return self.bounds[0], self.bounds[1]

    def get_params(self):
        return self.transformer[-1].params    

class DeepPolyInputTransformer(nn.Module):
    def __init__(self, last=None):
        super(DeepPolyInputTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        self.bounds = torch.stack([bounds[0], bounds[1]], 0)
        return self.bounds
    
    def __str__(self):
        return 'Input'


class DeepPolyAffineTransformer(nn.Module):

    def __init__(self, weight, bias=None, last=None, back_sub_steps=0, idx=None):
        super(DeepPolyAffineTransformer, self).__init__()
        self.weight = weight
        self.bias = bias
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.idx = idx

    def forward(self, bounds):
        upper = torch.matmul(self.W_plus, bounds[1]) + torch.matmul(self.W_minus, bounds[0])
        lower = torch.matmul(self.W_plus, bounds[0]) + torch.matmul(self.W_minus, bounds[1])
        self.bounds = torch.stack([lower, upper], 0) + self.bias.reshape(1, -1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds
    
    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params
        
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if params is None:
            Ml, Mu, bl, bu = self.weight, self.weight, self.bias, self.bias
        else:
            Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta
            blnew = bl + torch.matmul(torch.clamp(Ml, max=0), self.last.mu)
            bunew = bu + torch.matmul(torch.clamp(Mu, min=0), self.last.mu)
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[0]) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[1]) + bl
            upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[1]) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[0]) + bu
            return torch.stack([lower, upper], 0), params

    def __str__(self):
        return f'Linear {self.idx}'



class DeepPolyReLUTansformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None):
        super(DeepPolyReLUTansformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.idx = idx
    
    def forward(self, bounds):
        ind2 = bounds[0]>=0 
        ind3 = (bounds[1]>0) * (bounds[0]<0) 
        ind4 = (bounds[1] > -bounds[0]) * ind3 * False
        self.bounds = torch.zeros_like(bounds)
        self.bounds[1, ind3] = bounds[1, ind3]
        self.bounds[:, ind4] = bounds[:, ind4]
        self.lmbda = torch.zeros_like(bounds[1])
        self.beta = torch.zeros_like(bounds[1])
        self.mu = torch.zeros_like(bounds[1])
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2])

        diff = bounds[1, ind3] - bounds[0, ind3] 
        self.lmbda[ind3] = torch.div(bounds[1, ind3], diff)
        self.beta[ind4] = torch.ones_like(self.beta[ind4])
        self.mu[ind3] = torch.div(-bounds[0, ind3] * bounds[1, ind3], diff)
        self.bounds[:, ind2] = bounds[:, ind2]
        self.beta[ind2] = torch.ones_like(self.beta[ind2])
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds

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

    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if params is None:
            Ml, Mu, bl, bu = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu), self.mu
        else:
            Ml, Mu, bl, bu = params

        Mlnew = torch.matmul(Ml, self.last.weight)
        Munew = torch.matmul(Mu, self.last.weight) 
        blnew = bl + torch.matmul(Ml, self.last.bias)
        bunew = bu + torch.matmul(Mu, self.last.bias)
        
        return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))


if __name__ == '__main__':
    torch.manual_seed(1)


    # net = FC(4, [3, 4, 20, 3]).eval()
    # lower = torch.Tensor([-0.4, -0.5, -0.4, 0.2])
    # upper = torch.Tensor([0.6, 0.7, 0.6, 0.4])

    # d = DeepPoly(net, back_sub_steps=10)
    # l, u = d(lower, upper)
    # print(l)
    # print(u)

    # net = CorinaNet().eval()
    # lower = torch.Tensor([2, 1])
    # upper = torch.Tensor([3, 4])


    net = NetworkNNET('example/random.nnet')

    # net = CorinaNet().eval()
    lower = torch.Tensor([-2, 1])
    upper = torch.Tensor([2, 3])

    d = DeepPoly(net, back_sub_steps=10)
    l, u = d(lower, upper)
    print(l)
    print(u)

    # Ml, Mu, bl, bu = d.get_params()
    # print('Ml', Ml)
    # print('Mu', Mu)
    # print('bl', bl)
    # print('bu', bu)

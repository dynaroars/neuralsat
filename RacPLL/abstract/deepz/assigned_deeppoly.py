import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import numpy as np

from typing import Tuple

from utils.read_nnet import NetworkTorch
from dnn_solver.utils import InputParser


class AssignedDeepPoly:

    def __init__(self, net, back_sub_steps=0):
        self.net = net
        self.back_sub_steps = back_sub_steps

        self.vars_mapping, self.layers_mapping = InputParser.parse(net)

        self._build_network_transformer()

    def _build_network_transformer(self):
        last = AssignedDeepPolyInputTransformer()
        self.layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                last = AssignedDeepPolyAffineTransformer(layer.weight, layer.bias, last=last, back_sub_steps=self.back_sub_steps, idx=idx)
                self.layers += [last]
            elif isinstance(layer, torch.nn.ReLU):
                last = AssignedDeepPolyReLUTansformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx, kwargs=(self.vars_mapping, self.layers_mapping))
                idx += 1
                self.layers += [last]
            else:
                raise NotImplementedError
    
    @torch.no_grad()
    def __call__(self, lower, upper, assignment=None):
        bounds = (lower, upper)
        for layer in self.layers:
            bounds = layer(bounds, assignment)
        self.bounds = bounds
        return self.bounds[0], self.bounds[1]

    def get_params(self):
        return self.layers[-1].params    

class AssignedDeepPolyInputTransformer(nn.Module):
    def __init__(self, last=None):
        super(AssignedDeepPolyInputTransformer, self).__init__()
        self.last = last

    def forward(self, bounds, assignment):
        self.bounds = torch.stack([bounds[0], bounds[1]], 0)
        return self.bounds
    
    def __str__(self):
        return 'Input'


class AssignedDeepPolyAffineTransformer(nn.Module):

    def __init__(self, weight, bias=None, last=None, back_sub_steps=0, idx=None):
        super(AssignedDeepPolyAffineTransformer, self).__init__()
        self.weight = weight
        self.bias = bias
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.idx = idx

    def forward(self, bounds, assignment):
        # print('\n\n--------start linear--------')
        # print('b[0]:', bounds[0].numpy().tolist())
        # print('b[1]:', bounds[1].numpy().tolist())
        # print('W+:', self.W_plus.numpy().tolist())
        # print('W-:', self.W_minus.numpy().tolist())
        # print()

        upper = torch.matmul(self.W_plus, bounds[1]) + torch.matmul(self.W_minus, bounds[0])
        lower = torch.matmul(self.W_plus, bounds[0]) + torch.matmul(self.W_minus, bounds[1])

        # print('upper:')
        # print('\tW+ @ b[1]:', torch.matmul(self.W_plus, bounds[1]).numpy().tolist())
        # print('\tW- @ b[0]:', torch.matmul(self.W_minus, bounds[0]).numpy().tolist())
        # print('\tupper    :', upper.numpy().tolist())

        # print('lower:')
        # print('\tW+ @ b[0]:', torch.matmul(self.W_plus, bounds[0]).numpy().tolist())
        # print('\tW- @ b[1]:', torch.matmul(self.W_minus, bounds[1]).numpy().tolist())
        # print('\tlower    :', lower.numpy().tolist())
        # print()


        # print('bias     :', self.bias.numpy().tolist())
        # print()


        self.bounds = torch.stack([lower, upper], 0) + self.bias.reshape(1, -1)
        # print('lower before:', self.bounds[0].numpy().tolist())
        # print('upper before:', self.bounds[1].numpy().tolist())
        # print()
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        print('--------start linear--------')
        print('lower:', self.bounds[0].numpy().tolist())
        print('upper:', self.bounds[1].numpy().tolist())
        print('--------end linear--------')
        print()
        return self.bounds
    
    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        # print('lower new_bounds:', new_bounds[0].numpy().tolist())
        # print('upper new_bounds:', new_bounds[1].numpy().tolist())
        # print()
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params
        
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        # print('back_sub linear')
        if params is None:
            Ml, Mu, bl, bu = self.weight, self.weight, self.bias, self.bias
        else:
            Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            # print('- beta:', self.last.beta.numpy().tolist())
            # print('- lambda:', self.last.lmbda.numpy().tolist())
            # print('- mu:', self.last.mu.numpy().tolist())
            # print('- Ml:', Ml.numpy().tolist())
            # print('- Mu:', Mu.numpy().tolist())
            # print('- bl:', bl.numpy().tolist())
            # print('- bu:', bu.numpy().tolist())
            # print()

            Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta
            blnew = bl + torch.matmul(torch.clamp(Ml, max=0), self.last.mu)
            bunew = bu + torch.matmul(torch.clamp(Mu, min=0), self.last.mu)

            # print('- Ml+:', torch.clamp(Ml, min=0).numpy().tolist())
            # print('- beta:', self.last.beta.numpy().tolist())
            # print('- Ml+ @ beta:', (torch.clamp(Ml, min=0) * self.last.beta.numpy()).tolist())
            # print()

            # print('- Ml-:', torch.clamp(Ml, max=0).numpy().tolist())
            # print('- lmbda:', self.last.lmbda.numpy().tolist())
            # print('- Ml- @ lmbda:', (torch.clamp(Ml, max=0) * self.last.lmbda.numpy()).tolist())
            # print()
            # print('- Mlnew:', Mlnew.numpy().tolist())
            # print()
            # print()

            # print('- Mu+:', torch.clamp(Mu, min=0).numpy().tolist())
            # print('- lmbda:', self.last.lmbda.numpy().tolist())
            # print('- Mu+ @ lmbda:', (torch.clamp(Mu, min=0) * self.last.lmbda.numpy()).tolist())
            # print()

            # print('- Mu-:', torch.clamp(Mu, max=0).numpy().tolist())
            # print('- beta:', self.last.beta.numpy().tolist())
            # print('- Mu- @ beta:', (torch.clamp(Mu, max=0) * self.last.beta.numpy()).tolist())
            # print()

            # print('- Munew:', Munew.numpy().tolist())
            # print()
            # print()
            # print()
            # print('- blnew:', blnew.numpy().tolist())
            # print()
            # print('- bunew:', bunew.numpy().tolist())
            # print()



            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            # print('- Ml+:', torch.clamp(Ml, min=0).numpy().tolist())
            # print('- b[0]:', self.last.bounds[0].numpy().tolist())
            # print()
            # print('- Ml-:', torch.clamp(Ml, max=0).numpy().tolist())
            # print('- b[1]:', self.last.bounds[1].numpy().tolist())
            # print()
            # print('- bl:', bl.numpy().tolist())
            # print()


            lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[0]) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[1]) + bl
            # print('- lower:', lower.numpy().tolist())
            # print()



            # print('- Mu+:', torch.clamp(Mu, min=0).numpy().tolist())
            # print('- b[1]:', self.last.bounds[1].numpy().tolist())
            # print()
            # print('- Mu-:', torch.clamp(Mu, max=0).numpy().tolist())
            # print('- b[0]:', self.last.bounds[0].numpy().tolist())
            # print()
            # print('- bu:', bu.numpy().tolist())
            # print()

            upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[1]) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[0]) + bu
            # print('- upper:', upper.numpy().tolist())
            # print()
            return torch.stack([lower, upper], 0), params

    def __str__(self):
        return f'Linear {self.idx}'



class AssignedDeepPolyReLUTansformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None, kwargs=None):
        super(AssignedDeepPolyReLUTansformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.idx = idx

        self.vars_mapping, self.layers_mapping = kwargs
    
    def forward(self, bounds, assignment):
        # print('\n\n--------start relu--------')
        # print('b[0]:', bounds[0].numpy().tolist())
        # print('b[1]:', bounds[1].numpy().tolist())
        ind2 = bounds[0]>=0 
        ind3 = (bounds[1]>=0) * (bounds[0]<=0) 
        # print('- ind3:', ind3.numpy().tolist())

        # ind4 = (bounds[1] > -bounds[0]) * ind3
        # ind4 = ind3
        # print('- ind4:', ind4.numpy().tolist())

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
        # print('lower before:', self.bounds[0].numpy().tolist())
        # print('upper before:', self.bounds[1].numpy().tolist())
        # print()

        if assignment is not None:
            la = np.array([assignment.get(i, None) for i in self.layers_mapping[self.idx]])

            # print('\t- assignment:', self.layers_mapping[self.idx])
            # print('\t- layer assignment:', la)
            # print('\t- layer assignment active:', la == True)
            # print('\t- layer assignment non-active:', la == False)
            # print('\t- bounds[0] :', bounds[0].numpy().tolist())
            # print('\t- bounds[0] w assignment (T):', bounds[0][active_ind].numpy().tolist())
            # print('\t- bounds[0] w assignment (F):', bounds[0][la==False].numpy().tolist())

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
        print('--------start relu--------')
        print('lower:', self.bounds[0].numpy().tolist())
        print('upper:', self.bounds[1].numpy().tolist())
        print('--------end relu--------')
        print()
        return self.bounds

    def __str__(self):
        return f'Relu {self.idx}'

    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps)
        new_bounds = new_bounds.reshape(self.bounds.shape)
        # print('lower new_bounds:', new_bounds[0].numpy().tolist())
        # print('upper new_bounds:', new_bounds[1].numpy().tolist())
        # print()
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        self.params = new_params

    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        # print('back_sub relu')
        if params is None:
            Ml, Mu, bl, bu = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu), self.mu
        else:
            Ml, Mu, bl, bu = params


        # print('- Ml:', Ml.numpy().tolist())
        # print('- Mu:', Mu.numpy().tolist())
        # print('- bl:', bl.numpy().tolist())
        # print('- bu:', bu.numpy().tolist())
        # print()

        # print('- self.last.weight:', self.last.weight.numpy().tolist())
        # print('- self.last.bias:', self.last.bias.numpy().tolist())
        # print()

        Mlnew = torch.matmul(Ml, self.last.weight)
        Munew = torch.matmul(Mu, self.last.weight) 
        blnew = bl + torch.matmul(Ml, self.last.bias)
        bunew = bu + torch.matmul(Mu, self.last.bias)

        # print('- Mlnew:', Mlnew.numpy().tolist())
        # print('- Munew:', Munew.numpy().tolist())
        # print('- blnew:', blnew.numpy().tolist())
        # print('- bunew:', bunew.numpy().tolist())
        # print()
        
        return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))


if __name__ == '__main__':


    torch.manual_seed(1)
    # random.seed(1)

    # net = FC(4, [3, 4, 20, 3]).eval()
    # lower = torch.Tensor([-0.4, -0.5, -0.4, 0.2])
    # upper = torch.Tensor([0.6, 0.7, 0.6, 0.4])

    # d = AssignedDeepPoly(net, back_sub_steps=10)
    # l, u = d(lower, upper)
    # print(l)
    # print(u)
    net = NetworkTorch('example/random.nnet')

    # net = CorinaNet().eval()
    lower = torch.Tensor([-1, -2])
    upper = torch.Tensor([1, 2])
    

    d = AssignedDeepPoly(net, back_sub_steps=10)

    # assignment = {v: random.choice([True, False, None]) for k, v in d.vars_mapping.items()}
    assignment = {1: False, 2: None}
    print(assignment)
    print()

    print('Without assignment')
    l, u = d(lower, upper, assignment=None)
    print(l)
    print(u)
    Ml, Mu, bl, bu = d.get_params()
    print('Ml', Ml.numpy().tolist())
    print('Mu', Mu.numpy().tolist())
    print('bl', bl.numpy().tolist())
    print('bu', bu.numpy().tolist())
    print()

    print('With assignment')
    l, u = d(lower, upper, assignment=assignment)
    print(l)
    print(u)
    print(d.get_params())

    Ml, Mu, bl, bu = d.get_params()
    print('Ml', Ml.numpy().tolist())
    print('Mu', Mu.numpy().tolist())
    print('bl', bl.numpy().tolist())
    print('bu', bu.numpy().tolist())

import torch.nn.functional as F
import torch.nn as nn
import torch

from typing import Tuple

class CorinaNet(nn.Module):

    def __init__(self):
        super().__init__()

        fc1 = nn.Linear(2, 2, bias=False)
        fc1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, 1]]))

        fc2 = nn.Linear(2, 2, bias=False)
        fc2.weight = torch.nn.Parameter(torch.Tensor([[0.5, -0.2], [-0.5, 0.1]]))

        fc3 = nn.Linear(2, 2, bias=False)
        fc3.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [-1, 1]]))

        self.layers = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)


    def forward(self, x):
        return self.layers(x)


class Normalization(nn.Module):

    def __init__(self):
        super(Normalization, self).__init__()

    def forward(self, x):
        return x


class FC(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        layers = [Normalization(), nn.Flatten()]
        prev_size = input_size
        for idx, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            if idx < len(hidden_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeepPoly():
    def __init__(self, net, eps, inputs, true_label, back_sub_steps=0, use_interpolation=False, only_first=True):
        self.net = net
        self.eps = eps
        self.only_first = only_first
        self.inputs = inputs
        self.true_label = true_label
        self.use_interpolation = use_interpolation
        self.back_sub_steps = back_sub_steps
        self.verifier = self._build_network_transformer()
    
    def _build_network_transformer(self):
        last = None
        layers=[DeepPolyInputTransformer(self.eps)]
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                last = DeepPolyAffineTransformer(layer._parameters['weight'].detach(),layer._parameters['bias'].detach(), last=last, back_sub_steps=self.back_sub_steps)
                layers += [last]
            elif isinstance(layer, torch.nn.Conv2d):
                last = DeepPolyConvTransformer(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer._parameters["weight"].detach(), layer._parameters["bias"].detach(), last, back_sub_steps=self.back_sub_steps)
                layers += [last]
            elif isinstance(layer, torch.nn.ReLU):
                last = DeepPolyReLUTansformer(last=last, back_sub_steps=self.back_sub_steps, use_interpolation= self.use_interpolation)
                layers += [last]
            elif isinstance(layer, torch.nn.Flatten):
                last = DeepPolyFlattenTransformer(last=last)
                layers += [last]
            elif isinstance(layer, Normalization):
                last = DeepPolyNormalizingTransformer()
                layers += [last]
            else:
                raise TypeError("Layer type unknown!")
        return nn.Sequential(*layers)
    
    def verify(self):
        return self.verifier(self.inputs)
    
class DeepPolyInputTransformer(nn.Module):
    def __init__(self, eps):
        super(DeepPolyInputTransformer, self).__init__()
        self.eps = eps
    
    def forward(self, input):
        self.bounds = input.repeat(2, 1, 1, 1)
        self.bounds += torch.FloatTensor([[[[-self.eps]]], [[[self.eps]]]])
        self.bounds = torch.clamp(self.bounds, 0., 1.)
        return self.bounds

class DeepPolyFlattenTransformer(nn.Module):
    def __init__(self, last=None):
        super(DeepPolyFlattenTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        print(torch.stack([bounds[0,:,:,:].flatten(), bounds[1,:,:,:].flatten()], 1))
        return torch.stack([bounds[0,:,:,:].flatten(), bounds[1,:,:,:].flatten()], 1)
    
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        bounds = self.last._back_sub(max_steps, params=params)
        bounds = torch.stack([bounds[:int(len(bounds)/2)], bounds[int(len(bounds)/2):]],1)
        return bounds
    
    @property
    def beta(self):
        return self.last.beta.flatten()

    @property
    def mu(self):
        return self.last.mu.flatten()

    @property
    def lmbda(self):
        return self.last.lmbda.flatten()
    
    @property
    def bounds(self):
        bounds = self.last.bounds
        return torch.stack([bounds[0,:,:,:].flatten(),bounds[1,:,:,:].flatten()], 1)

class DeepPolyNormalizingTransformer(nn.Module):
    def __init__(self, last=None):
        super(DeepPolyNormalizingTransformer, self).__init__()
        self.last = last
        self.mean = torch.FloatTensor([0.1307])
        self.sigma = torch.FloatTensor([0.3081])
    
    def forward(self, bounds):
        self.bounds = bounds
        return self.bounds

class DeepPolyAffineTransformer(nn.Module):
    def __init__(self, weights, bias=None, last=None, back_sub_steps=0):
        super(DeepPolyAffineTransformer, self).__init__()
        self.weights = weights
        self.bias = bias
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weights, min=0.)
        self.W_minus = torch.clamp(self.weights, max=0.)

    def forward(self, bounds):
        upper = torch.matmul(self.W_plus, bounds[:,1]) + torch.matmul(self.W_minus, bounds[:,0])
        lower = torch.matmul(self.W_plus, bounds[:,0]) + torch.matmul(self.W_minus, bounds[:,1])
        self.bounds = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds
    
    def back_sub(self, max_steps):
        new_bounds = self._back_sub(max_steps)
        indl = new_bounds[:,0] > self.bounds[:,0]
        indu = new_bounds[:,1] < self.bounds[:,1]
        self.bounds[indl, 0] = new_bounds[indl,0]
        self.bounds[indu, 1] = new_bounds[indu,1]
        
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if params is None:
            Ml, Mu, bl, bu = self.weights, self.weights, self.bias, self.bias
        else:
            Ml, Mu, bl, bu = params
        if max_steps > 0 and self.last.last.last is not None:
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0)* self.last.lmbda
            Munew = torch.clamp(Mu, min=0)* self.last.lmbda + torch.clamp(Mu, max=0)* self.last.beta
            blnew = bl + torch.matmul(torch.clamp(Ml, max=0), self.last.mu)
            bunew = bu + torch.matmul(torch.clamp(Mu, min=0), self.last.mu) 
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[:, 1]) + bl
            upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[:, 0]) + bu
            return torch.stack([lower, upper], 1)

class DeepPolyReLUTansformer(nn.Module):
    def __init__(self, last=None, back_sub_steps=0, use_interpolation=False):
        super(DeepPolyReLUTansformer, self).__init__()
        self.last = last
        self.last_conv_flag = isinstance(self.last, DeepPolyConvTransformer)
        self.back_sub_steps = back_sub_steps
        self.use_interpolation = use_interpolation
        if use_interpolation:
            if self.last_conv_flag:
                self.alpha = None
            else:
                alpha_init = torch.randn(last.weights.shape[0])
                self.alpha = nn.Parameter(alpha_init)
    
    def forward(self, bounds):
        if self.last_conv_flag:
            ind2 = bounds[0,:,:,:]>=0 
            ind3 = (bounds[1,:,:,:]>0) * (bounds[0,:,:,:]<0) 
            ind4 = (bounds[1,:,:,:] > -bounds[0,:,:,:]) * ind3 
            self.bounds = torch.zeros_like(bounds)
            self.bounds[1, ind3] = bounds[1, ind3]
            self.bounds[:, ind4] = bounds[:,ind4]
            self.lmbda = torch.zeros_like(bounds[1,:,:,:])
            self.beta = torch.zeros_like(bounds[1,:,:,:])
            self.mu = torch.zeros_like(bounds[1,:,:,:])
            self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2])
            ind5 = ind3+ind4 #
            diff = bounds[1,ind5] - bounds[0, ind5] 
            self.lmbda[ind5] = torch.div(bounds[1, ind5], diff)
            self.beta[ind4] = torch.ones_like(self.beta[ind4])
            if self.use_interpolation:
                self.beta[ind3] = torch.ones_like(self.beta[ind3]) 
                if isinstance(self.alpha, type(None)):
                    self.alpha = nn.Parameter(torch.randn_like(self.beta))
                self.beta *= torch.sigmoid(self.alpha)
                self.bounds[:, ind3] = bounds[:, ind3]
                self.bounds[0,:,:,:] *= torch.sigmoid(self.alpha)
            self.mu[ind5] = torch.div(-bounds[0, ind5]*bounds[1, ind5], diff)
            self.bounds[:, ind2] = bounds[:, ind2]
            self.beta[ind2] = torch.ones_like(self.beta[ind2])
        else:
            ind2 = bounds[:, 0]>=0 
            ind3 = (bounds[:, 1]>0) * (bounds[:, 0]<0) 
            ind4 = (bounds[:, 1] > -bounds[:, 0]) * ind3
            self.bounds = torch.zeros_like(bounds)
            self.bounds[ind3,1] = bounds[ind3,1]
            self.bounds[ind4,:] = bounds[ind4,:]
            self.lmbda = torch.zeros_like(bounds[:, 1])
            self.beta = torch.zeros_like(bounds[:, 1])
            self.mu = torch.zeros_like(bounds[:, 1])
            self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2])
            ind5 = ind3+ind4 
            diff = bounds[ind5, 1] - bounds[ind5, 0] 
            self.lmbda[ind5] = torch.div(bounds[ind5, 1], diff)
            self.beta[ind4] = torch.ones_like(self.beta[ind4])
            if self.use_interpolation:
                self.beta[ind3] = torch.ones_like(self.beta[ind3]) 
                self.beta *= torch.sigmoid(self.alpha)
                self.bounds[ind3,:] = bounds[ind3,:]
                self.bounds[:, 0] *= torch.sigmoid(self.alpha)
            self.mu[ind5] = torch.div(-bounds[ind5, 0]*bounds[ind5, 1], diff)
            self.bounds[ind2,:] = bounds[ind2,:]
            self.beta[ind2] = torch.ones_like(self.beta[ind2])
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds

    def back_sub(self, max_steps):
        new_bounds = self._back_sub(max_steps).reshape(self.bounds.shape)
        if self.last_conv_flag:
            indl = new_bounds[0,:,:,:] > self.bounds[0,:,:,:]
            indu = new_bounds[1,:,:,:] < self.bounds[1,:,:,:]
            self.bounds[0,indl] = new_bounds[0, indl]
            self.bounds[1,indu] = new_bounds[1, indu]
        else:
            indl = new_bounds[:,0] > self.bounds[:,0]
            indu = new_bounds[:,1] < self.bounds[:,1]
            self.bounds[indl, 0] = new_bounds[indl,0]
            self.bounds[indu, 1] = new_bounds[indu,1]
        
    def _back_sub(self, max_steps, params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if self.last_conv_flag:
            if params is None:
                Ml, Mu, bl, bu = torch.diag(self.beta.flatten()), torch.diag(self.lmbda.flatten()), torch.zeros_like(self.mu).flatten(), self.mu.flatten()
            else:
                Ml, Mu, bl, bu = params
            if max_steps > 0 and self.last.last is not None:
                Mlnew = torch.matmul(Ml, self.last.weights_backsub)
                Munew = torch.matmul(Mu, self.last.weights_backsub)
                blnew = bl + torch.matmul(Ml, self.last.bias_backsub).flatten()
                bunew = bu + torch.matmul(Mu, self.last.bias_backsub).flatten()
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[0,:,:,:].flatten()) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[1,:,:,:].flatten()) + bl
                upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[1,:,:,:].flatten()) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[0,:,:,:].flatten()) + bu
                return torch.cat([lower, upper], 0)
        else:
            if params is None:
                Ml, Mu, bl, bu = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu), self.mu
            else:
                Ml, Mu, bl, bu = params
            if max_steps > 0 and self.last.last is not None:
                Mlnew = torch.matmul(Ml, self.last.weights)
                Munew = torch.matmul(Mu, self.last.weights) 
                blnew = bl + torch.matmul(Ml, self.last.bias)
                bunew = bu + torch.matmul(Mu, self.last.bias)
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                lower = torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[:, 1]) + bl
                upper = torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[:, 0]) + bu
                return torch.stack([lower, upper], 1)

class DeepPolyPairwiseDifference(nn.Module):
    def __init__(self, true_label, last=None, back_sub_steps=0, only_first=True):
        super(DeepPolyPairwiseDifference, self).__init__()
        self.last = last
        self.true_label = true_label
        self.back_sub_steps = back_sub_steps
        self.only_first = only_first
        self.n_labels = self.last.weights.shape[0]
        self._set_weights()
        self.W1_plus = torch.clamp(self.weights1, min=0)
        self.W1_minus = torch.clamp(self.weights1, max=0)
        self.W2_plus = torch.clamp(self.weights1, min=0)
        self.W2_minus = torch.clamp(self.weights1, max=0)

    def forward(self, bounds):
        upper1 = torch.matmul(self.W1_plus, bounds[:,1]) + torch.matmul(self.W1_minus, bounds[:,0])
        lower1 = torch.matmul(self.W1_plus, bounds[:,0]) + torch.matmul(self.W1_minus, bounds[:,1])
        self.bounds1 = torch.stack([lower1, upper1], 1)
        if not self.only_first:
            upper2 = torch.matmul(self.W2_plus, bounds[:,1]) + torch.matmul(self.W2_minus, bounds[:,0])
            lower2 = torch.matmul(self.W2_plus, bounds[:,0]) + torch.matmul(self.W2_minus, bounds[:,1])
            self.bounds2 = torch.stack([lower2, upper2], 1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        if not self.only_first:
            return self.bounds1, self.bounds2
        else:
            return self.bounds1
        
    def back_sub(self, max_steps):
        if not self.only_first:
            new_bounds1, new_bounds2 = self._back_sub(max_steps)
            indl = new_bounds1[:,0] > self.bounds1[:,0]
            indu = new_bounds1[:,1] < self.bounds1[:,1]
            self.bounds1[indl, 0] = new_bounds1[indl,0]
            self.bounds1[indu, 1] = new_bounds1[indu,1]
            indl = new_bounds2[:,0] > self.bounds2[:,0]
            indu = new_bounds2[:,1] < self.bounds2[:,1]
            self.bounds2[indl, 0] = new_bounds2[indl,0]
            self.bounds2[indu, 1] = new_bounds2[indu,1]
        else:
            new_bounds = self._back_sub(max_steps)
            indl = new_bounds[:,0] > self.bounds1[:,0]
            indu = new_bounds[:,1] < self.bounds1[:,1]
            self.bounds1[indl, 0] = new_bounds[indl,0]
            self.bounds1[indu, 1] = new_bounds[indu,1]
        
    def _back_sub(self, max_steps):
        if not self.only_first:
            Ml1, Mu1, Ml2, Mu2  = self.weights1, self.weights1, self.weights2, self.weights2
        else:
            Ml1, Mu1  = self.weights1, self.weights1
            
        if max_steps > 0 and self.last.last is not None:
            Ml1new = torch.matmul(Ml1, self.last.weights) 
            Mu1new = torch.matmul(Mu1, self.last.weights) 
            bl1new = torch.matmul(Ml1, self.last.bias)
            bu1new = torch.matmul(Mu1, self.last.bias)
            if not self.only_first:
                Ml2new = torch.matmul(Ml2, self.last.weights) 
                Mu2new = torch.matmul(Mu2, self.last.weights) 
                bl2new = torch.matmul(Ml2, self.last.bias)
                bu2new = torch.matmul(Mu2, self.last.bias)
                return self.last._back_sub(max_steps-1, params=(Ml1new, Mu1new, bl1new, bu1new)), self.last._back_sub(max_steps-1, params=(Ml2new, Mu2new, bl2new, bu2new))
            else:
                return self.last._back_sub(max_steps-1, params=(Ml1new, Mu1new, bl1new, bu1new))
        else:
            lower1 = torch.matmul(torch.clamp(Ml1, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml1, max=0), self.last.bounds[:, 1]) 
            upper1 = torch.matmul(torch.clamp(Mu1, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu1, max=0), self.last.bounds[:, 0]) 
            if not self.only_first:
                lower2 = torch.matmul(torch.clamp(Ml2, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml2, max=0), self.last.bounds[:, 1]) 
                upper2 = torch.matmul(torch.clamp(Mu2, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu2, max=0), self.last.bounds[:, 0]) 
                return torch.stack([lower1, upper1], 1), torch.stack([lower2, upper2], 1)
            else:
                return torch.stack([lower1, upper1], 1)
    
    def _set_weights(self):
        self.weights1=torch.zeros((self.n_labels-1,self.n_labels))
        self.weights1[:,self.true_label]-=1
        self.weights1[:self.true_label, :self.true_label] += torch.eye(self.true_label)
        self.weights1[self.true_label:, self.true_label + 1:] += torch.eye(self.n_labels - self.true_label - 1)
        self.weights2=torch.zeros((self.n_labels-1,self.n_labels))
        self.weights2[:,self.true_label]+=1
        self.weights2[:self.true_label, :self.true_label] -= torch.eye(self.true_label)
        self.weights2[self.true_label:, self.true_label + 1:] -= torch.eye(self.n_labels - self.true_label - 1)
    
class DeepPolyConvTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weights, bias=None, last=None, back_sub_steps=0):
        super(DeepPolyConvTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights = weights
        self.last = last
        self.bias = bias
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weights, min=0)
        self.W_minus = torch.clamp(self.weights, max=0)
        self.conv_plus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_minus = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv_plus._parameters['weight'] = self.W_plus
        self.conv_minus._parameters['weight'] = self.W_minus
        self.conv_plus._parameters['bias'] = self.bias/2.
        self.conv_minus._parameters['bias'] = self.bias/2.
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride= self.stride, padding=self.padding)
        self.conv.requires_grad = False
        self.conv_plus.requires_grad = False
        self.conv_minus.requires_grad = False
        self.conv._parameters['weight'] = self.weights
        self.conv._parameters['bias'] = None
        self.weights_backsub = None
        self.bias_backsub = None
    
    def toeplitz_convmatrix2d(self):
        inputs = torch.ones_like(self.last.bounds[1,:,:,:].flatten())
        reshape_conv = ReshapeConv(self.last.bounds[1,:,:,:].shape[2], self.last.bounds[1,:,:,:].shape[2] // self.stride[0], self.in_channels, self.out_channels, self.conv)
        ## hacky but works: find toeplitz by jacobian
        j = jacobian(reshape_conv, inputs)
        j.requires_grad = False
        return j

    def forward(self, bounds):
        if isinstance(self.weights_backsub, type(None)):
            self.weights_backsub = self.toeplitz_convmatrix2d()
        bounds = bounds.unsqueeze(0)
        upper = self.conv_plus(bounds[:,1,:,:,:]) + self.conv_minus(bounds[:,0,:,:,:])
        lower = self.conv_plus(bounds[:,0,:,:,:]) + self.conv_minus(bounds[:,1,:,:,:])
        self.bounds = torch.stack([lower, upper],0).squeeze(1)
        if isinstance(self.bias_backsub, type(None)):
            self.bias_backsub = self.bias.repeat_interleave(self.bounds[1,:,:,:].shape[1]*self.bounds[1,:,:,:].shape[2])
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds
    
    def back_sub(self, max_steps):
        new_bounds = self._back_sub(max_steps).reshape(self.bounds.shape)
        indl = new_bounds[0,:,:,:] > self.bounds[0,:,:,:]
        indu = new_bounds[1,:,:,:] < self.bounds[1,:,:,:]
        self.bounds[0,indl] = new_bounds[0, indl]
        self.bounds[1,indu] = new_bounds[1, indu]
    
    def _back_sub(self, max_steps,  params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None):
        if params is None:
            Ml, Mu, bl, bu = self.weights_backsub, self.weights_backsub, self.bias_backsub, self.bias_backsub
        else:
            Ml, Mu, bl, bu = params
        if max_steps > 0 and self.last.last is not None:
            Mlnew = torch.clamp(Ml, min=0) * self.last.beta.flatten() + torch.clamp(Ml, max=0)* self.last.lmbda.flatten()
            Munew = torch.clamp(Mu, min=0) * self.last.lmbda.flatten() + torch.clamp(Mu, max=0)* self.last.beta.flatten()
            blnew = bl + torch.matmul(torch.clamp(Ml, max=0), self.last.mu.flatten())
            bunew = bu + torch.matmul(torch.clamp(Mu, min=0), self.last.mu.flatten()) 
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = (torch.matmul(torch.clamp(Ml, min=0), self.last.bounds[0,:,:,:].flatten().reshape(-1,1)) + torch.matmul(torch.clamp(Ml, max=0), self.last.bounds[1,:,:,:].flatten().reshape(-1,1))).flatten() + bl
            upper = (torch.matmul(torch.clamp(Mu, min=0), self.last.bounds[1,:,:,:].flatten().reshape(-1,1)) + torch.matmul(torch.clamp(Mu, max=0), self.last.bounds[0,:,:,:].flatten().reshape(-1,1))).flatten() + bu
            return torch.cat([lower, upper], 0)


if __name__ == "__main__":
    torch.manual_seed(1)
    net = FC(4, [3, 4, 2]).eval()



    inputs = torch.Tensor([0.5, 0.6, 0.5, 0.3]).view(1, 1, 2, 2)
    eps = 0.1

    d = DeepPoly(net, eps, inputs, None, back_sub_steps=10, only_first=False)
    l, u = d.verify()
    print(l)
    print(u)
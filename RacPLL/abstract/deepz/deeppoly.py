import torch.nn.functional as F
import torch.nn as nn
import torch




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



class LinearTransformer(nn.Module):
    def __init__(self, layer):
        super(LinearTransformer, self).__init__()

        self.weights = layer.weight
        self.bias = layer.bias
        self.W_plus = torch.clamp(self.weights, min=0.)
        self.W_minus = torch.clamp(self.weights, max=0.)

    def forward(self, bounds):
        upper = torch.matmul(self.W_plus, bounds[:,1]) + torch.matmul(self.W_minus, bounds[:,0])
        lower = torch.matmul(self.W_plus, bounds[:,0]) + torch.matmul(self.W_minus, bounds[:,1])
        self.bounds = torch.stack([lower, upper], 1)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1)
        return self.bounds
    
class ReLUTransformer(nn.Module):
    def __init__(self):
        super(ReLUTransformer, self).__init__()
    
    def forward(self, bounds):
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
        self.mu[ind5] = torch.div(-bounds[ind5, 0]*bounds[ind5, 1], diff)
        self.bounds[ind2,:] = bounds[ind2,:]
        self.beta[ind2] = torch.ones_like(self.beta[ind2])
        return self.bounds
    

class DeepPolyFake:

    def __init__(self, net):
        layers = []
        for layer in net.layers:
            if isinstance(layer, torch.nn.Linear):
                layers.append(LinearTransformer(layer))
            elif isinstance(layer, torch.nn.ReLU):
                layers.append(ReLUTransformer())
            else:
                raise TypeError("Layer type unknown!")
        self.layers = nn.Sequential(*layers)
    
    def __call__(self, lower, upper):
        x = torch.stack([lower, upper]).transpose(0, 1)
        x = self.layers(x)
        return x[:, 0], x[:, 1]


class DeepPoly:
    def __init__(self, size, lb, ub):
        iden = torch.diag(torch.ones(size))
        self.slb = torch.cat([iden, torch.zeros(size).unsqueeze(1)], dim=1)
        self.sub = self.slb
        self.lb = lb
        self.ub = ub
        self.hist = []
        self.layers = 0
        self.is_relu = False

    def save(self):
        lb = torch.cat([self.lb, torch.ones(1)])
        ub = torch.cat([self.ub, torch.ones(1)])
        if not self.is_relu:
            keep_bias = torch.zeros(1, self.slb.shape[1])
            keep_bias[0, self.slb.shape[1] - 1] = 1
            slb = torch.cat([self.slb, keep_bias], dim=0)
            sub = torch.cat([self.sub, keep_bias], dim=0)
        else:
            slb = self.slb
            sub = self.sub
        self.layers += 1
        self.hist.append((slb, sub, lb, ub, self.is_relu))
        return self

    def resolve(self, cstr, layer, lower=True):
        cstr_relu_pos = F.relu(cstr)
        cstr_relu_neg = F.relu(-cstr)
        dp = self.hist[layer]
        is_relu = dp[-1]
        if layer == 0:
            lb, ub = dp[2], dp[3]
        else:
            lb, ub = dp[0], dp[1]
        if not lower:  # switch lb and ub
            lb, ub = ub, lb
        if is_relu:
            lb_diag, lb_bias = lb[0], lb[1]
            ub_diag, ub_bias = ub[0], ub[1]
            lb_bias = torch.cat([lb_bias, torch.ones(1)])
            ub_bias = torch.cat([ub_bias, torch.ones(1)])
            m1 = torch.cat([cstr_relu_pos[:, :-1] * lb_diag, torch.matmul(cstr_relu_pos, lb_bias).unsqueeze(1)], dim=1)
            m2 = torch.cat([cstr_relu_neg[:, :-1] * ub_diag, torch.matmul(cstr_relu_neg, ub_bias).unsqueeze(1)], dim=1)
            return m1 - m2
        else:
            return torch.matmul(cstr_relu_pos, lb) - torch.matmul(cstr_relu_neg, ub)

class DPLinear(nn.Module):
    def __init__(self, nested: nn.Linear):
        super().__init__()
        self.weight = nested.weight.detach()
        if nested.bias is not None:
            self.bias = nested.bias.detach()
        else:
            self.bias = torch.zeros(self.weight.shape[0])
        self.in_features = nested.in_features
        self.out_features = nested.out_features

    def forward(self, x):
        x.save()

        # append bias as last column
        init_slb = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)

        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        # loop layer backwards
        for i in range(x.layers - 1, -1, -1):
            x.lb = x.resolve(x.lb, i, lower=True)
            x.ub = x.resolve(x.ub, i, lower=False)
        x.is_relu = False
        return x


class DPReLU(nn.Module):
    def __init__(self, in_features):
        super(DPReLU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        # have lambdas as trainable parameter
        self.lam = torch.nn.Parameter(torch.zeros(in_features))  # TODO: pick different lambdas here? train them?

    def forward(self, x):
        x.save()

        lb, ub = x.lb, x.ub

        # cases 1-3.2
        mask_1, mask_2 = lb.ge(0), ub.le(0)
        mask_3 = ~(mask_1 | mask_2)
        # this should be the right area minimizing heuristic (starting from `ones` lambdas)
        # self.lam.data = self.lam.where(ub.gt(-lb), torch.zeros_like(self.lam))
        a = torch.where((ub - lb) == 0, torch.ones_like(ub), ub / (ub - lb + 1e-6))
        
        x.lb = lb * mask_1 + 1/(1+torch.exp(self.lam)) * lb * mask_3
        x.ub = ub * mask_1 + ub * mask_3
        curr_slb = 1 * mask_1 + 1/(1+torch.exp(self.lam)) * mask_3
        curr_sub = 1 * mask_1 + a * mask_3
        bias = - lb * a * mask_3

        # only save slb and sub as vectors, which we know encodes a diag matrix
        x.slb = torch.cat([curr_slb.unsqueeze(0), torch.zeros(len(lb)).unsqueeze(0)], dim=0)
        x.sub = torch.cat([curr_sub.unsqueeze(0), bias.unsqueeze(0)], dim=0)
        x.is_relu = True
        return x


def build_verifier_network(net, init_features):
    # ordered_layers = [module for module in net.modules()
                      # if type(module) not in [networks.FullyConnected, networks.Conv, nn.Sequential]]
    verif_layers = []
    for layer in net.layers:
        in_features = verif_layers[-1].out_features if len(verif_layers) > 0 else init_features
        if type(layer) == nn.Linear:
            verif_layers.append(DPLinear(layer))
        elif type(layer) == nn.ReLU:
            verif_layers.append(DPReLU(in_features))
        else:
            print("Layer type not implemented: ", str(type(layer)))
            exit(-1)
    return nn.Sequential(*verif_layers)



def test():
    torch.manual_seed(0)
    net = CorinaNet().eval()
    net = FC(5, [2, 2, 5]).eval()

    lower = torch.Tensor([-5, -4, 1, 2, 3])
    upper = torch.Tensor([-1, -2, 4, 5, 6])

    deeppoly = DeepPolyFake(net)
    y = deeppoly(lower, upper)
    print(y)


if __name__ == '__main__':
    net = CorinaNet().eval()
    # net = FC(2, [3, 4, 5]).eval()
    verifier = build_verifier_network(net, 2)

    lower = torch.Tensor([-5, -4])
    upper = torch.Tensor([-1, -2])

    x = DeepPoly(lower.shape[0], lower, upper)
    y = verifier(x)
    print(y.lb)
    print(y.ub)
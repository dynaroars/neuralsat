from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from interval import SymbolicInterval


class IntervalNetwork(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.net = []

        for layer in model:
            if (isinstance(layer, nn.Linear)):
                self.net.append(IntervalDense(layer))
            if (isinstance(layer, nn.ReLU)):
                self.net.append(IntervalReLU(layer))
            if 'Flatten' in (str(layer.__class__.__name__)):
                self.net.append(IntervalFlatten())
        self.net = nn.Sequential(*self.net)


    def forward(self, ix):
        return self.net(ix)



class IntervalDense(nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, ix):
        ix.c = F.linear(ix.c, self.layer.weight, bias=self.layer.bias)
        ix.idep = F.linear(ix.idep, self.layer.weight)
        for i in range(len(ix.edep)):
            ix.edep[i] = F.linear(ix.edep[i], self.layer.weight)
        ix.shape = list(ix.c.shape[1:])
        ix.n = list(ix.c[0].view(-1).size())[0]
        ix.concretize()
        return ix

        

class IntervalReLU(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, ix):
        lower = ix.l.clamp(max=0)
        upper = ix.u.clamp(min=0)
        upper = torch.max(upper, lower + 1e-8)
        mask = upper / (upper - lower)
        # vector that describes if both the lb < 0 and ub > 0
        appr_condition = ((ix.l < 0) * (ix.u > 0))  
        appr_ind = appr_condition.view(-1, ix.n).nonzero()
        appr_err = mask * (-lower) / 2.0

        m = int(appr_condition.sum().item())

        if (m != 0):

            if (ix.use_cuda):
                error_row = torch.zeros((m, ix.n), dtype=ix.c.dtype, device=lower.get_device())
            else:
                error_row = torch.zeros((m, ix.n), dtype=ix.c.dtype)

            error_row = error_row.scatter_(1, appr_ind[:, 1, None], appr_err[appr_condition][:, None])

            edep_ind = lower.new(appr_ind.size(0), lower.size(0)).zero_()
            edep_ind = edep_ind.scatter_(1, appr_ind[:, 0][:, None], 1)

        ix.c = ix.c * mask + appr_err * appr_condition.type_as(lower)

        for i in range(len(ix.edep)):
            ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

        ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)

        if (m != 0):
            ix.edep = ix.edep + [error_row]
            ix.edep_ind = ix.edep_ind + [edep_ind]

        return ix

    

class IntervalFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ix):
        ix.extend()
        return ix


class IntervalBound(nn.Module):
    def __init__(self, net, epsilon, method="sym", use_cuda=True, norm="linf", worst_case=True):
        super().__init__()
        self.net = net
        self.epsilon = epsilon
        self.use_cuda = use_cuda
        assert method in ["sym", "naive", "inverse", "center_sym", "new", "mix"], "No such interval methods!"
        self.method = method
        self.norm = norm
        # assert self.norm in ["linf", "l2", "l1"], "norm" + norm + "not supported"

        self.worst_case = worst_case

    def forward(self, X, y):

        out_features = self.net[-1].out_features

        if self.worst_case:
            c = torch.eye(out_features).type_as(X)[y].unsqueeze(1) - torch.eye(out_features).type_as(X).unsqueeze(0)
        else:
            c = None

        # Transfer original model to interval models
        inet = IntervalNetwork(self.net, c)

        minimum = (X - self.epsilon).min().item()
        maximum = (X + self.epsilon).max().item()

        # Create symbolic inteval classes from X
        if (self.method == "naive"):
            ix = Interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)
        if (self.method == "inverse"):
            ix = Inverse_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)
        if (self.method == "center_sym"):
            ix = Center_symbolic_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)

        if self.method == "mix":
            assert self.norm == "linf", "only support linf for now"
            ix = mix_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), use_cuda=self.use_cuda)

        if (self.method == "sym"):
            if self.norm == "linf":
                ix = Symbolic_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), use_cuda=self.use_cuda)
            elif self.norm == "l2":
                ix = Symbolic_interval(X, X, self.epsilon, norm="l2", use_cuda=self.use_cuda)
            elif self.norm == "l1":
                ix = Symbolic_interval(X, X, self.epsilon, norm="l1", use_cuda=self.use_cuda)

        # Propagate symbolic interval through interval networks
        ix = inet(ix)
        # print(ix.u)
        # print(ix.l)

        # Calculate the worst case outputs
        if self.method != "naive":
            wc = ix.worst_case(y, out_features)
            return wc

        return -ix.l



def naive_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, norm="linf"):
    # Transfer original model to interval models

    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="naive", use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="naive", use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y).type(torch.Tensor)
    ierr = ierr.sum().item() / X.shape[0]

    return iloss, ierr


def sym_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, proj=None, norm="linf"):
    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="sym", proj=proj, use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="sym", proj=proj, use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y)
    ierr = ierr.sum().item() / X.size(0)

    return iloss, ierr


def gen_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, proj=None, norm=["linf", "l2", "l1"]):
    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="gen", proj=proj, use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="gen", proj=proj, use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y)
    ierr = ierr.sum().item() / X.size(0)

    return iloss, ierr

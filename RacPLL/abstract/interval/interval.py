from __future__ import print_function

import warnings
import numpy as np
import torch


class Interval:

    def __init__(self, lower, upper, use_cuda=False):
        assert not ((upper - lower) < 0).any(), "upper less than lower"
        self.l = lower
        self.u = upper
        self.c = (lower + upper) / 2
        self.e = (upper - lower) / 2
        self.mask = []
        self.use_cuda = use_cuda

    def update_lu(self, lower, upper):
        assert not ((upper - lower) < 0).any(), "upper less than lower"
        self.l = lower
        self.u = upper
        self.c = (lower + upper) / 2
        self.e = (upper - lower) / 2

    def update_ce(self, center, error):
        assert not (error < 0).any(), "upper less than lower"
        self.c = center
        self.e = error
        self.u = self.c + self.e
        self.l = self.c - self.e

    def __str__(self):
        string = "Interval shape:" + str(self.c.shape)
        string += "\n    - lower:" + str(self.l.item())
        string += "\n    - upper:" + str(self.u.item())
        string += "\n    - center:" + str(self.c.item())
        return string

    def worst_case(self, y, output_size):
        '''Calculate the wrost case of the analyzed output ranges.
        In details, it returns the upper bound of other label minus
        the lower bound of the target label. If the returned value is
        less than 0, it means the worst case provided by interval
        analysis will never be larger than the target label y's.
        '''
        assert y.shape[0] == self.l.shape[0] == self.u.shape[0], "wrong input shape"

        for i in range(y.shape[0]):
            t = self.l[i, y[i]]
            self.u[i] = self.u[i] - t
            self.u[i, y[i]] = 0.0
        return self.u



class SymbolicInterval(Interval):
    
    def __init__(self, lower, upper, epsilon=0, norm="linf", use_cuda=False):
        assert lower.shape[0] == upper.shape[0], "each symbolic" + "should have the same shape"

        Interval.__init__(self, lower, upper)
        self.use_cuda = use_cuda
        self.shape = list(self.c.shape[1:])
        self.n = list(self.c[0].reshape(-1).size())[0]
        self.input_size = self.n
        self.batch_size = self.c.shape[0]
        self.epsilon = epsilon
        self.norm = norm
        if (self.use_cuda):
            self.idep = torch.eye(self.n, device=self.c.get_device(), dtype=self.c.dtype).unsqueeze(0)
        else:
            self.idep = torch.eye(self.n, dtype=self.c.dtype).unsqueeze(0)
        self.edep = []
        self.edep_ind = []


    def concretize(self):
        self.extend()
        if self.norm == "linf":

            e = (self.idep * self.e.view(self.batch_size, self.input_size, 1)).abs().sum(dim=1)

        elif self.norm == "l2":
            idep = torch.norm(self.idep, dim=1, keepdim=False)

            e = idep * self.epsilon

        elif self.norm == "l1":
            idep = self.idep.abs().max(dim=1, keepdim=False)[0]

            e = idep * self.epsilon

        if self.edep:
            for i in range(len(self.edep)):
                e = e + self.edep_ind[i].t().mm(self.edep[i].abs())

        self.l = self.c - e
        self.u = self.c + e

        return self


    def extend(self):
        self.c = self.c.reshape(self.batch_size, self.n)
        self.idep = self.idep.reshape(-1, self.input_size, self.n)

        for i in range(len(self.edep)):
            self.edep[i] = self.edep[i].reshape(-1, self.n)



    def shrink(self):
        self.c = self.c.reshape(tuple([-1] + self.shape))
        self.idep = self.idep.reshape(tuple([-1] + self.shape))

        for i in range(len(self.edep)):
            self.edep[i] = self.edep[i].reshape(tuple([-1] + self.shape))

    '''Calculate the wrost case of the analyzed output ranges.
    Return the upper bound of other output dependency minus target's
    output dependency. If the returned value is less than 0, it means
    the worst case provided by interval analysis will never be larger
    than the target label y's. 
    '''

    def worst_case(self, y, output_size):

        assert y.shape[0] == self.l.shape[0] == self.batch_size, "wrong label shape"
        if (self.use_cuda):
            kk = torch.eye(output_size, dtype=torch.uint8, requires_grad=False, device=y.get_device())[y]
        else:
            kk = torch.eye(output_size, dtype=torch.uint8, requires_grad=False)[y]

        c_t = self.c.masked_select(kk).unsqueeze(1)
        self.c = self.c - c_t

        idep_t = self.idep.masked_select(kk.view(self.batch_size, 1, output_size)).view(self.batch_size, self.input_size, 1)
        self.idep = self.idep - idep_t

        for i in range(len(self.edep)):
            edep_t = self.edep[i].masked_select((self.edep_ind[i].mm(kk.type_as(self.edep_ind[i]))).type_as(kk)).view(-1, 1)
            self.edep[i] = self.edep[i] - edep_t

        self.concretize()

        return self.u


class GenSym(SymbolicInterval):

    def __init__(self, lower, upper, epsilon=[0, 0, 0], norm=["linf", "l2", "l1"], use_cuda=False):

        Symbolic_interval.__init__(self, lower, upper, epsilon, norm, use_cuda)
        self.use_cuda = use_cuda
        self.shape = list(self.c.shape[1:])
        self.n = list(self.c[0].reshape(-1).size())[0]
        self.input_size = self.n
        self.batch_size = self.c.shape[0]
        self.epsilon = epsilon
        self.norm = norm
        if (self.use_cuda):
            self.idep = torch.eye(self.n, device=self.c.get_device()).unsqueeze(0)
        else:
            self.idep = torch.eye(self.n).unsqueeze(0)
        self.edep = []
        self.edep_ind = []

    def concretize(self):
        self.extend()
        e = None
        for i in range(len(self.norm)):

            if self.norm[i] == "linf":
                e0 = (self.idep * self.e.view(self.batch_size, self.input_size, 1)).abs().sum(dim=1)

            elif self.norm[i] == "l2":
                idep = torch.norm(self.idep, dim=1, keepdim=False)

                e0 = idep * self.epsilon[i]

            elif self.norm[i] == "l1":
                idep = self.idep.abs().max(dim=1, keepdim=False)[0]

                e0 = idep * self.epsilon[i]

            if e is None:
                e = e0
            else:
                e = torch.where(e > e0, e, e0)

        if self.edep:
            for i in range(len(self.edep)):
                e = e + self.edep_ind[i].t().mm(self.edep[i].abs())

        self.l = self.c - e
        self.u = self.c + e

        return self

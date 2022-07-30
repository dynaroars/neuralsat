from deeppoly import DeepPolyReLUTansformer
import torch.optim as optim
import torch
import numpy as np
from time import time
import torch.nn.functional as F

class OptimizerDeepPoly():
    def __init__(self, deeppoly, lr = 2e-2, patience = 50):
        self.deeppoly = deeppoly
        self.lr = lr
        self.patience = patience
        self.get_number_of_relu()

    def optimize(self):
        MAX_TIME = 80 / self.n_relu
        fwd_start=time()
        b = self.deeppoly.verify()
        fwd_time=time()-fwd_start
        for layer in self.deeppoly.verifier:
            if isinstance(layer, DeepPolyReLUTansformer):
                layer.alpha.requires_grad = False
        n_steps=max(int(np.ceil(100/fwd_time)),11) 
        for layer in self.deeppoly.verifier:
            if isinstance(layer, DeepPolyReLUTansformer):        
                fwd_start=time()
                layer.alpha.requires_grad = True
                optimizer = optim.Adam(layer.parameters(), lr=self.lr, betas=[0.90, 0.999])
                MULTIPLIER = 100
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr * MULTIPLIER, total_steps=n_steps, pct_start=0.1,base_momentum=0.85, max_momentum=0.90, div_factor=MULTIPLIER, final_div_factor=MULTIPLIER+12, anneal_strategy='linear')
                min_loss = np.inf
                no_improvement = 0
                while time()-fwd_start < MAX_TIME and no_improvement < self.patience: 
                    self.deeppoly.verifier.zero_grad()
                    b = self.deeppoly.verify()
                    l = self.loss(b)
                    l.backward()
                    optimizer.step()
                    if l+1e-3 < min_loss:
                        no_improvement = 0
                        min_loss = l
                    else:
                        no_improvement += 1
                    try:
                        scheduler.step()
                    except:
                        pass
                layer.alpha.requires_grad = False
            if sum(b[:,1]>0)==0:
                return True
        for layer in self.deeppoly.verifier:
            if isinstance(layer, DeepPolyReLUTansformer):
                layer.alpha.requires_grad = True
        n_steps=max(int(np.ceil(100/fwd_time)),6)
        optimizer = optim.Adam(self.deeppoly.verifier.parameters(), lr=self.lr, betas=[0.90, 0.999])
        MULTIPLIER = 80
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr * MULTIPLIER, total_steps=n_steps, pct_start=0.3, base_momentum=0.85, max_momentum=0.90, div_factor=MULTIPLIER, final_div_factor=MULTIPLIER/2+12, anneal_strategy='linear')
        min_loss = np.inf
        no_improvement = 0
        while True:
            self.deeppoly.verifier.zero_grad()
            b = self.deeppoly.verify()
            l = self.loss(b)
            l.backward()
            optimizer.step()
            try:
                scheduler.step()
            except:
                pass
            if sum(b[:,1]>0)==0:
                return True
            if min_loss <= l:
                no_improvement += 1
            else:
                no_improvement = 0
                min_loss = l
            if no_improvement > self.patience:
                break
        return False
    
    def get_number_of_relu(self):
        self.n_relu = 0
        for layer in self.deeppoly.verifier:
            if isinstance(layer, DeepPolyReLUTansformer):
                self.n_relu += 1
    
    @staticmethod
    def loss(b):
        return torch.sum(F.relu(b))
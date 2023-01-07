from sortedcontainers import SortedList
from torch import nn
import numpy as np
import torch
import math
import time


class InputDomainList:

    def __init__(self):
        self.domains = SortedList()

    def __len__(self):
        return len(self.domains)


    def add_batch(self, input_lower, input_upper, output_lower, output_upper, c, rhs, split_idx=None, remaining_index=None):
        device = output_lower.device

        rhs = rhs.to(device=device)
        input_lower = input_lower.to(device=device)
        input_upper = input_upper.to(device=device)

        if remaining_index is None:
            # remaining_index = range(len(input_lower))
            remaining_index = torch.where((output_lower <= rhs).all(1))[0]

        for i in remaining_index:
            domain = InputDomain(
                input_lower=input_lower[i : i + 1],
                input_upper=input_upper[i : i + 1],
                output_lower=output_lower[i],
                output_upper=output_upper[i] if output_upper is not None else None,
                c=c,
                rhs=rhs,
                split_idx=(split_idx[i : i + 1] if split_idx is not None else None),
            )
            self.domains.add(domain)
            # if not domain.verify_criterion():
                # self.domains.add(domain)



    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available() and device=='cuda':
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self.domains), batch)
        dm_l_all, dm_u_all = [], []
        split_idx = []
        for i in range(batch):
            # Pop out domains from the list one by one (SLOW).
            selected_candidate_domain = self.domains.pop(0)
            selected_candidate_domain.to_device(device, partial=True)
            dm_l_all.append(selected_candidate_domain.input_lower)
            dm_u_all.append(selected_candidate_domain.input_upper)
            split_idx.append(selected_candidate_domain.split_idx)

        # Input split domains.
        return (
            torch.cat(dm_l_all).to(device=device, non_blocking=True),
            torch.cat(dm_u_all).to(device=device, non_blocking=True),
            torch.cat(split_idx),
        )


    def get_topk_indices(self, k=1, largest=False):
        k = min(k, self.__len__())
        return -torch.arange(k) - 1 if largest else torch.arange(k)


    def __getitem__(self, idx):
        domain = self.domains[idx]
        return domain.input_lower, domain.input_upper




class InputDomain:

    def __init__(self, input_lower, input_upper, output_lower, output_upper, split_idx=None, c=None, rhs=None, device='cpu'):
        self.input_lower = input_lower
        self.input_upper = input_upper

        self.output_lower = output_lower
        self.output_upper = output_upper

        self.split_idx = split_idx
        self.rhs = rhs
        self.c = c

        self.device = device


    def __lt__(self, other):
        return (self.output_lower - self.rhs).max() < (other.output_lower - other.rhs).max()

    def __le__(self, other):
        return (self.output_lower - self.rhs).max() <= (other.output_lower - other.rhs).max()

    def __eq__(self, other):
        return (self.output_lower - self.rhs).max() == (other.output_lower - other.rhs).max()

    # @torch.no_grad()
    # def verify_criterion(self):
    #     ret = ((self.c > 0) * self.output_lower - (self.c < 0) * self.output_upper).sum(-1)
    #     return (ret > self.rhs).any()
 

    def to_device(self, device, partial=False):
        self.input_lower = self.input_lower.to(device, non_blocking=True)
        self.input_upper = self.input_upper.to(device, non_blocking=True)
        self.output_lower = self.output_lower.to(device, non_blocking=True)
        self.output_upper = self.output_upper.to(device, non_blocking=True) if self.output_upper is not None else None
        self.rhs = self.rhs.to(device, non_blocking=True)
        self.c = self.c.to(device, non_blocking=True)
        return self



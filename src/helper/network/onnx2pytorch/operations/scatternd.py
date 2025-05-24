from torch import nn
import torch


class ScatterND(nn.Module):
    
    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        output = data.clone()
        k = indices.shape[-1]
        indices_list = []
        for i in range(k):
            indices_list.append(indices[:, i])
        output[indices_list] = updates
        return output

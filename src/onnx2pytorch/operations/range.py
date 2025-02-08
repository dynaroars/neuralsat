import torch
from torch import nn

from onnx2pytorch.utils import PRINT_DEBUG

class Range(nn.Module):
    def forward(self, start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor):
        
        if PRINT_DEBUG:
            print('RANGE:', start, limit, delta)

        if start.numel() == 1:
            start = start.item()
            
        if limit.numel() == 1:
            limit = limit.item()
            
        if delta.numel() == 1:
            delta = delta.item()
        
        if PRINT_DEBUG:
            print('\t- start:', start)
            print('\t- limit:', limit)
            print('\t- delta:', delta)
            
        final = torch.arange(start=start, end=limit, step=delta)
        
        if PRINT_DEBUG:
            print('\t- final:', final)
            print('\t- final:', final.shape)

        return final

import torch
from torch import nn

from onnx2pytorch.utils import PRINT_DEBUG

class Gather(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        # selection = self.selection + [indices.to(torch.int64)]
        # return data.__getitem__(selection)
        if PRINT_DEBUG:
            print('GATHER:', data.shape, data.is_floating_point())
            # if data.ndim == 1:
            #     data = data[None]
                
            print('\t- self.dim:', self.dim)
            print('\t- indices:', indices)
            
        if data.is_floating_point():
            dim = max(self.dim, 1)
        else:
            dim = self.dim
            
        
        if indices.numel() == 1 and indices == -1:
            indices = torch.tensor(data.shape[dim] - 1, device=data.device)

        # if indices.ndim == 0:
        #     final = torch.index_select(data, dim=dim, index=indices)
        #     # indices = indices.view(-1)#.repeat(data.shape[0])
        #     print('\t- final 1:', final.shape)
        #     # final = final.squeeze(dim)
        #     # self.dim += 1
        # else:
        #     final = torch.index_select(data, dim=dim, index=indices) 
        #     raise
        
        final = torch.index_select(data, dim=dim, index=indices).squeeze(dim)
            
        if PRINT_DEBUG:
            print('\t- dim:', dim)
            print('\t- final:', final)
            print('\t- final:', final.shape)
            assert final.numel() > 0
        # exit()
        return final


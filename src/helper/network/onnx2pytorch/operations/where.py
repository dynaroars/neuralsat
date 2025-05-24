from torch import nn
import torch


class Where(nn.Module):
    
    def forward(self, condition: torch.Tensor, X: torch.Tensor, Y=torch.Tensor):
        res_type = torch.result_type(X, Y)
        output = torch.where(condition, X.to(res_type), Y.to(res_type))
        return output

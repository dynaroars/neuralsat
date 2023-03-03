from torch.nn import functional as F
import torch.nn as nn
import torch


class Example(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        linear1 = nn.Linear(2, 2)
        # linear1.weight.data = torch.Tensor([[-1, 1], [2, 2]])
        # linear1.bias.data = torch.Tensor([-1, 1])
        
        linear2 = nn.Linear(2, 2)
        # linear2.weight.data = torch.Tensor([[1, 1], [2, 2]])
        # linear2.bias.data = torch.Tensor([-1, 1])
        
        self.layers = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2
        )
        
        self.input_shape = (1, 2)
        self.output_shape = (1, 2)
        
    def forward(self, x):
        return self.layers(x)
    
    
if __name__ == '__main__':

    model = Example()
    # print(model)
    model.eval()
    
    # for name, layer in model.layers.named_modules():
    #     if name:
    #         if hasattr(layer, 'weight'):
    #             print(name, layer)
    #             print('\t-', layer.weight.data.shape)
    #             print('\t-', layer.bias.data.shape)
    #             print()
                
    # with torch.no_grad():
    #     x = torch.randn(model.input_shape)
    #     y = model(x)
    #     print(y)
        
    torch.save({'state_dict': model.layers.state_dict()}, 'extra/checkpoints/example.pth')
    
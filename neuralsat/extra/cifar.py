from torch.nn import functional as F
import torch.nn as nn
import torch



class CifarConv(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 2, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        self.input_shape = (1, 3, 32, 32)
        self.output_shape = (1, 10)
        
    def forward(self, x):
        return self.layers(x)
    
    
if __name__ == '__main__':
    model = CifarConv()
    x = torch.randn(model.input_shape)
    print(model(x))
    # print(model.layers.state_dict())
    torch.save({'state_dict': model.layers.state_dict()}, 'checkpoints/cifar_conv.pth')
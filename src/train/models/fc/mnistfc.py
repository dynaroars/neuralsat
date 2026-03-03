# import torch.nn.functional as F
import torch.nn as nn
# import torch

# try:
#     from ..models.registry import register_model
# except ModuleNotFoundError:
#     def register_model(func):
#         """
#         Fallback wrapper in case timm isn't installed
#         """
#         return func
                
                
class MNISTFC(nn.Module):

    def __init__(self, n_layers, n_hiddens=256, num_classes=10):
        super().__init__()
        layers = [
            nn.Flatten(), 
            nn.Linear(784, n_hiddens),
            nn.ReLU()
        ]
        
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(n_hiddens, n_hiddens),
                nn.ReLU(),
            ] 
        
        layers += [
            nn.Linear(n_hiddens, num_classes)    
        ]
        print(layers)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x
    

# @register_model
def mnist_256x2(*args, **kwargs):
    return MNISTFC(n_layers=2)

# @register_model
def mnist_256x3(*args, **kwargs):
    return MNISTFC(n_layers=3)

# @register_model
def mnist_256x4(*args, **kwargs):
    return MNISTFC(n_layers=4)

# @register_model
def mnist_256x5(*args, **kwargs):
    return MNISTFC(n_layers=5)

# @register_model
def mnist_256x6(*args, **kwargs):
    return MNISTFC(n_layers=6)

def mnist_small(*args, **kwargs):
    class PaperNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(784, 6),
                nn.ReLU(),
                nn.Linear(6, 6),
                nn.ReLU(),
                nn.Linear(6, 10),
            ])
            
        def forward(self, x):
            return self.layers(x)
    return PaperNet()

# if __name__ == "__main__":
#     model = mnist_256x3()
#     x = torch.randn(1, 1, 28, 28)
#     y = model(x)
#     print(model)
#     print(y)

import torch
import torch.nn as nn

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
        
    def forward(self, x):
        return self.layers(x)
    
    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 50),
            nn.Sigmoid(),
            nn.Linear(50, 25),
            nn.Sigmoid(),
            nn.Linear(25, 10),
        )

    def forward(self, x):
        return self.layer(x)
    
    
def test_sigmoid():
    net = Net()
    x = torch.randn(1, 1, 28, 28)
    print(net(x).shape)
    
    torch.onnx.export(
        net,
        x,
        "fnn_signmoid.onnx",
        verbose=False,
    )
    
    
def test_relu():
    net = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(2, 3), 
        nn.ReLU(), 
        nn.Linear(3, 4), 
        nn.ReLU(), 
        nn.Linear(4, 2)
    )
    print(net)
    x = torch.tensor([[1.0, 2.0]])
    y = net(x)
    print(y)
    torch.onnx.export(
        net, 
        x, 
        "fnn_relu.onnx", 
        verbose=False,
    )
    
    
class ReLUNet(nn.Module):
    
    def __init__(self):
        super(ReLUNet, self).__init__()
        
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = nn.Parameter(torch.tensor([[1, 2], [3, 4], [-5, -6]]).float())
        self.linear1.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
        self.linear2 = nn.Linear(3, 2)
        self.linear2.weight.data = nn.Parameter(torch.tensor([-1, -2, -3, 4, 5, 6]).view(2, 3).float())
        self.linear2.bias.data = nn.Parameter(torch.tensor([2, 3]).float())
        # print( self.linear1.weight.data.shape)
        # print( self.linear1.bias.data.shape)
        
    def forward(self, x):
        
        x = self.linear1(x)
        # x = x.relu()
        x = x.pow(3)
        x = self.linear2(x)
        # x = torch.log(x)
        return x
    
    
def test_relu2():
    
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA import BoundedModule, BoundedTensor

    net = ReLUNet()
    x_U = torch.tensor([[1.0, 2.0]])
    x_L = torch.tensor([[-1.0, -2.0]])
    
    if 0:
        torch.onnx.export(
            net, 
            x_L, 
            "example/relu2.onnx", 
            verbose=False,
        )
    
    print(x_L.shape)
    device = 'cpu'
    
        
    abstractor = BoundedModule(
        model=net, 
        global_input=torch.zeros_like(x_L, device=device),
        bound_opts={
            'relu': 'adaptive', 
            'conv_mode': 'matrix', 
        },
        device=device,
        verbose=False,
    )
    new_x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)
    
    
    # print(lb, ub)
    C = torch.tensor([1, -1]).view(1, 1, 2).float()
    # print(x_L.shape, C.shape)
    # print(abstractor(x_U))
    with torch.no_grad():
        lb, ub = abstractor.compute_bounds(x=(new_x,), method='forward', C=None, bound_upper=False)
        print('lower', lb)
        print('upper', ub)
    
if __name__ == '__main__':
    test_relu2()
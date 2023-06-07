import torch
import torch.nn as nn

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
    
if __name__ == '__main__':
    test_relu()
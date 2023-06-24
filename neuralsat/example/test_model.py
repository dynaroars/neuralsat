import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
    
    
class NetSigmoid(nn.Module):
    
    def __init__(self):
        super(NetSigmoid, self).__init__()
        
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
    net = NetSigmoid()
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
        "example/fnn.onnx", 
        verbose=False,
    )
    
    
class ReLUNet(nn.Module):
    
    def __init__(self):
        super(ReLUNet, self).__init__()
        
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = nn.Parameter(torch.tensor([[1, 2], [3, 4], [5, 6]]).float())
        self.linear1.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
        self.linear2 = nn.Linear(3, 2)
        self.linear2.weight.data = nn.Parameter(torch.tensor([1, 2, 3, -4, -5, -6]).view(2, 3).float())
        self.linear2.bias.data = nn.Parameter(torch.tensor([2, 3]).float())
        # print( self.linear1.weight.data.shape)
        # print( self.linear1.bias.data.shape)
        
        self.linear3 = nn.Linear(2, 3)
        self.linear3.weight.data = nn.Parameter(torch.tensor([[1, 2], [-3, -4], [-5, -6]]).float())
        self.linear3.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        
        # x = torch.log(x)
        # x = x.max(dim=1).values
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
    if 0:
        method = 'forward'
        with torch.no_grad():
            lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=None, bound_upper=True)
            print('[forward] lower', lb)
            print('[forward] upper', ub)
        print()
    else:
        method = 'backward'
        with torch.no_grad():
            lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=None, bound_upper=True)
            print('[backward] lower', lb)
            print('[backward] upper', ub)
    
    
def test_load_model():
    from onnx2torch import convert
    import onnxruntime as ort
    path = '../benchmark/vnncomp23-instances/vnncomp23/ml4acopf/./onnx/14_ieee_ml4acopf.onnx'
    input_shape = (1, 22)
    x = torch.randn(input_shape)
    
    # pytorch
    torch_model = convert(path)
    print(torch_model)
    exit()
    out_torch = torch_model(x)
    # onnx
    ort_sess = ort.InferenceSession(path)
    names = [i.name for i in ort_sess.get_inputs()]
    outputs_ort = ort_sess.run(None, dict(zip(names, [x.numpy()])))
    
    print(torch.max(torch.abs(torch.tensor(outputs_ort) - out_torch)))
    print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-5))

    # print(.shape)
    
    trace, out = torch.jit._get_trace_graph(torch_model, x)
    
    
    
class NetReLU(nn.Module):
    
    def __init__(self):
        super(NetReLU, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 9),
            nn.ReLU(),
            nn.Linear(9, 7),
            nn.ReLU(),
            nn.Linear(7, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
        )

    def forward(self, x):
        return self.layer(x)
    
    
def test_relu3():
    net = NetReLU()
    x = torch.randn(1, 1, 28, 28)
    print(net(x).shape)
    
    torch.onnx.export(
        net,
        x,
        "example/mnist_relu.onnx",
        verbose=False,
    )
    
    
def load(path):
    import onnx2pytorch
    import onnx
    # cifar10_8_255_simplified.onnx
    # path = '/home/droars/Desktop/neuralsat/benchmark/cifar2020/nnet/cifar10_8_255_simplified.onnx'
    
    onnx_model = onnx.load(path)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks={'Reshape': {'fix_batch_size': True}})
    pytorch_model.eval()
    
    print(pytorch_model)
    
    

    
class NetConv(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Linear(5, 512)
        self.bn1 = nn.BatchNorm2d(128)
        self.ct1 = nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.ct2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))
        self.ct3 = nn.ConvTranspose2d(64, 4, kernel_size=(4, 4), stride=(2, 2))
        self.l2 = nn.Linear(3600, 1)

    def forward(self, x):
        x = self.l1(x)
        # print(1, x.shape)
        x = x.view(-1, 128, 2, 2)
        # print(2, x.shape)
        x = self.bn1(x)
        # print(3, x.shape)
        x = self.ct1(x)
        # print(4, x.shape)
        x = self.bn1(x)
        # print(5, x.shape)
        x = x.relu()
        x = self.ct2(x)
        # print(6, x.shape)
        x = x.relu()
        x = self.ct3(x)
        x = x.relu()
        x = x.flatten(1)
        x = self.l2(x)
        return x
    
def test_conv1():
    net = NetConv()
    
    x = torch.randn(1, 5)
    print(net(x).shape)
   
    net.eval()
    torch.onnx.export(
        net,
        x,
        "example/conv.onnx",
        verbose=False,
        opset_version=12,
        
    )
    
    
if __name__ == '__main__':
    # load('/home/droars/Desktop/neuralsat/benchmark/cifar2020/nnet/cifar10_2_255_simplified.onnx')
    test_conv1()
    
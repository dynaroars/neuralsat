import gurobipy as grb
from example.test_model import extract_instance
from verifier.verifier import Verifier 
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 5),
        ])
        
        self.layers[0].weight.data = torch.tensor([
            [1., -1., 2., -2., 3.],
            [1., 2., -3., -4., 5.],
            [-1., -2., 1., -1., -3.],
            [-1., 1., 1., -1., 4.],
        ])
        self.layers[0].bias.data = torch.tensor(
            [1., -1., 2., -4.],
        )
            
        self.layers[2].weight.data = torch.tensor([
            [1., -1., -2., -3.],
            [1., 2., -4., 5.],
            [-1., 5., -1., 3.],
        ])
        self.layers[2].bias.data = torch.tensor(
            [1., -1., 2.],
        )
        
        self.layers[4].weight.data = torch.tensor([
            [1., -2., 3.],
            [2., -4., -5.],
        ])
        self.layers[4].bias.data = torch.tensor(
            [1., -1.],
        )
        
        self.layers[6].weight.data = torch.tensor([
            [1., -2.],
            [-4., -5.],
            [-1., 5.],
            [1., -3.],
            [2., 3.],
        ])
        self.layers[6].bias.data = torch.tensor(
            [1., -1., -2., -3., 1.],
        )
        
        
        
    @torch.no_grad()
    def forward(self, x):
        for layer in self.layers:
            # print(x)
            x = layer(x)
            # print(x)
            # print()
        return x
        # return self.layers(x)



if __name__ == "__main__":
    # model = grb.Model()
    # x0 = model.addVar(name='x0', lb=-1, ub=1)
    # x1 = model.addVar(name='x1', lb=-1, ub=2)
    
    # model.addConstr(x0 + x1 <= 0)
    # model.addConstr(x0 - x1 >= -2)
    
    # a = model.addVar(name='new_var_a', lb=-10, ub=20)
    # b = model.addVar(name='new_var_b', lb=-10, ub=20)
    # model.addConstr(x0 + 2 * x1 >= a)
    # model.addConstr(2*x0 - 3 * x1 == b)
    # model.update()
    # print(model)
    
    # model.remove(a)
    # model.update()
    # print(model)
    # model.write('test_gurobi.lp')
    
    net_path = 'example/gurobi.onnx'
    vnnlib_path = 'example/prop_3.vnnlib'
    device = 'cpu'
    # torch.manual_seed(1)
    
    net = Net()
    x = torch.randn(1, 5)
    print(net(x))
    
    net.eval()
    torch.onnx.export(
        net,
        x,
        net_path,
        verbose=False,
        opset_version=12,
    )
    
    exit()
    
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1,
        device=device,
    )
    
    status = verifier.verify(objectives)
import torch.nn as nn
import torch

from abstractor.auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from helper.network.read_onnx import parse_onnx, _parse_onnx

from train.models.rnn.mnist import *

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, T, num_classes):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: [batch, seq, input] -> [seq, batch, input] for batch_first=False
        x = x.transpose(0, 1)
        x, _ = self.rnn(x) # [seq, batch, hidden]
        x = x.transpose(0, 1) # [batch, seq, hidden]
        x = x[:, -1]
        x = x.relu()
        x = self.linear(x)
        return x
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, T, num_classes):
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: [batch, seq, input] -> [seq, batch, input] for batch_first=False
        x = x.transpose(0, 1)
        x, _ = self.rnn(x) # [seq, batch, hidden]
        x = x.transpose(0, 1) # [batch, seq, hidden]
        x = x[:, -1]
        x = x.relu()
        # x = self.flatten(x)
        x = self.linear(x)
        return x
    
def test_bound(onnx_path, device):
    
    model, input_shape, output_shape = _parse_onnx(onnx_path)
    print(model, input_shape, output_shape)

    polytope = BoundedModule(
        model=model, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
        device=device,
        verbose=True,
    )
    polytope.eval()
    # polytope.visualize('example/scripts/graph')
    
    x_L = torch.randn(input_shape)
    x_U = x_L + 0.05
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)
    lb, ub = polytope.compute_bounds(x=(x,), method='backward', bound_upper=True)
    print(f'{lb=}')
    print(f'{ub=}')
    
def export_onnx(model, dummy_input, output_name):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        verbose=False,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        }
    )
    print(f"Successfully exported to {output_name=}")
    

def test_lstm():
    
    net = LSTMModel(input_size=3, hidden_size=11, num_layers=1, T=10, num_classes=4)
    x = torch.randn(5, 10, 3)  # [batch, seq_len, input_size]
    print(net(x).shape)
   
    output_name = "example/onnx/lstm.onnx"
    export_onnx(net, x, output_name)
    test_bound(output_name)
    
def test_gru():
    # net = GRUModel(input_size=3, hidden_size=11, num_layers=1, T=10, num_classes=4)
    net = mnist_gru_128x2()
    x = torch.randn(5, 1, 28, 28)  # [batch, seq_len, input_size]
    print('output shape:', net(x).shape)
   
    output_name = "example/onnx/gru.onnx"
    export_onnx(net, x, output_name)
    test_bound(output_name, device='cuda')
    
    
    
if __name__ == "__main__":
    # test_lstm()
    test_gru()
    
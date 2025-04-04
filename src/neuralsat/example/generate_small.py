import torch.nn as nn
import torch

import neuralsat as ns

class PaperNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
        ])
        
    def forward(self, x):
        return self.layers(x)
          
def verify(onnx_name, vnnlib_name, timeout, device):
    
    model, input_shape, objectives = ns.test.extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    # print(model)
    # print(f'{input_shape=}')

    v = ns.Verifier(
        net=model,
        input_shape=input_shape,
        batch=100,
        device=device,
    )
    
    status = v.verify(objectives, timeout=timeout)
    
    return status, v.iteration

def test1():
    
    net_name = 'example/onnx/mnistfc-medium-net-554.onnx'
    vnnlib_name = 'example/vnnlib/test.vnnlib'

    verify(net_name, vnnlib_name, 200.0, 'cpu')
    
    
if __name__ == "__main__":
    input_shape = (1, 3)
    vnnlib_name = 'example/vnnlib/fnn.vnnlib'
    net_name = 'example/onnx/fnn.vnnlib'
    
    setting = ns.setting.Settings
    setting.setup(None)
    print(setting)
    
    logger = ns.util.misc.logger.logger
    logger.setLevel(0)
    
    for i in range(10000):
        torch.manual_seed(i)
        
        net = PaperNet()
        net.eval()
        
        torch.onnx.export(
            net,
            torch.zeros(input_shape),
            net_name,
            verbose=False,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )
        status, iteration = verify(net_name, vnnlib_name, 200.0, 'cpu')
        print(f'{i=} {status=} {iteration=}')
        
        if iteration > 0:
            
            torch.onnx.export(
                net,
                torch.zeros(input_shape),
                f'example/onnx/small/fnn_{i}_{iteration}.vnnlib',
                verbose=False,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                }
            )
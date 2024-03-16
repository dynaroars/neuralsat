from pathlib import Path
import torch.nn as nn
import random
import torch
import time
import os

from verifier.verifier import Verifier 
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.objective import Objective, DnfObjectives
from util.misc.logger import logger
from setting import Settings


class PaperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        ])
        self._init_weights()
        
    def random(self, size):
        while True:
            w = (torch.FloatTensor(*size).uniform_(-1, 1) * 10).int().float() / 10
            if (w != 0).all():
                return w
            
    def _init_weights(self):
        self.layers[0].weight.data = self.random(size=(2, 2))
        self.layers[0].bias.data = self.random(size=(2,))
        
        self.layers[2].weight.data = self.random(size=(2, 2))
        self.layers[2].bias.data = self.random(size=(2,))
        
        self.layers[4].weight.data = self.random(size=(2, 2))
        self.layers[4].bias.data = self.random(size=(2,))
        
        self.layers[6].weight.data = self.random(size=(2, 2))
        self.layers[6].bias.data = self.random(size=(2,))
        
        self.layers[8].weight.data = self.random(size=(2, 2))
        self.layers[8].bias.data = self.random(size=(2,))

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)

    def print_w_b(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                print(layer)
                print('\t[+] w:', layer.weight.data.flatten())
                print('\t[+] b:', layer.bias.data.flatten())
                print()


def generate_network(seed):
    # seed = 79616
    if seed is not None:
        torch.manual_seed(seed)
    net = PaperNet()
    x = torch.tensor([[1, 2]]).float()
    net.print_w_b()
    # print(net(x))
    
    net.eval()
    root_dir = 'example/onnx'
    output_name = f'{root_dir}/motivation_example.onnx' #'example/cacmodel.onnx'
    os.system(f'rm -rf {output_name}')
    
    torch.onnx.export(
        net,
        x,
        output_name,
        verbose=False,
        opset_version=12,
    )
    print(f'Exported to {output_name}')
    
    assert os.path.exists(output_name)
    

def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives


def verify():
    net_path = 'example/onnx/motivation_example.onnx'
    vnnlib_path = 'example/vnnlib/motivation_example.vnnlib'
    device = 'cpu'
    
    print('Running test with', net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(status, verifier.iteration)
    return status, verifier.iteration

def save_network(seed):
    net_path = 'example/onnx/motivation_example.onnx'
    new_net_path = f'example/onnx/motivation_example_{seed}.onnx'
    
    vnnlib_path = 'example/vnnlib/motivation_example.vnnlib'
    new_vnnlib_path = f'example/vnnlib/motivation_example_{seed}.vnnlib'
    
    os.system(f'cp {net_path} {new_net_path}')
    os.system(f'cp {vnnlib_path} {new_vnnlib_path}')
    

if __name__ == "__main__":
    seed = None
    Settings.setup_test()
    logger.setLevel(2)
    
    for trial in range(500000):
        print('\n\n[+] Trail:', trial)
        status, iteration = None, 0
            
        seed = random.randint(0, 100000)
        # seed = 98556
        print('seed:', seed)
        try:
            generate_network(seed)
            status, iteration = verify()
        except:
            print('[!] Error occurred')
            pass
        
        if status == 'unsat' and iteration > 2:
            save_network(trial)
            break
        
        time.sleep(0.1)
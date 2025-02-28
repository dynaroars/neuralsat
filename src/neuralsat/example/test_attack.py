import logging
import torch

import neuralsat as ns

def get_activation_shape(name, result):
    def hook(model, input, output):
        result[name] = output.shape
    return hook


def attack(onnx_name, vnnlib_name, timeout, device):
    model, input_shape, objectives = ns.test.extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    print(model)
    print(f'{input_shape=}')
    
    while len(objectives):
        objective = objectives.pop(1)
        
        atk = ns.attacker.Attacker(model, objective, input_shape, device)
        is_attacked, adv = atk.run(timeout=timeout)
        out = None
        cs = None
        if is_attacked:
            out = model(adv)
            cs = objective.cs.flatten().int().detach().cpu().numpy().tolist()
            out = out.flatten().detach().cpu().numpy().tolist()
            # print(out)
        print(f'{objective.ids} {is_attacked=} {cs=} {out=}')

    
if __name__ == "__main__":
    net_name = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_name = 'example/vnnlib/prop_1_0.05.vnnlib'

    # net_name = 'example/onnx/motivation_example.onnx'
    # vnnlib_name = 'example/vnnlib/motivation_example.vnnlib'
    
    attack(net_name, vnnlib_name, 2.0, 'cpu')
    
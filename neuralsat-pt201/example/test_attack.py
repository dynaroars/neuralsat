import logging
import torch
from util.misc.logger import logger
from attacker.attacker import Attacker, PGDAttacker
from test import extract_instance


def get_activation_shape(name, result):
    def hook(model, input, output):
        result[name] = output.shape
    return hook


def attack(onnx_name, vnnlib_name, timeout, device):
    model, input_shape, objectives = extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    print(model)
    print(f'{input_shape=}')
    
    # Define a list to store the outputs
    outputs = []

    # Define the hook function
    # def hook_fn(module, input, output):
    #     outputs.append(output)
        
    # for layer in model.children():
    #     if isinstance(layer, torch.nn.ReLU):
    #         layer.register_forward_hook(hook_fn)
    
    # atk = PGDAttacker(model, objectives, input_shape, device)
    # is_attacked, adv = atk.run(iterations=10, restarts=5, timeout=timeout)

    atk = Attacker(model, objectives, input_shape, device)
    is_attacked, adv = atk.run(timeout=timeout)
    
    # for i, output in enumerate(outputs):
    #     print(f"Output after ReLU layer {i+1}: {output}")
    # print(adv)
    print('adv dtype=', adv.dtype if adv is not None else None)
    if is_attacked:
        assert adv is not None
        return 'sat'
    
    return 'unknown'
    
    
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    net_name = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_name = 'example/vnnlib/prop_1_0.05.vnnlib'

    net_name = 'example/onnx/motivation_example.onnx'
    vnnlib_name = 'example/vnnlib/motivation_example.vnnlib'
    
    preconditions = [
        [],
        [],
    ]
    print(attack(net_name, vnnlib_name, 2.0, 'cpu'))
    
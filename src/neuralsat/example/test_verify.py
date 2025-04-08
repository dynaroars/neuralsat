import logging
import torch

import neuralsat as ns

def verify(onnx_name, vnnlib_name, timeout, device):
    
    model, input_shape, objectives = ns.test.extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    print(model)
    print(f'{input_shape=}')

    v = ns.Verifier(
        net=model,
        input_shape=input_shape,
        batch=100,
        device=device,
    )
    
    status = v.verify(objectives, timeout=timeout)
    
    print(f'{status=}')

    
if __name__ == "__main__":
    net_name    = 'example/onnx/mnistfc-medium-net-554.onnx'
    vnnlib_name = 'example/vnnlib/test.vnnlib'

    net_name    = '../../../code/data/benchmark/cifar10-cnn-one-example/onnx/cifar10_2_255_simplified.onnx'
    vnnlib_name = '../../../code/data/benchmark/cifar10-cnn-one-example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib'
    setting = ns.setting.Settings
    setting.setup(None)
    print(setting)
    
    logger = ns.util.misc.logger.logger
    logger.setLevel(2)
    
    verify(net_name, vnnlib_name, 200.0, 'cuda')
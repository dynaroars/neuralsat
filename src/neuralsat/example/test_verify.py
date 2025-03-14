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
    net_name = 'example/onnx/mnistfc-medium-net-554.onnx'
    vnnlib_name = 'example/vnnlib/test.vnnlib'

    setting = ns.setting.Settings
    setting.setup(None)
    print(setting)
    
    logger = ns.util.misc.logger.logger
    logger.setLevel(2)
    
    verify(net_name, vnnlib_name, 200.0, 'cpu')
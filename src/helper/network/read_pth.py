import onnxruntime as ort
import torch.nn as nn
import numpy as np
import torch
import copy
import onnx

from .read_onnx import custom_quirks
            
def inference_onnx(path, *inputs: np.ndarray):
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


def remove_dropout(module):
    module_output = module
    if isinstance(module, (nn.Dropout)):
        # print("[!] removing Dropout")
        module_output = nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(name, remove_dropout(child))
    del module
    return module_output
    

def fuse_bn(module):
    module_output = module
    if isinstance(module, (nn.Sequential,)):
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(module[idx + 1], nn.BatchNorm2d):
                continue
            # print("[!] fusing BatchNorm2d")
            conv = module[idx]
            bn = module[idx + 1]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (conv.weight * bn.weight[:, None, None, None] * invstd[:, None, None, None])
            if conv.bias is  None:
                conv.bias =  nn.Parameter(torch.zeros(conv.out_channels))
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(name, fuse_bn(child))
    del module
    return module_output



def simplify_model(pytorch_model):
    print("[nn.Sequential]\tFusing BN and dropout")
    model = copy.deepcopy(pytorch_model)
    model = model.eval()
    model = fuse_bn(model)
    # model = remove_dropout(model)
    return model


def parse_pth(pth_path: str, input_shape: list, output_shape: None | list = None) -> tuple:
    custom_quirks['Softmax']['skip_last_layer'] = False
    
    input_shape = tuple(input_shape)
    
    for iter in range(10):
        pytorch_model_raw = torch.load(pth_path, weights_only=False)
        pytorch_model = simplify_model(pytorch_model_raw)
        pytorch_model.eval()
        pytorch_model.to(dtype=torch.get_default_dtype())
        
        if output_shape is None:
            output_shape = tuple(pytorch_model(torch.zeros(input_shape)).shape)
        else:
            output_shape = tuple(output_shape)

        # check conversion
        onnx_path = pth_path.replace('.pth', '.onnx')
        batch = 2
        dummy = torch.randn(batch, *input_shape[1:], dtype=torch.get_default_dtype())
        # print(dummy.shape)
        output_onnx = torch.cat([
            torch.from_numpy(inference_onnx(onnx_path, dummy[i].view(input_shape).float().numpy())[0]).view(output_shape) 
                for i in range(batch)
        ])
        # print('output_onnx:', output_onnx)
        output_pytorch = pytorch_model(dummy).detach()
        # print('output_pytorch:', output_pytorch)
        correct_conversion = torch.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
        print(f'[{iter}] convertion diff:', torch.norm(output_onnx - output_pytorch))
        if correct_conversion:
            print(f'[+] Loaded: {pth_path}')
            return pytorch_model, input_shape, output_shape
    raise NotImplementedError
            
import onnxruntime as ort
import torch.nn as nn
import onnx2pytorch
import numpy as np
import collections
import warnings
import torch
import onnx
import gzip

from beartype import beartype

@beartype
def load_onnx(path: str):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model


@beartype
def inference_onnx(path: str, *inputs: np.ndarray) -> list[np.ndarray]:
    sess = ort.InferenceSession(load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res

@beartype
def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])
    
    if shape[0] not in [-1, 1]:
        return (1, *shape)
    
    return shape
        


@beartype
def parse_onnx(path: str) -> tuple:
    onnx_model = load_onnx(path)
    
    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]
    
    onnx_input_dims = inputs[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    
    orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
    orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims)
    
    batched_input_shape = add_batch(orig_input_shape)
    batched_output_shape = add_batch(orig_output_shape)

    # print(batched_input_shape, batched_output_shape)
    # exit()
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks={'Reshape': {'fix_batch_size': True}})
    pytorch_model.eval()
    # exit()
    
    # check conversion
    correct_conversion = True
    try:
        dummy = torch.randn(batched_input_shape)
        # print(dummy.shape)
        output_pytorch = pytorch_model(dummy).detach().numpy()
        output_onnx = inference_onnx(path, dummy.view(orig_input_shape).numpy())[0]
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-4, 1e-5)
    except:
        warnings.warn(f'Unable to check conversion correctness')
        import traceback; print(traceback.format_exc())
        exit()
    
    if not correct_conversion:
        warnings.warn('Model was converted incorrectly.')
        exit()
    # else:
    #     print(pytorch_model)
    #     print(output_onnx)
    #     print(output_pytorch)
    #     print('DEBUG: correct')
    #     exit()
        
    return pytorch_model, batched_input_shape, batched_output_shape


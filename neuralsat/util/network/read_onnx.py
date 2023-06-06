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
def parse_onnx(path: str) -> tuple:
    onnx_model = load_onnx(path)
    
    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
    # assert all([_ > 0 for _ in orig_input_shape]), print(orig_input_shape)

    if len(onnx_input_dims) == 1:
        batched_input_shape = (1, orig_input_shape[0])
    else:
        batched_input_shape = orig_input_shape
        
    output_shape = tuple(d.dim_value for d in onnx_output_dims) if len(onnx_output_dims) > 1 else (1, onnx_output_dims[0].dim_value)

    # convert ONNX to Pytorch model (experimental=True for supporting batch processing)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True)
    pytorch_model.eval()
    
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
    #     print('DEBUG: correct')
    #     exit()
        
    return pytorch_model, batched_input_shape, output_shape


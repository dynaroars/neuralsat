from beartype import beartype
import onnxruntime as ort
import torch.nn as nn
import numpy as np
import traceback
import warnings
import torch
import onnx
import io

try:
    import onnxsim
    use_onnxsim = True
except:
    print("onnxsim not found. Please install it with `pip install onnxsim`.")
    use_onnxsim = False
    
from helper.misc.error import *
from . import onnx2pytorch

custom_quirks = {
    'Reshape': {
        'fix_batch_size': False
    },
    'Transpose': {
        'merge_batch_size_with_channel': True,
        'remove_gdvb_transpose': True,
    },
    'Softmax' :{
        'skip_last_layer': True
    },
    'Squeeze' :{
        'skip_last_layer': True
    },
    'Conv' :{
        'merge_batch_norm': True
    },
}

@beartype
def _load_onnx(path: str | io.BytesIO):
    # print('Loading ONNX with customized quirks:', custom_quirks)
    # print(type(path))
    if isinstance(path, str):
        onnx_model = onnx.load(path)
    else:
        onnx_model = onnx.load_model_from_string(path.getvalue())
    # print(onnx_model)
    return onnx_model

@beartype
# def inference_onnx(path: str, *inputs: np.ndarray) -> list[np.ndarray]:
def inference_onnx(path: str | io.BytesIO, *inputs: np.ndarray):
    sess = ort.InferenceSession(_load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


@beartype
def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])
    
    if shape[0] not in [-1, 1]:
        return (1, *shape)
    
    return shape
        
def onnxsim_convert(path):
    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(model)
    if not check:
        print(f"[!] ONNXSIM failed to simplify {path}.")
        return None
    return model_simp
    

@beartype
def _parse_onnx(path: str | io.BytesIO, input_shape: None | list = None, output_shape: None | list = None) -> tuple:
    # load model
    onnx_model = _load_onnx(path)
    
    try:
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=custom_quirks)
    except IndexError:
        print(f'[!] onnx2pytorch failed, try onnxsim')
        onnx_model = onnxsim_convert(path)
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=custom_quirks)
    
    # extract shapes
    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]
    # print(f'{inputs = }')
    # print(f'{onnx_model.graph.input = }')
    
    if input_shape is None:
        onnx_input_dims = inputs[0].type.tensor_type.shape.dim
        orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
        batched_input_shape = add_batch(orig_input_shape)
    else:
        orig_input_shape = batched_input_shape = tuple(input_shape)
        
    if output_shape is None:
        onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
        orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims) if len(onnx_output_dims) else (1,)
        batched_output_shape = add_batch(orig_output_shape)
    else:
        batched_output_shape = tuple(output_shape)

    if len(batched_output_shape) > 2:
        pytorch_model = nn.Sequential(pytorch_model, nn.Flatten(start_dim=1))
        batched_output_shape = (batched_output_shape[0], int(np.prod(batched_output_shape[1:])))
    
    pytorch_model.eval()
    pytorch_model.to(torch.get_default_dtype())
    
    if custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
        custom_quirks['Softmax']['skip_last_layer'] = getattr(pytorch_model, 'is_last_removed', {}).get('Softmax', False)
    
    if custom_quirks.get('Squeeze', {}).get('skip_last_layer', False):
        custom_quirks['Squeeze']['skip_last_layer'] = getattr(pytorch_model, 'is_last_removed', {}).get('Squeeze', False)
    
    # print(pytorch_model)
    # print(batched_input_shape, batched_output_shape)
    
    # check conversion
    correct_conversion = True
    try:
        batch = 2
        dummy = torch.randn(batch, *batched_input_shape[1:], dtype=torch.get_default_dtype())
        # print(dummy.shape)
        output_onnx = torch.cat([torch.from_numpy(inference_onnx(path, dummy[i].view(orig_input_shape).float().numpy())[0]).view(batched_output_shape) for i in range(batch)])
        # print('output_onnx:', output_onnx)
        output_pytorch = pytorch_model(dummy).detach().numpy()
        # print('output_pytorch:', output_pytorch)
        correct_conversion = np.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
        # print('correct_conversion:', torch.norm(output_onnx - output_pytorch))
    except:
        raise OnnxConversionError
    
    if not correct_conversion and custom_quirks.get('Conv', {}).get('merge_batch_norm', False):
        raise OnnxMergeBatchNormError
    
    if not correct_conversion and not custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
        raise OnnxOutputAllCloseError

    assert correct_conversion

    # print(pytorch_model)
    # print(batched_input_shape)
    # print(batched_output_shape)
    # print('DEBUG: correct')
    # exit()
        
    return pytorch_model, batched_input_shape, batched_output_shape

@beartype
def is_activation_node(node: onnx.NodeProto):
    # Add more activation functions to this list as needed
    activation_functions = ["relu", "sigmoid", "tanh"]
    return node.op_type.lower() in activation_functions

@beartype
def decompose_onnx(onnx_path: str | io.BytesIO, split_idx: int):
    assert split_idx > 0
    
    # model 
    model = _load_onnx(onnx_path)
    nodes = model.graph.node

    # extractor
    extractor = onnx.utils.Extractor(model)
    
    # find split_idx
    # activation_count = 0
    split_layer = None
    for node_idx, node in enumerate(nodes):
        # if is_activation_node(node):
            # activation_count += 1
            
        if node_idx == split_idx:
            split_layer = node
            break
    # ensure we found the split_idx-th activation function
    # print(f'{split_layer=}')
    if split_layer is None:
        return None, None

    # in-out
    n1_input = [_.name for _ in model.graph.input]
    n2_output = [_.name for _ in model.graph.output]
    
    print(f'{n1_input=}')
    print(f'{n2_output=}')
    print(f'{split_layer.input=}')
    print(f'{split_layer.output=}')
    # prefix model: input -> split_idx
    prefix = extractor.extract_model(n1_input, split_layer.input)
    prefix_buffer = io.BytesIO()
    onnx.save(prefix, prefix_buffer)
    onnx.save(prefix, 'cac_prefix.onnx')
    prefix_buffer.seek(0)
    
    # suffix model: split_idx -> output
    suffix = extractor.extract_model(split_layer.input, n2_output)
    suffix_buffer = io.BytesIO()
    onnx.save(suffix, suffix_buffer)
    onnx.save(suffix, 'cac_suffix.onnx')
    suffix_buffer.seek(0)
    
    return prefix_buffer, suffix_buffer


@beartype
def decompose_pytorch(pytorch_model: onnx2pytorch.ConvertModel, input_shape: tuple, split_idx: int):
    onnx_buffer = io.BytesIO()
    net = pytorch_model.cpu().eval()
    
    # export
    torch.onnx.export(
        net,
        torch.zeros(input_shape),
        onnx_buffer,
        verbose=False,
        opset_version=12, # TODO: matter?
        # input_names=["input"],
        # output_names=["output"],
        # dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'},
        # }
    )
    onnx_buffer.seek(0)
    
    return decompose_onnx(onnx_buffer, split_idx)
    

@beartype
def parse_onnx(path: str | io.BytesIO, input_shape: None | list = None, output_shape: None | list = None) -> tuple:
    while True:
        try:
            return _parse_onnx(path=path, input_shape=input_shape, output_shape=output_shape)
        except OnnxMergeBatchNormError:
            custom_quirks['Conv']['merge_batch_norm'] = False
            continue
        except OnnxOutputAllCloseError:
            # print(f'[{i}] Model was converted incorrectly. Try again.')
            continue
        except OnnxConversionError:
            if not custom_quirks['Reshape']['fix_batch_size']:
                custom_quirks['Reshape']['fix_batch_size'] = True
                continue
            else:
                warnings.warn(f'Unable to convert onnx to pytorch model')
                traceback.print_exc()
                exit()
        except SystemExit:
            exit()
        except:
            warnings.warn(f'Unable to convert onnx to pytorch model')
            traceback.print_exc()
            exit()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import onnxruntime as ort
from pathlib import Path
import numpy as np
import onnx
import sys


from dnnv.nn.transformers.simplifiers import (simplify as dnnv_simplify, ReluifyMaxPool) # type: ignore
from dnnv.nn import parse # type: ignore
from onnxsim import simplify as onnxsim_simplify# type: ignore

def inference_onnx(path, *inputs):
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res[0]

def check_conversion(input_onnx_name, output_onnx_name):
    # check conversion
    onnx_model = onnx.load(output_onnx_name)
    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_input_shape = tuple(d.dim_value for d in onnx_input_dims)
    dummy = np.random.randn(*onnx_input_shape).astype(np.float32)
    o1 = inference_onnx(input_onnx_name, dummy)
    o2 = inference_onnx(output_onnx_name, dummy)
    # print(np.sum(np.abs(o1 - o2)))
    return np.allclose(o1, o2, rtol=1e-4, atol=1e-5)
    
    
def dnnv_convert(input_onnx_name, output_onnx_name):
    os.system(f'rm -rf {output_onnx_name}')
    
    # load  model
    op_graph = parse(Path(input_onnx_name))
    
    # convert model
    sim_op_graph = dnnv_simplify(op_graph, ReluifyMaxPool(op_graph))
    
    # export model
    sim_op_graph.export_onnx(output_onnx_name)
    
    # check
    if check_conversion(input_onnx_name, output_onnx_name):
        print("[+] DNNV exported to:", output_onnx_name)
        return True
    
    print(f"[!] DNNV failed to simplify {input_onnx_name}.")
    os.system(f'rm -rf {output_onnx_name}')
    return False


def onnxsim_convert(input_onnx_name, output_onnx_name):
    os.system(f'rm -rf {output_onnx_name}')
    
    # load  model
    model = onnx.load(input_onnx_name)
    
    # convert model
    model_simp, check = onnxsim_simplify(model)
    
    # check
    if not check:
        print(f"[!] ONNXSIM failed to simplify {input_onnx_name}.")
        os.system(f'rm -rf {output_onnx_name}')
        return False
    
    # export model
    onnx.save(model_simp, output_onnx_name)
    
    # check
    if check_conversion(input_onnx_name, output_onnx_name):
        print("[+] ONNXSIM exported to:", output_onnx_name)
        return True
    
    print(f"[!] ONNXSIM failed to simplify {input_onnx_name}.")
    os.system(f'rm -rf {output_onnx_name}')
    return False
    
    
def convert(input_onnx_name, output_onnx_name_wo_ext):
    os.makedirs(os.path.dirname(output_onnx_name_wo_ext), exist_ok=True)
    
    dnnv_output = output_onnx_name_wo_ext + '-dnnv.onnx'
    onnxsim_output = output_onnx_name_wo_ext + '-onnxsim.onnx'

    check_dnnv = dnnv_convert(input_onnx_name, dnnv_output)
    if check_dnnv:
        check_onnxsim = onnxsim_convert(dnnv_output, onnxsim_output)
    else:
        check_onnxsim = onnxsim_convert(input_onnx_name, onnxsim_output)
        
    return (check_dnnv or check_onnxsim)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "expected 1 arguments: [input w/ .onnx, output w/o .onnx]"
    convert(sys.argv[1], sys.argv[2])
    # python3 util/network/simplify_onnx.py example/onnx/mnist-net_256x2.onnx ./output/mnist-simplified

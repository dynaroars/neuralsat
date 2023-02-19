
import onnxruntime as ort
from pathlib import Path
import numpy as np
import time
import sys
import os

from dnnv.nn.transformers.simplifiers import (simplify, ReluifyMaxPool)
from dnnv.nn import parse
import onnx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_onnx(path):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model


def inference_onnx(path, *inputs):
    sess = ort.InferenceSession(load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res

def main():
    """main entry point"""

    assert len(sys.argv) == 2, "expected 1 arguments: [input .onnx filename]"

    onnx_filename = sys.argv[1]
    os.makedirs('outputs', exist_ok=True)
    os.system(f'rm -rf outputs/*')
    os.system(f'cp {onnx_filename} outputs')
    
    op_graph = parse(Path(onnx_filename))

    # print("[+] Starting...")
    t = time.perf_counter()
    op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))
    op_graph = simplify(op_graph)
    diff = time.perf_counter() - t
    # print(f"\t- Simplify runtime: {diff}")
    
    # print("[+] Exporting...")
    t = time.perf_counter()
    output_onnx_name = f'outputs/{os.path.basename(onnx_filename)[:-5]}_simplified.onnx'
    op_graph.export_onnx(output_onnx_name)
    diff = time.perf_counter() - t
    # print(f"\t- Export runtime: {diff}")
    
    onnx_model = load_onnx(output_onnx_name)
    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_input_shape = tuple(d.dim_value for d in onnx_input_dims)
    
    dummy = np.random.randn(*onnx_input_shape).astype(np.float32)
    o1 = inference_onnx(onnx_filename, dummy)
    o2 = inference_onnx(output_onnx_name, dummy)
    
    assert np.allclose(o1, o2), f"Failed to convert {onnx_filename}"
    print("[+] Exported to:", output_onnx_name)
    
    

if __name__ == "__main__":
    main()

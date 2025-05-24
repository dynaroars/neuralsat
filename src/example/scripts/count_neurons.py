from helper.network.onnx2pytorch import ConvertModel
from helper.misc.utility import recursive_walk
import torch
import onnx
import os

def count(path):
    print(path)

    onnx_model = onnx.load(path)
    model = ConvertModel(onnx_model, enable_recording=True)
    # print(model)
    
    x = torch.randn(1, 3, 32, 32)
    y, hs = model(x)
    assert all([h.shape[0]==1 for h in hs])
    # print(y.shape)
    print(os.path.basename(path), 'neurons:', sum([h.numel() for h in hs]))
    
if __name__ == "__main__":
    files = []
    for file in recursive_walk('/home/roars/decomposition/benchmarks/'):
        if not file.endswith('.onnx'):
            continue

        if file.endswith('_bak.onnx'):
            continue
        files.append(file)
        # break
    
    # file = files[4]
        count(file)
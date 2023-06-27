from pathlib import Path
import time
import os

import warnings
warnings.filterwarnings(action='ignore')

from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.objective import Objective, DnfObjectives
from verifier.verifier import Verifier 
from util.misc.result import ReturnStatus

def evaluate_one(net_path, vnnlib_path, device='cuda', batch=1000):
    vnnlib_path = Path(vnnlib_path)
    device = 'cuda'
    
    print(
        f'\nRunning test:\n\t'
        f'- net: {net_path}\n\t'
        f'- vnnlib: {vnnlib_path}\n\t'
        f'- batch: {batch}\n\t'
        f'- device: {device}\n'
    )

    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    model.to(device)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=batch,
        device=device,
    )
    
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)
    tic = time.time()
    status = verifier.verify(objectives)
    
    return status, time.time() - tic
    
    
def evaluate_benchmark(result_file, root_dir, benchmark, device='cuda', batch=1000):
    lines = open(os.path.join(root_dir, benchmark, 'instances.csv')).read().strip().split('\n')
    with open(result_file, 'w') as fp:
        for line in lines:
            net_path, vnnlib_path, _ = line.split(',')
            net_path = os.path.join(root_dir, benchmark, net_path)
            vnnlib_path = os.path.join(root_dir, benchmark, vnnlib_path)
            assert os.path.exists(net_path)
            assert os.path.exists(vnnlib_path)
            
            status, runtime = evaluate_one(net_path, vnnlib_path, device=device, batch=batch)
            print(status, runtime)
            print(f'{os.path.basename(net_path)},{os.path.basename(vnnlib_path)},{status},{runtime:.04f}', file=fp)
        
if __name__ == "__main__":
    root_dir = '/home/droars/Desktop/neuralsat/benchmark/'
    benchmark = 'acasxu'
    device = 'cuda'
    batch = 1000
    
    result_file = f'/home/droars/Desktop/bab-smt/result/{benchmark}.csv'
    
    evaluate_benchmark(
        result_file=result_file,
        root_dir=root_dir, 
        benchmark=benchmark, 
        device=device, 
        batch=batch
    )
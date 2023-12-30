
from pathlib import Path
import torch
import tqdm
import os

from verifier.objective import Objective, DnfObjectives
from util.misc.utility import recursive_walk
from attacker.attacker import Attacker
from test import extract_instance

def attack(onnx_name, vnnlib_name, timeout, device):
    model, input_shape, objectives = extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    
    atk = Attacker(model, objectives, input_shape, device)
    
    is_attacked, adv = atk.run(timeout)
    if is_attacked:
        assert adv is not None
        return 'sat'
    
    return 'unknown'
    
    
if __name__ == "__main__":
    timeout = 0.5
    device = 'cuda'
        
    # print(attack(net_path, vnnlib_path, timeout=1.0, device='cuda'))
    csvs = [f for f in recursive_walk('example/scripts2') if f.endswith('.csv')]
    # csvs = [f for f in recursive_walk('example/vnncomp_23_scripts') if f.endswith('.csv')]
    with open('example/attackable2.txt', 'w') as fp:
        for i, csv in enumerate(csvs):
            pbar = tqdm.tqdm(open(csv).read().strip().split('\n'))
            benchmark = os.path.basename(csv)[:-4]
            pbar.set_description(f'[{i+1}/{len(csvs)}] {benchmark}')
            for line in pbar:
                if line:
                    _, _, _, onnx_path, _, vnnlib_path = line.split(' ')
                    # print(onnx_path, vnnlib_path)
                    try:
                        rv = attack(onnx_path, Path(vnnlib_path), timeout=timeout, device=device)
                    except:
                        print(f'{benchmark},{onnx_path},{vnnlib_path},error', file=fp)
                    else:
                        print(f'{benchmark},{onnx_path},{vnnlib_path},{rv}', file=fp)
        
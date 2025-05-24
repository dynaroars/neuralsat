import argparse
import copy
import tqdm
import json
import os


from attacker.attacker import Attacker
from verifier.verifier import Verifier 
from test import extract_instance


def evaluate_one(net_path, vnnlib_path, device):
    # print('\n\Evaluating test with', net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    
    attacker = Attacker(model, copy.deepcopy(objectives), input_shape, device=device)
    is_attacked, adv = attacker.run(timeout=2.0)
    # print(f'{is_attacked=} {adv=}')
    if is_attacked:
        assert adv is not None
        return False
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=100,
        device=device,
    )
    
    objective = objectives.pop(100)
    verifier._init_abstractor('crown-optimized', objective)
    ret = verifier.abstractor.initialize(objective)
    # print(f'{ret.output_lbs=}')
    return (-2.0 < ret.output_lbs < 0).all()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='fc')
    parser.add_argument('--model_name', default='mnist_256x2')
    parser.add_argument('--benchmark_dir', default='example/generated_benchmark/')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--eps', type=float, default=0.04)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    return args

    
def main():
    args = parse_args()
    output_dir = os.path.join(args.benchmark_dir, args.model_type, f'eps_{args.eps:.06f}_{args.model_name}')
    csv_path = os.path.join(output_dir, 'instances.csv')
    
    stat_path = os.path.join(output_dir, 'stats.json')
    stats = json.load(open(stat_path))
    
    pbar = tqdm.tqdm(open(csv_path).read().strip().split('\n'), desc=f'{args.model_name}')
    valid, invalid = len(stats['valid']), len(stats['invalid'])
    print(f'[{args.model_name}] {valid=}')
    
    extract_dir = os.path.join(args.benchmark_dir, args.model_type, 'valid', f'{args.model_name}')
    os.makedirs(extract_dir, exist_ok=True)
    print(f'{extract_dir=}')
    
    with open(os.path.join(extract_dir, 'instances.csv'), 'w') as fp:
        for id, line in enumerate(pbar):
            if id not in stats['valid']:
                continue
            
            onnx_name, vnnlib_name, _ = line.split(',')
            old_onnx_path = os.path.join(output_dir, onnx_name) 
            old_vnnlib_path = os.path.join(output_dir, vnnlib_name)
            # print(f'{output_dir=}')
            # print(f'{old_onnx_path=}')
            # print(f'{old_vnnlib_path=}')
            
            new_onnx_path = os.path.join(extract_dir, onnx_name) 
            new_vnnlib_path = os.path.join(extract_dir, vnnlib_name)
            
            os.makedirs(os.path.dirname(new_onnx_path), exist_ok=True)
            os.makedirs(os.path.dirname(new_vnnlib_path), exist_ok=True)
            
            if not os.path.exists(new_onnx_path):
                os.system(f'cp "{old_onnx_path}" "{new_onnx_path}"')
                
            if not os.path.exists(new_vnnlib_path):
                os.system(f'cp "{old_vnnlib_path}" "{new_vnnlib_path}"')
                
            print(line, file=fp)
                
            # exit()
            # break
            # is_valid = evaluate_one(onnx_path, vnnlib_path, args.device)
        
        # assert is_valid
        # exit()
    
if __name__ == "__main__":
    main()
import argparse
import copy
import tqdm
import json
import os

from attacker.attacker import Attacker
from verifier.verifier import Verifier 
from test import extract_instance
from setting import Settings


def evaluate_one(net_path, vnnlib_path, device):
    # print('\n\Evaluating test with', net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    
    # attacker = Attacker(model, copy.deepcopy(objectives), input_shape, device=device)
    # is_attacked, adv = attacker.run(timeout=2.0)
    # # print(f'{is_attacked=} {adv=}')
    # if is_attacked:
    #     assert adv is not None
    #     return False
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=100,
        device=device,
    )
    
    # objective = objectives.pop(100)
    # verifier._init_abstractor('crown-optimized', objective)
    # ret = verifier.abstractor.initialize(objective)
    status = verifier.verify(objectives, timeout=10.0)
    if status == 'sat':
        assert verifier.adv is not None
        return False

    assert status in ['early_stop', 'unsat', 'timeout']
    if status in ['early_stop', 'timeout']:
        return False

    if not hasattr(verifier, 'domains_list'):
        return False
    
    if isinstance(verifier.domains_list, list):
        return False
    
    visited = verifier.domains_list.visited
    print(f'{status=} {verifier.iteration=} {visited=}')
    return visited >= 5 and verifier.iteration >= 5
        
    # print(f'{ret.output_lbs=}')
    # return (-2.0 < ret.output_lbs < 0).all()
    
    
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
    Settings.setup(args)
    
    output_dir = os.path.join(args.benchmark_dir, args.model_type, f'eps_{args.eps:.06f}_{args.model_name}')
    csv_path = os.path.join(output_dir, 'instances.csv')
    stat_path = os.path.join(output_dir, 'stats.json')
    if os.path.exists(stat_path):
        stats = json.load(open(stat_path))
    else:
        stats = {'valid': [], 'invalid': []}
    print(f'{csv_path=}')
    
    total, skip = len(stats['valid']), len(stats['invalid'])
    pbar = tqdm.tqdm(open(csv_path).read().strip().split('\n'), desc=f'{args.model_name}')
    for id, line in enumerate(pbar):
        # print(line)
        if (id in stats['valid']) or (id in stats['invalid']):
            continue
        
        onnx_name, vnnlib_name, _ = line.split(',')
        onnx_path = os.path.join(output_dir, onnx_name) 
        vnnlib_path = os.path.join(output_dir, vnnlib_name)
        # print(f'{onnx_path=}') 
        # print(f'{vnnlib_path=}') 
        try:
            is_valid = evaluate_one(onnx_path, vnnlib_path, args.device)
        except NotImplementedError:
            is_valid = False
        except:
            import traceback
            traceback.print_exc()
            break
        # print(f'{is_valid=}')
        if is_valid:
            stats['valid'].append(id)
            total += 1
        else:
            stats['invalid'].append(id)
            skip += 1
        pbar.set_postfix(valid=total, invalid=skip)
        

    with open(stat_path, 'w') as fp:
        json.dump(stats, fp, indent=4)
        
if __name__ == "__main__":
    main()
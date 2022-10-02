import os
import settings

# for benchmark in settings.BENCHMARKS:
#     if benchmark == 'acasxu':
#         continue

#     benchmark_folder = f'benchmark/{benchmark}'
#     instances_file = f'{benchmark_folder}/instances.csv'
#     if not os.path.isfile(instances_file):
#         raise

#     script_folder = f'{benchmark_folder}/script'
#     os.makedirs(script_folder, exist_ok=True)

#     if benchmark == 'mnistfc':
#         dataset = 'mnist'
#     elif benchmark == 'cifar2020':
#         dataset = 'cifar'
#     else:
#         raise

#     lines = open(instances_file).read().strip().split('\n')
#     with open(f'{script_folder}/run_instances.sh', 'w') as fp:
#         for line in lines:
#             nnet_name, spec_name, timeout = line.split(',')
#             nnet_name = f'{benchmark_folder}/{nnet_name}'
#             spec_name = f'{benchmark_folder}/{spec_name}'
#             print(f'python3 main.py --net {nnet_name} --spec {spec_name} --dataset {dataset} --attack', file=fp)

from utils.misc import recursive_walk
import tqdm
import os

BENCHMARKS = ['acasxu', 'mnistfc', 'cifar2020']


def gen_scripts():
    root = os.path.abspath('.')
    script_dir = 'script'
    os.makedirs(script_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        with open(f'{script_dir}/{benchmark}_scripts.sh', 'w') as fp:
            for line in open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'):
                nnet, spec, _ = line.split(',')
                result_file = 'result_' + os.path.basename(nnet).replace('.onnx', '') + '_' + os.path.basename(spec).replace('.vnnlib', '') + '.txt'
                # print('start=$(date +%s.%N)', file=fp)
                if benchmark == 'acasxu':
                    dataset = 'acasxu'
                elif benchmark == 'mnistfc':
                    dataset = 'mnist'
                elif benchmark == 'cifar2020':
                    dataset = 'cifar'

                cmd = f'python3 main.py --net benchmark/{benchmark}/{nnet} --spec benchmark/{benchmark}/{spec} --dataset {dataset} --file {result_file} --attack --solution --timer\n'
                fp.write(cmd)
                # print('dur=$(echo "$(date +%s.%N) - $start" | bc)', file=fp)
                # print(f'printf "{root}/benchmark/{benchmark}/{nnet} {root}/benchmark/{benchmark}/{spec}: %.6f seconds" $dur', file=fp)
                # fp.write('echo\n')
                # fp.write('echo\n\n')


if __name__ == '__main__':
    gen_scripts()
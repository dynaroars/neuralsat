import os
import settings

for benchmark in settings.BENCHMARKS:
    if benchmark == 'acasxu':
        continue

    benchmark_folder = f'benchmark/{benchmark}'
    instances_file = f'{benchmark_folder}/instances.csv'
    if not os.path.isfile(instances_file):
        raise

    script_folder = f'{benchmark_folder}/script'
    os.makedirs(script_folder, exist_ok=True)

    lines = open(instances_file).read().strip().split('\n')
    with open(f'{script_folder}/run_instances.sh', 'w') as fp:
        for line in lines:
            nnet_name, spec_name, timeout = line.split(',')
            nnet_name = f'{benchmark_folder}/{nnet_name}'
            spec_name = f'{benchmark_folder}/{spec_name}'
            print(f'python3 main.py --net {nnet_name} --spec {spec_name}', file=fp)



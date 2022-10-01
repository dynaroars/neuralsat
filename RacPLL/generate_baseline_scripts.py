import os

BENCHMARKS = ['acasxu', 'mnistfc', 'cifar2020']


def gen_scripts():
    root = os.path.abspath('.')
    baseline_dir = 'baseline/script'
    os.makedirs(baseline_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        with open(f'{baseline_dir}/{benchmark}_scripts.sh', 'w') as fp:
            for line in open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'):
                nnet, spec, _ = line.split(',')
                cmd = f'./vnncomp_scripts/run_instance.sh v1 {benchmark} {root}/benchmark/{benchmark}/{nnet} {root}/benchmark/{benchmark}/{spec} log.txt 3600\n'
                fp.write(cmd)


if __name__ == '__main__':

    gen_scripts()

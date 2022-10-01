import os

BENCHMARKS = ['acasxu', 'mnistfc', 'cifar2020']


def gen_scripts_nnenum():
    root = os.path.abspath('.')
    baseline_dir = 'baseline/nnenum'
    os.makedirs(baseline_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        with open(f'{baseline_dir}/{benchmark}_scripts.sh', 'w') as fp:
            fp.write('export OPENBLAS_NUM_THREADS=1\n')
            fp.write('export OMP_NUM_THREADS=1\n\n')

            for line in open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'):
                nnet, spec, _ = line.split(',')
                cmd = f'python3 -m nnenum.nnenum {root}/benchmark/{benchmark}/{nnet} {root}/benchmark/{benchmark}/{spec}\n'
                fp.write(cmd)

def gen_scripts_eran():
    root = os.path.abspath('.')
    baseline_dir = 'baseline/eran'
    os.makedirs(baseline_dir, exist_ok=True)
    pass

def gen_scripts_abcrown():
    pass


if __name__ == '__main__':

    gen_scripts_nnenum()
    # for baseline in os.listdir('baseline'):
    #     print(baseline)
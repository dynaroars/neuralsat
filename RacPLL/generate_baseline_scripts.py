from utils.read_vnnlib import read_vnnlib_simple
from utils.misc import recursive_walk
import tqdm
import os

BENCHMARKS = ['acasxu', 'mnistfc', 'cifar2020']

N_INPUT = {
    'acasxu': 5,
    'mnistfc': 28 * 28,
    'cifar2020': 3 * 32 * 32
}

N_OUTPUT = {
    'acasxu': 5,
    'mnistfc': 10,
    'cifar2020': 10
}

def gen_scripts():
    root = os.path.abspath('.')
    baseline_dir = 'baseline/script'
    os.makedirs(baseline_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        with open(f'{baseline_dir}/{benchmark}_scripts.sh', 'w') as fp:
            for line in open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'):
                nnet, spec, _ = line.split(',')
                result_file = 'result_' + os.path.basename(nnet).replace('.onnx', '') + '_' + os.path.basename(spec).replace('.vnnlib', '') + '.txt'
                cmd = f'./vnncomp_scripts/run_instance.sh v1 {benchmark} {root}/benchmark/{benchmark}/{nnet} {root}/benchmark/{benchmark}/{spec} {result_file} 3600\n'
                fp.write(cmd)


def gen_marabou_spec():
    root = os.path.abspath('.')
    for benchmark in BENCHMARKS:
        baseline_dir = f'baseline/spec/{benchmark}'
        os.makedirs(baseline_dir, exist_ok=True)
        bar = tqdm.tqdm(open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'))
        bar.set_description(benchmark)
        for line in bar:
            nnet, spec, _ = line.split(',')
            # if 'prop_6' not in spec:
            #     continue
            # print(spec)
            spec_list = read_vnnlib_simple(f'{root}/benchmark/{benchmark}/{spec}', N_INPUT[benchmark], N_OUTPUT[benchmark])
            for i, (bound, mat) in enumerate(spec_list):
                # print(bound)
                # print(mat)
                for j, (prop_mat, prop_rhs) in enumerate(mat):
                    # print(prop_mat, prop_rhs)
                    with open(f"{baseline_dir}/{os.path.basename(spec).replace('.vnnlib', '')}_{i}_{j}.vnnlib", 'w') as fp:
                        for bi, (bl, bu) in enumerate(bound):
                            fp.write(f'x{bi} >= {bl}\n')
                            fp.write(f'x{bi} <= {bu}\n')

                        for l, r in zip(prop_mat, prop_rhs):
                            # print(j, l, r)
                            q = []
                            mul = 1
                            for lv in l:
                                if lv ==0:
                                    continue
                                if lv > 0:
                                    break
                                if lv < 0:
                                    mul = -1
                                    break

                            for li, lv in enumerate(l):
                                lv = lv * mul
                                if lv == 1.0:
                                    q.append(f'+y{li}')
                                elif lv == -1:
                                    q.append(f'-y{li}')

                            if len(q) == 1:
                                q[0] = q[0][1:]
                            if mul == 1:
                                fp.write(f'{" ".join(q)} <= {r}\n')
                            else:
                                fp.write(f'{" ".join(q)} >= {-r if r != 0.0 else 0.0}\n')

def gen_marabou_scripts():
    root = os.path.abspath('.')
    baseline_dir = 'baseline/marabou'
    os.makedirs(baseline_dir, exist_ok=True)

    for benchmark in BENCHMARKS:
        benchmark_specs = list(recursive_walk(f'baseline/spec/{benchmark}'))
        with open(f'{baseline_dir}/{benchmark}_scripts.sh', 'w') as fp:
            bar = tqdm.tqdm(open(f'benchmark/{benchmark}/instances.csv').read().strip().split('\n'))
            bar.set_description(benchmark)
            for line in bar:
                nnet, spec, _ = line.split(',')
                spec_name = os.path.basename(spec).replace('.vnnlib', '')
                specs = [s for s in benchmark_specs if spec_name+'_' in s]
                # print(spec_name, specs)
                for si in specs:
                    cmd = f'./build/Marabou {root}/benchmark/{benchmark}/{nnet} {root}/{si} --snc --num-workers=16 --initial-divides=4 --initial-timeout=5 --num-online-divides=4 --timeout-factor=1.5\n'
                    fp.write(cmd)




if __name__ == '__main__':

    gen_marabou_spec()
    # gen_marabou_scripts()
    # gen_scripts()



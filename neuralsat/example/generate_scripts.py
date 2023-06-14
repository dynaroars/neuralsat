import os


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            
            
if __name__ == "__main__":
    BENCHMARK_DIR = '../benchmark/vnncomp23-instances'
    CMD = 'python3 main.py --net {} --spec {}'
    csv_files = [f for f in recursive_walk(BENCHMARK_DIR) if 'instances.csv' in f]
    
    os.makedirs('example/scripts/', exist_ok=True)
    
    for csv in csv_files:
        benchmark_dir = os.path.dirname(csv)
        benchmark = os.path.basename(benchmark_dir)
        print(benchmark, benchmark_dir)
        with open(f'example/scripts/{benchmark}.csv', 'w') as fp:
            for line in open(csv).read().strip().split('\n'):
                net_name, spec_name, timeout = line.split(',')
                net = os.path.join(benchmark_dir, net_name)
                spec = os.path.join(benchmark_dir, spec_name)
                if not os.path.exists(net):
                    net = os.path.join(benchmark_dir, 'onnx', net_name)
                if not os.path.exists(spec):
                    spec = os.path.join(benchmark_dir, 'vnnlib', spec_name)
                    
                if not (os.path.exists(net) and os.path.exists(spec)):
                    print('skip', benchmark, net_name, spec_name)
                    continue
            
                print(CMD.format(net, spec), file=fp)
import os

if __name__ == '__main__':
    os.makedirs('script', exist_ok=True)

    with open('instances.csv', 'w') as fp:
        pass

    fp = open('instances.csv', 'a')

    # property 1
    p = 1
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 2
    p = 2
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if i >= 2:
                    if (i,j) in [(3, 3), (4, 2), (5, 3)]:
                        continue
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)
    
    # property 3
    p = 3
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i,j) in [(1, 7), (1, 8), (1, 9)]:
                    continue
                print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)
    
    # property 4
    p = 4
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i,j) in [(1, 7), (1, 8), (1, 9)]:
                    continue
                print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 5
    p = 5
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (1, 1):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 6
    p = 6
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (1, 1):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 7
    p = 7
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (1, 9):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 8
    p = 8
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (2, 9):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 9
    p = 9
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (3, 3):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    # property 10
    p = 10
    with open(f'script/run_property_{p}.sh', 'w') as f:
        for i in range(1, 6):
            for j in range(1, 10):
                if (i, j) == (4, 5):
                    print(f'python3 main.py --net benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet --spec benchmark/acasxu/spec/prop_{p}.vnnlib', file=f)
                    print(f'nnet/ACASXU_run2a_{i}_{j}_batch_2000.onnx,spec/prop_{p}.vnnlib,300', file=fp)

    fp.close()
    
    os.system('chmod +x script/*')
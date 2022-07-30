'''
Measurement script for ACAS Xu networks. Runs all benchmarks and produces a summary file in the results folder.

Stanley Bak, 2020
'''

import sys
import time
from pathlib import Path
import subprocess

from termcolor import cprint

def main():
    'main entry point'

    start = time.time()

    full_filename = 'results/full_acasxu.dat'
    hard_filename = 'results/hard_acasxu.dat'
    timeout = 600.0

    if len(sys.argv) > 1:
        timeout = 60.0 * float(sys.argv[1])
        print(f"Running measurements with timeout = {timeout} secs")

    instances = []

    for spec in range(1, 5):
        for a_prev in range(1, 6):
            for tau in range(1, 10):
                instances.append([str(a_prev), str(tau), str(spec)])

    instances.append(["1", "1", "5"])
    instances.append(["1", "1", "6"])
    instances.append(["1", "9", "7"])
    instances.append(["2", "9", "8"])
    instances.append(["3", "3", "9"])
    instances.append(["4", "5", "10"])

    acasxu_hard = [["4", "6", "1"],
                   ["4", "8", "1"],
                   ["3", "3", "2"],
                   ["4", "2", "2"],
                   ["4", "9", "2"],
                   ["5", "3", "2"],
                   ["3", "6", "3"],
                   ["5", "1", "3"],
                   ["1", "9", "7"],
                   ["3", "3", "9"]]

    Path("./results").mkdir(parents=True, exist_ok=True)

    with open(hard_filename, "w") as h:
        with open(full_filename, "w") as f:
            for instance in instances:
                a_prev, tau, spec = instance
                net_pair = (int(a_prev), int(tau))

                res_str = "none"
                secs = -1

                cprint(f"\nRunning net {a_prev}-{tau} with spec {spec}", "grey", "on_green")

                res_str, secs = verify_acasxu(net_pair, spec, timeout)

                s = f"{a_prev}_{tau}\t{spec}\t{res_str}\t{secs}"
                f.write(s + "\n")
                f.flush()
                print(s)

                if instance in acasxu_hard:
                    h.write(s + "\n")
                    h.flush()

    mins = (time.time() - start) / 60.0

    print(f"Completed all measurements in {round(mins, 2)} minutes")

def verify_acasxu(net_pair, spec, timeout):
    'returns res_str, secs'

    prev, tau = net_pair

    onnx_path = f'./data/ACASXU_run2a_{prev}_{tau}_batch_2000.onnx'
    spec_path = f'./data/prop_{spec}.vnnlib'

    args = [sys.executable, '-m', 'nnenum.nnenum', onnx_path, spec_path, f'{timeout}', 'out.txt']

    start = time.perf_counter()

    result = subprocess.run(args, check=False)

    if result.returncode == 0:
        with open('out.txt', 'r') as f:
            res_str = f.readline()
    else:
        res_str = 'error_exit_code_{result.returncode}'

    diff = time.perf_counter() - start

    return res_str, diff

if __name__ == '__main__':
    main()

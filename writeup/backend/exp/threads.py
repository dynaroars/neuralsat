'''Find how many torch threads this machine survives.

Raises the thread count one level at a time and runs a cache-heavy matmul
loop at each level. Before and after every level a row is appended to the CSV
and flushed to disk, so after a crash the last ``ok`` row is the highest level
that finished and the level of the last ``start`` row is the crash level.
The ramp also stops early when the kernel log shows a new machine check or
hardware error.

Usage (from the project root, one process):
    python -m writeup.backend.exp.threads --seconds 20
'''

import argparse
import csv
import os
import re
import subprocess
import time

import torch

LOG = 'writeup/backend/data/threads.csv'
FIELDS = ['time', 'level', 'event', 'iterations', 'temp', 'note']
WARNING = re.compile(r'machine check|\bmce:|hardware error|\bedac\b', re.I)


def log(path, event, level='', iterations='', temp='', note=''):
    # Append a CSV row and force it to disk (the machine may die right after).
    row = [time.strftime('%Y-%m-%d %H:%M:%S'), level, event, iterations,
           temp, note]
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(FIELDS)
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    print(*row, flush=True)


def kernel_warnings(since):
    # Machine-check / hardware-error lines in the kernel log since ``since``.
    out = subprocess.run(
        ['journalctl', '-k', '--no-pager', f'--since=@{int(since)}'],
        capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if WARNING.search(line)]


def temperature():
    # Highest temperature reported by ``sensors`` (degrees C), or -1.
    out = subprocess.run(['sensors'], capture_output=True, text=True).stdout
    # first value of each line only; later ones are high/crit limits
    values = re.findall(r'^[^\n:]+:\s+\+(\d+\.\d)°C', out, re.M)
    return max((float(v) for v in values), default=-1.0)


def stress(threads, seconds, size=1024):
    # Matmuls larger than L2 on ``threads`` threads; returns the iterations.
    torch.set_num_threads(threads)
    a = torch.randn(size, size, dtype=torch.float64)
    b = torch.randn(size, size, dtype=torch.float64)
    deadline, iterations = time.perf_counter() + seconds, 0
    while time.perf_counter() < deadline:
        a = (a @ b).div_(size ** 0.5)
        iterations += 1
    return iterations


def main():
    parser = argparse.ArgumentParser(description='thread-count ramp')
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--max', type=int, default=os.cpu_count())
    parser.add_argument('--log', default=LOG, help='CSV to append to')
    args = parser.parse_args()

    began = time.time()
    log(args.log, 'begin', temp=temperature(),
        note=f'cpus={os.cpu_count()} seconds={args.seconds}')
    for level in range(args.start, args.max + 1):
        log(args.log, 'start', level, temp=temperature())
        iterations = stress(level, args.seconds)
        warnings = kernel_warnings(began)
        if warnings:
            log(args.log, 'warning', level, note=warnings[0][:150])
            return
        log(args.log, 'ok', level, iterations, temperature())
    log(args.log, 'done', note=f'max={args.max}')


if __name__ == '__main__':
    main()

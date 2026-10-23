'''Run one (search, abstraction) configuration on all instances.

``crown`` bounds only the l_inf ball, so it cannot use the constraints (they
only filter counterexamples); ``neuralsat`` uses them in its bounds. Results
go to ``{csv-dir}/{search}_{abstraction}.csv``, one row per instance, and a
rerun skips the instances already in the file (``--clear`` starts over).
Exceptions go to ``*_errors.csv`` next to it.

Usage (from the project root, one process; see CLAUDE.md for thread limits):
    python -m writeup.exp.main --search=bab --abstraction=crown
'''

import argparse
import functools
import os
import time

import pandas as pd
import torch
import tqdm

from abstraction.lirpa import crown
from abstraction.neuralsat import neuralsat
from search.bab import bab
from search.dpll import dpll
from writeup.exp.config import (
    ABSTRACTIONS,
    CSV_DIR,
    N_INSTANCES,
    SEARCHES,
    THREADS,
    TIMEOUT,
)
from writeup.exp.instance import make_instance

torch.set_num_threads(THREADS)

RUN = {
    'bab': functools.partial(bab, max_nodes=10 ** 9),
    'dpll': functools.partial(dpll, max_checks=10 ** 9),
}


def with_deadline(bound_fn, deadline):
    # Wrap bound_fn so that it raises TimeoutError after the deadline.
    def bounded(layers, x0, eps, **kwargs):
        if time.perf_counter() > deadline:
            raise TimeoutError
        return bound_fn(layers, x0, eps, **kwargs)
    return bounded


def bound_fn_of(abstraction, constraints):
    if abstraction == 'crown':
        return crown.layer_bounds
    return functools.partial(neuralsat.layer_bounds,
                             constraints=constraints, steps=30)


def solve(search, abstraction, instance, timeout):
    # Run one configuration on one instance: (status, wall-clock seconds).
    model, x0, eps, label, constraints = instance
    start = time.perf_counter()
    bound_fn = with_deadline(bound_fn_of(abstraction, constraints),
                             start + timeout)
    try:
        status, _ = RUN[search](model, x0, eps, label, bound_fn=bound_fn,
                                constraints=constraints)
    except TimeoutError:
        status = 'unknown'  # out of time
    return status, time.perf_counter() - start


def load_completed(csv_path):
    if not os.path.isfile(csv_path):
        return set()
    return set(pd.read_csv(csv_path)['idx'])


def append_row(csv_path, row):
    header = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=header,
                               index=False)


def log_error(csv_path, idx, exc):
    # Exceptions go to {search}_{abstraction}_errors.csv next to the results.
    append_row(csv_path.replace('.csv', '_errors.csv'), {
        'idx': idx, 'type': type(exc).__name__, 'message': str(exc)[:200]})


def main():
    parser = argparse.ArgumentParser(description='run one configuration')
    parser.add_argument('--search', choices=SEARCHES, default='bab')
    parser.add_argument('--abstraction', choices=ABSTRACTIONS,
                        default='crown')
    parser.add_argument('--n', type=int, default=N_INSTANCES)
    parser.add_argument('--timeout', type=float, default=TIMEOUT)
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--clear', action='store_true',
                        help='remove the CSV of this configuration first')
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    name = f'{args.search}_{args.abstraction}'
    csv_path = os.path.join(args.csv_dir, f'{name}.csv')
    if args.clear and os.path.exists(csv_path):
        os.remove(csv_path)
    completed = load_completed(csv_path)
    pending = [i for i in range(args.n) if i not in completed]

    stats = {'completed': 0, 'unsat': 0, 'sat': 0, 'error': 0, 'dt': 0.0}
    with tqdm.tqdm(total=len(pending), desc=name) as pbar:
        for idx in pending:
            instance = make_instance(idx)
            status, dt, error = 'unknown', 0.0, True
            try:
                status, dt = solve(args.search, args.abstraction, instance,
                                   args.timeout)
                error = False
            except Exception as exc:
                log_error(csv_path, idx, exc)
            append_row(csv_path, {
                'idx': idx,
                'search': args.search,
                'abstraction': args.abstraction,
                'eps': instance[2],
                'status': status,
                'error': error,
                'dt': dt,
            })
            stats['completed'] += 1
            stats['unsat'] += status == 'unsat'
            stats['sat'] += status == 'sat'
            stats['error'] += error
            stats['dt'] += dt
            pbar.set_postfix(
                completed=stats['completed'],
                unsat=stats['unsat'],
                sat=stats['sat'],
                error=stats['error'],
                avg_dt=f"{stats['dt'] / stats['completed']:.3f}s")
            pbar.update(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

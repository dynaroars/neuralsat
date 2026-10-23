'''Cactus plot: number of solved instances versus cumulative runtime.

Reads ``{csv-dir}/{search}_{abstraction}.csv`` written by
``writeup.backend.exp.main`` and draws one curve per configuration. An
instance is solved when its status is ``unsat`` (property proven) or ``sat``
(counterexample found). As in the paper plots, instances solved by every
configuration are left out so the curves show the hard ones (``--all`` keeps
them). Prints a markdown summary table and writes it to
``frontend/data/summary.json``; the figures are ``cactus-light.png`` and
``cactus-dark.png`` in ``frontend/static/figure``.

Usage (from the project root):
    python -m writeup.backend.plot.cactus
'''

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from writeup.backend.exp.config import (
    ABSTRACTIONS,
    CSV_DIR,
    FIGURE_DIR,
    SEARCHES,
    SUMMARY_PATH,
    TIMEOUT,
)
from writeup.backend.plot.style import THEMES, rc

SOLVED = ('unsat', 'sat')
LABELS = {'ibp': 'IBP', 'zonotope': 'Zonotope', 'crown': 'CROWN',
          'alphacrown': 'alpha-CROWN', 'neuralsat': 'NeuralSAT', 'bab': 'BaB',
          'dpll': 'DPLL(T)'}
# the colour of each abstraction in THEMES
PALETTE = {'ibp': 'orange', 'zonotope': 'green', 'crown': 'blue',
           'alphacrown': 'purple', 'neuralsat': 'red'}
STYLES = {'bab': '-', 'dpll': '--'}


def label_of(search, abstraction):
    return f'{LABELS[search]} + {LABELS[abstraction]}'


def load_frames(csv_dir):
    # {(search, abstraction): DataFrame} of the configurations that ran.
    frames = {}
    for search in SEARCHES:
        for abstraction in ABSTRACTIONS:
            path = os.path.join(csv_dir, f'{search}_{abstraction}.csv')
            if os.path.isfile(path):
                frames[(search, abstraction)] = pd.read_csv(path)
    return frames


def solved(frame):
    return frame[frame['status'].isin(SOLVED) & ~frame['error']]


def universally_solved(frames):
    # Instance indices solved by every configuration.
    sets = [set(solved(frame)['idx']) for frame in frames.values()]
    return set.intersection(*sets) if sets else set()


def cactus_curve(frame):
    times = np.sort(frame['dt'].to_numpy())
    return np.cumsum(times), np.arange(1, len(times) + 1)


def par2(frame, timeout):
    # Penalised average runtime: an unsolved instance counts twice the limit.
    done = solved(frame)
    penalty = 2 * timeout * (len(frame) - len(done))
    return (float(done['dt'].sum()) + penalty) / max(len(frame), 1)


def summary_rows(frames, timeout=TIMEOUT):
    '''One dict per configuration, best first.

    The score is the number of solved instances; ties go to the lower PAR-2.
    '''
    rows = []
    for (search, abstraction), frame in frames.items():
        done = solved(frame)
        counts = frame['status'].value_counts()
        rows.append({
            'search': LABELS[search], 'abstraction': LABELS[abstraction],
            'key': f'{search}_{abstraction}',
            'solved': len(done), 'total': len(frame),
            'unsat': int(counts.get('unsat', 0)),
            'sat': int(counts.get('sat', 0)),
            'unknown': int(counts.get('unknown', 0)),
            'time': round(float(done['dt'].sum()), 2),
            'par2': round(par2(frame, timeout), 3)})
    rows.sort(key=lambda row: (-row['solved'], row['par2']))
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
    return rows


def summary(rows):
    # Markdown table of the summary rows.
    lines = ['| rank | configuration | solved | unsat | sat | unknown '
             '| solved time (s) | PAR-2 (s) |',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append(
            f'| {row["rank"]} | {row["search"]} + {row["abstraction"]} '
            f'| {row["solved"]}/{row["total"]} | {row["unsat"]} '
            f'| {row["sat"]} | {row["unknown"]} | {row["time"]:.2f} '
            f'| {row["par2"]:.3f} |')
    return '\n'.join(lines)


def draw(frames, exclude, mode, path):
    # One cactus plot in the colours of the theme ``mode``.
    colors = {name: THEMES[mode][key] for name, key in PALETTE.items()}
    with plt.rc_context(rc(mode)):
        fig, ax = plt.subplots(figsize=(6, 4))
        for (search, abstraction), frame in frames.items():
            done = solved(frame)
            done = done[~done['idx'].isin(exclude)]
            if done.empty:
                continue
            runtime, count = cactus_curve(done)
            ax.plot(count, runtime, STYLES[search],
                    color=colors[abstraction], linewidth=2,
                    label=label_of(search, abstraction))
        ax.set_xlabel('Solved instances')
        ax.set_ylabel('Cumulative runtime (s)')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.5)
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)


def plot(csv_dir=CSV_DIR, keep_all=False):
    frames = load_frames(csv_dir)
    if not frames:
        print('[cactus] no result csvs found')
        return None
    rows = summary_rows(frames)
    print(summary(rows))
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, 'w') as handle:
        json.dump(rows, handle, indent=2)
    exclude = set() if keep_all else universally_solved(frames)
    if exclude:
        print(f'[cactus] excluding {len(exclude)} instances solved by all')
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for mode in THEMES:
        path = os.path.join(FIGURE_DIR, f'cactus-{mode}.png')
        draw(frames, exclude, mode, path)
        print(f'[cactus] saved {path}')


def main():
    parser = argparse.ArgumentParser(description='cactus plot')
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--all', action='store_true',
                        help='keep instances solved by every configuration')
    args = parser.parse_args()
    plot(args.csv_dir, args.all)


if __name__ == '__main__':
    main()

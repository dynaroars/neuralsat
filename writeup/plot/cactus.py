'''Cactus plot: number of solved instances versus cumulative runtime.

Reads ``{csv-dir}/{search}_{abstraction}.csv`` written by
``writeup.exp.main`` and draws one curve per configuration. An instance is
solved when its status is ``unsat`` (property proven) or ``sat``
(counterexample found). As in the paper plots, instances solved by every
configuration are left out so the curves show the hard ones (``--all`` keeps
them). Prints a markdown summary table.

Usage (from the project root):
    python -m writeup.plot.cactus
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from writeup.exp.config import ABSTRACTIONS, CSV_DIR, FIGURE_DIR, SEARCHES

SOLVED = ('unsat', 'sat')
LABELS = {'crown': 'CROWN', 'neuralsat': 'NeuralSAT', 'bab': 'BaB',
          'dpll': 'DPLL(T)'}
COLORS = {'crown': 'tab:blue', 'neuralsat': 'tab:red'}
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


def summary(frames):
    # Markdown table: solved count and total solved time per configuration.
    lines = ['| configuration | solved | unsat | sat | time (s) |',
             '|---|---|---|---|---|']
    for (search, abstraction), frame in frames.items():
        done = solved(frame)
        counts = done['status'].value_counts()
        lines.append(
            f'| {label_of(search, abstraction)} | {len(done)}/{len(frame)} '
            f'| {counts.get("unsat", 0)} | {counts.get("sat", 0)} '
            f'| {done["dt"].sum():.2f} |')
    return '\n'.join(lines)


def plot(csv_dir=CSV_DIR, keep_all=False):
    frames = load_frames(csv_dir)
    if not frames:
        print('[cactus] no result csvs found')
        return None
    print(summary(frames))
    exclude = set() if keep_all else universally_solved(frames)
    if exclude:
        print(f'[cactus] excluding {len(exclude)} instances solved by all')

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (search, abstraction), frame in frames.items():
        done = solved(frame)
        done = done[~done['idx'].isin(exclude)]
        if done.empty:
            continue
        runtime, count = cactus_curve(done)
        ax.plot(count, runtime, STYLES[search], color=COLORS[abstraction],
                linewidth=2, label=label_of(search, abstraction))
    ax.set_xlabel('Solved instances')
    ax.set_ylabel('Cumulative runtime (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, 'cactus.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'[cactus] saved {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description='cactus plot')
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--all', action='store_true',
                        help='keep instances solved by every configuration')
    args = parser.parse_args()
    plot(args.csv_dir, args.all)


if __name__ == '__main__':
    main()

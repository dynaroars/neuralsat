'''Print the progress of every (search, abstraction) configuration.

Reads ``{csv-dir}/{search}_{abstraction}.csv`` written by
``writeup.backend.exp.main`` and prints, per configuration, how many of the
instances are done, their statuses and an ETA from the recent mean runtime.

Usage (from the project root):
    python -m writeup.backend.plot.progress
    python -m writeup.backend.plot.progress --markdown
'''

import argparse
import os

import numpy as np
import pandas as pd

from writeup.backend.exp.config import (
    ABSTRACTIONS,
    CSV_DIR,
    N_INSTANCES,
    SEARCHES,
)

ETA_WINDOW = 20
HEADER = ['config', 'done', '%', 'unsat', 'sat', 'unknown', 'error',
          'remain', 'mean dt', 'eta']


def csv_path(csv_dir, search, abstraction):
    return os.path.join(csv_dir, f'{search}_{abstraction}.csv')


def load_results(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['idx', 'status', 'error', 'dt'])
    frame = pd.read_csv(path)
    frame['error'] = frame['error'].astype(str).str.lower().eq('true')
    return frame.drop_duplicates(subset='idx', keep='last')


def format_eta(seconds):
    if seconds < 120:
        return f'{seconds:.0f} s'
    return f'{seconds / 60:.1f} min'


def summarize(search, abstraction, total, csv_dir):
    frame = load_results(csv_path(csv_dir, search, abstraction))
    done = len(frame)
    unsat = int((frame['status'] == 'unsat').sum())
    sat = int((frame['status'] == 'sat').sum())
    error = int(frame['error'].sum())
    remain = total - done
    valid = frame[~frame['error']]
    mean_dt = valid['dt'].mean() if len(valid) else np.nan
    if remain == 0:
        eta = '-'
    elif len(valid):
        eta = format_eta(remain * valid['dt'].tail(ETA_WINDOW).mean())
    else:
        eta = '?'
    return {'config': f'{search}+{abstraction}', 'done': done,
            'total': total, 'pct': 100.0 * done / total, 'unsat': unsat,
            'sat': sat, 'unknown': done - unsat - sat - error,
            'error': error, 'remain': remain, 'mean_dt': mean_dt,
            'eta': eta}


def table_cells(row):
    mean = '-' if pd.isna(row['mean_dt']) else f"{row['mean_dt']:.2f}"
    return [row['config'], f"{row['done']}/{row['total']}",
            f"{row['pct']:.1f}", str(row['unsat']), str(row['sat']),
            str(row['unknown']), str(row['error']), str(row['remain']),
            mean, row['eta']]


def print_table(rows, markdown=False):
    lines = [HEADER] + [table_cells(row) for row in rows]
    if markdown:
        print('| ' + ' | '.join(HEADER) + ' |')
        print('|' + '|'.join(['---'] + ['---:'] * (len(HEADER) - 1)) + '|')
        for cells in lines[1:]:
            print('| ' + ' | '.join(cells) + ' |')
        return
    widths = [max(len(line[i]) for line in lines) for i in range(len(HEADER))]

    def render(cells):
        rest = [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return '  '.join([cells[0].ljust(widths[0])] + rest)

    print(render(HEADER))
    print('-' * (sum(widths) + 2 * (len(widths) - 1)))
    for cells in lines[1:]:
        print(render(cells))


def main():
    parser = argparse.ArgumentParser(description='experiment progress')
    parser.add_argument('--n', type=int, default=N_INSTANCES)
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()
    rows = [summarize(search, abstraction, args.n, args.csv_dir)
            for search in SEARCHES for abstraction in ABSTRACTIONS
            if os.path.exists(csv_path(args.csv_dir, search, abstraction))]
    rows.sort(key=lambda row: (-row['pct'], row['config']))
    print()
    print_table(rows, args.markdown)


if __name__ == '__main__':
    main()

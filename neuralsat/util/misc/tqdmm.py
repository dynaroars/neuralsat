import time

import numpy as np
from tqdm import tqdm


class tqdmm(tqdm):

    def __init__(self, *args, **kwargs):
        if not len(args) and not len(kwargs):
            kwargs.update(total=100)
        super().__init__(*args, **kwargs)

    def set_n(self, n):
        self.update(n - self.n)

    def set_percentage(self, p):
        if p > 1:
            raise ValueError(f'p must be in range [0, 1]. Got {p}.')
        self.total = 100
        self.set_n(round(p * 100))


def process(items):
    new_items = [None] * len(items) * 2
    base_p = np.clip(np.random.normal(0.45, 0.2), 0., 1.)
    valid_mask = np.random.binomial(1, p=np.full(100, base_p))
    valid_items = [new_items[i] for i in range(len(new_items)) if valid_mask[i]]
    return valid_items


def main(init_size=100, batch_size=30):
    items = [None] * init_size

    pbar = tqdmm()
    visited = 0
    while True:
        items, to_be_processed_items = items[:-batch_size], items[-batch_size:]
        new_items = process(to_be_processed_items)
        items.extend(new_items)
        visited += 2 * len(to_be_processed_items)

        time.sleep(0.1)
        pbar.set_percentage(1 - (len(new_items) / visited))
        pbar.set_description(f'n_items={len(items)}')
        if len(items) == 0:
            break
    pbar.close()


if __name__ == '__main__':
    main()

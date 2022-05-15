import time

__all__ = ['FPSTracker']


class FPSTracker:
    def __init__(self, sigma=100):
        self.sigma = sigma
        self.count = 0
        self.average_elapsed_time = None
        self._start = self._end = self._last_elapsed_time = None

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.time()
        self.update(self._end - self._start)

    def tick(self):
        self._start = time.time()

    def tock(self):
        self._end = time.time()
        self.update(self._end - self._start)

    def update(self, elapsed_time):
        self._last_elapsed_time = elapsed_time
        self.count = min(self.count + 1, self.sigma)
        if self.count == 1:
            self.average_elapsed_time = self._last_elapsed_time
        else:
            self.average_elapsed_time = \
                ((self.count - 1) * self.average_elapsed_time + self._last_elapsed_time) / self.count

    @property
    def fps(self):
        return 1. / self.average_elapsed_time if self.average_elapsed_time not in [0, None] else float('nan')

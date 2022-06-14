from collections import defaultdict, OrderedDict
import multiprocessing

import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            
class MP:

    @staticmethod
    def get_workloads(tasks, n_cpus=multiprocessing.cpu_count()):
        wloads = defaultdict(list)
        for i, task in enumerate(tasks):
            cpu_id = i % n_cpus
            wloads[cpu_id].append(task)

        _wloads = [wl for wl in sorted(
            wloads.values(), key=lambda wl: len(wl))]

        return _wloads

    
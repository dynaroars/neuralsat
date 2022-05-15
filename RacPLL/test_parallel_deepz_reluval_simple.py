import torch
import torch.multiprocessing as mp

from abstract.eran.deepz import forward as deepz_forward
from abstract.reluval.reluval import forward as reluval_forward
from utils.read_nnet import NetworkTorch


# =====================================
# Task
# =====================================
class _Task(object):
    """Task class to be sent from main process to subprocesses"""
    pass


class _NNetInitTask(_Task):
    def __init__(self, path):
        super(_NNetInitTask, self).__init__()
        self.path = path

    def __call__(self):
        # called in subprocess
        return NetworkTorch(self.path)


class _NNetForwardTask(_Task):
    def __init__(self, forward_func, lbs, ubs):
        super(_NNetForwardTask, self).__init__()
        self.forward_func = forward_func
        self.lbs = lbs
        self.ubs = ubs

    def __call__(self, nnet):
        # called in subprocess
        return self.forward_func(nnet, self.lbs, self.ubs)


def _racpll_parallel_abstract_execution_worker(task_queue: mp.Queue,
                                               output_queue: mp.Queue,
                                               nnet_initialized_event: mp.Event()):
    # must initialize nnet at least once
    nnet_init_task = task_queue.get()
    assert isinstance(nnet_init_task, _NNetInitTask)
    nnet = nnet_init_task()

    nnet_initialized_event.set()
    while True:
        task = task_queue.get()
        assert isinstance(task, _Task)
        if isinstance(task, _NNetForwardTask):
            ret = task(nnet)
            output_queue.put(ret)
        elif isinstance(task, _NNetInitTask):
            # reinitialize nnet?
            nnet = nnet_init_task()


class RacPLLAbstractProcess:
    def __init__(self, nnet_path, forward_func, **kwargs):
        self.nnet_path = nnet_path
        self.forward_func = forward_func

        #
        self._task_queue = mp.SimpleQueue()
        self._output_queue = mp.Queue()

        #
        self._nnet_initialized_event = mp.Event()

        self.process = mp.Process(
            target=_racpll_parallel_abstract_execution_worker,
            args=(self._task_queue,
                  self._output_queue,
                  self._nnet_initialized_event),
            **kwargs,
        )
        self.process.start()
        self.init_nnet()

    def init_nnet(self, nnet_path=None):
        if nnet_path is not None:
            self.nnet_path = nnet_path
        self._task_queue.put(_NNetInitTask(self.nnet_path))

    def forward(self, lbs, ubs, forward_func=None):
        if forward_func is not None:
            self.forward_func = forward_func
        self._task_queue.put(_NNetForwardTask(self.forward_func, lbs, ubs))

    def result(self, timeout=None):
        return self._output_queue.get(timeout=timeout)


def test():
    from utils.fps_tracker import FPSTracker
    from tqdm import trange

    nnet_path = 'example/corina.nnet'
    lbs = torch.tensor([-5., -4.])
    ubs = torch.tensor([-1., -2.])

    # sequential
    nnet = NetworkTorch(nnet_path)
    fps_tracker = FPSTracker()
    for _ in (pbar := trange(100)):
        with fps_tracker:
            deepz_result = deepz_forward(nnet, lbs, ubs)
            reluval_result = reluval_forward(nnet, lbs, ubs)
        pbar.set_description(f'[Sequential] fps={fps_tracker.fps}')
    print('deepz:', deepz_result)
    print('reluval:', reluval_result)
    print()

    # parallel
    deepz_process = RacPLLAbstractProcess(nnet_path=nnet_path,
                                          forward_func=deepz_forward,
                                          name='deepz',
                                          daemon=True)
    reluval_process = RacPLLAbstractProcess(nnet_path=nnet_path,
                                            forward_func=reluval_forward,
                                            name='reluval',
                                            daemon=True)

    fps_tracker = FPSTracker()
    for _ in (pbar := trange(100)):
        with fps_tracker:
            # start forward together
            deepz_process.forward(lbs, ubs)
            reluval_process.forward(lbs, ubs)
            # fetch results
            deepz_result = deepz_process.result()
            reluval_result = reluval_process.result()
        pbar.set_description(f'[Parallel] fps={fps_tracker.fps}')
    print('deepz:', deepz_result)
    print('reluval:', reluval_result)


if __name__ == '__main__':
    test()

import torch
import torch.multiprocessing as mp
import queue
import threading
from concurrent.futures import as_completed, wait, Future

from abstract.eran.deepz import forward as deepz_forward
from abstract.reluval.reluval import forward as reluval_forward
from utils.read_nnet import NetworkTorch


# =====================================
# Task
# =====================================
class _Task(object):
    """Task class to be sent from main process to subprocesses"""

    def __init__(self, work_id):
        self.work_id = work_id


class _NNetInitTask(_Task):
    def __init__(self, work_id, path):
        super(_NNetInitTask, self).__init__(work_id)
        self.path = path

    def __call__(self):
        # called in subprocess
        return NetworkTorch(self.path)


class _NNetForwardTask(_Task):
    def __init__(self, work_id, forward_func, lbs, ubs):
        super(_NNetForwardTask, self).__init__(work_id)
        self.forward_func = forward_func
        self.lbs = lbs
        self.ubs = ubs

    def __call__(self, nnet):
        # called in subprocess
        return self.forward_func(nnet, self.lbs, self.ubs)


# =====================================
# Work
# =====================================
class _WorkItem(object):
    def __init__(self, future: Future, task: _Task):
        self.future = future
        self.task = task

    @property
    def id(self):
        return self.task.work_id


class _ResultItem(object):
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result


def _racpll_parallel_abstract_execution_worker(task_queue: mp.Queue,
                                               result_queue: mp.Queue,
                                               nnet_initialized_event: mp.Event()):
    # must initialize nnet at least once
    nnet_init_task = task_queue.get()
    assert isinstance(nnet_init_task, _NNetInitTask)
    nnet = nnet_init_task()
    result_queue.put(_ResultItem(nnet_init_task.work_id))

    nnet_initialized_event.set()
    while True:
        task = task_queue.get()
        result_item = _ResultItem(task.work_id)
        try:
            assert isinstance(task, _Task)
            if isinstance(task, _NNetForwardTask):
                result_item.result = task(nnet)
            elif isinstance(task, _NNetInitTask):
                # reinitialize nnet?
                nnet = nnet_init_task()
        except Exception as e:
            result_item.exception = e
        finally:
            if result_item.work_id is not None:
                result_queue.put(result_item)


def _add_call_item_to_queue(pending_work_items,
                            work_ids_queue,
                            task_queue):
    while True:
        if task_queue.full():
            return
        try:
            work_id = work_ids_queue.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                task_queue.put(work_item.task, block=True)
            else:
                del pending_work_items[work_id]
                continue


def _work_management_worker(pending_work_items,
                            work_ids_queue,
                            task_queue):
    while True:
        work_id = work_ids_queue.get()
        task_queue.put(pending_work_items[work_id].task)


def _result_management_worker(pending_work_items,
                              result_queue: mp.SimpleQueue):
    # result_reader = result_queue._reader
    while True:
        # _add_call_item_to_queue(pending_work_items, work_ids_queue, task_queue)

        # ready = mp.connection.wait([result_reader])
        # if result_reader in ready:
        #     result_item = result_reader.recv()
        result_item = result_queue.get()

        work_item = pending_work_items.pop(result_item.work_id, None)
        # work_item can be None if another process terminated (see above)
        if work_item is not None:
            if result_item.exception:
                work_item.future.set_exception(result_item.exception)
            else:
                work_item.future.set_result(result_item.result)
            del work_item
        del result_item


class RacPLLAbstractProcess:
    def __init__(self, nnet_path, forward_func, **kwargs):
        self.nnet_path = nnet_path
        self.forward_func = forward_func

        #
        self._pending_works = {}
        self._work_ids_queue = queue.Queue()
        self._work_queue_count = 0

        self._task_queue = mp.SimpleQueue()
        self._result_queue = mp.SimpleQueue()
        self._result_manager_thread = self._work_manager_thread = None
        self._wakeup_manager_threads()

        #
        self._nnet_initialized_event = mp.Event()

        self.process = mp.Process(
            target=_racpll_parallel_abstract_execution_worker,
            args=(self._task_queue,
                  self._result_queue,
                  self._nnet_initialized_event),
            **kwargs,
        )
        self.process.start()
        self.init_nnet()

    def _wakeup_manager_threads(self):
        if self._work_manager_thread is None:
            self._work_manager_thread = threading.Thread(target=_work_management_worker,
                                                         args=(self._pending_works,
                                                               self._work_ids_queue,
                                                               self._task_queue,
                                                               ),
                                                         daemon=True)
            self._work_manager_thread.start()
        if self._result_manager_thread is None:
            self._result_manager_thread = threading.Thread(target=_result_management_worker,
                                                           args=(self._pending_works,
                                                                 # self._work_ids_queue,
                                                                 # self._task_queue,
                                                                 self._result_queue,
                                                                 ),
                                                           daemon=True)
            self._result_manager_thread.start()

    def init_nnet(self, nnet_path=None):
        if nnet_path is not None:
            self.nnet_path = nnet_path

        f = Future()
        w = _WorkItem(f, _NNetInitTask(self._work_queue_count, self.nnet_path))
        self._pending_works[w.id] = w
        self._work_ids_queue.put(w.id)
        self._work_queue_count += 1
        return f

    def forward(self, lbs, ubs, forward_func=None):
        if forward_func is not None:
            self.forward_func = forward_func

        f = Future()
        w = _WorkItem(f, _NNetForwardTask(self._work_queue_count, self.forward_func, lbs, ubs))
        self._pending_works[w.id] = w
        self._work_ids_queue.put(w.id)
        self._work_queue_count += 1
        return f


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
            deepz_future = deepz_process.forward(lbs, ubs)
            reluval_future = reluval_process.forward(lbs, ubs)
            # fetch results
            deepz_result = deepz_future.result()
            reluval_result = reluval_future.result()
        pbar.set_description(f'[Parallel] fps={fps_tracker.fps}')
    print('deepz:', deepz_result)
    print('reluval:', reluval_result)


if __name__ == '__main__':
    test()

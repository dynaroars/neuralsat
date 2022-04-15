import ctypes
import inspect
import threading

__all__ = [
    'TerminateableThread',
    'ThreadTerminatedError'
]


def _async_raise(tid, exception_type):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exception_type):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exception_type))
    if res == 0:
        raise ValueError(f"invalid thread id {res}")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadTerminatedError(threading.ThreadError):
    pass


class TerminateableThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, raise_exception=True, daemon=None):
        super(TerminateableThread, self).__init__(group, target, name, args, kwargs, daemon=daemon)
        self._terminate_event = threading.Event()
        self._raise_exception = raise_exception
        self._thread_id = None

    def start(self):
        super(TerminateableThread, self).start()
        self.thread_id()

    def thread_id(self):
        if not self.is_alive():
            raise threading.ThreadError("the thread is not active")
        # do we have it cached?
        if self._thread_id is not None:
            return self._thread_id
        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid
        raise AssertionError("unable to determine the thread's id")

    def raise_exception(self, exception_type):
        _async_raise(self.thread_id(), exception_type)

    def terminate(self):
        self._terminate_event.set()
        if self._raise_exception and self.is_alive():
            self.raise_exception(ThreadTerminatedError)

    def is_terminated(self):
        return self._terminate_event.is_set()


if __name__ == '__main__':
    import numpy as np
    import time

    running_threads = []
    start_event = threading.Event()
    start_time = time.time()

    def foo():
        current_thread = threading.current_thread()
        start_event.wait()
        while True:
            try:
                if np.random.rand() < 0.01:
                    now = time.time()
                    print(f'Result found by {current_thread.name} at {now - start_time:.03f}s')
                    # terminate other threads
                    for thread in running_threads:
                        if thread.name != current_thread.name:
                            thread.terminate()
                    break
                else:
                    time.sleep(0.1)
            except ThreadTerminatedError:
                now = time.time()
                print(f'{current_thread.name} terminated  at {now - start_time:.03f}s')
                break

    for i in range(10):  # initialize some threads and add to running_threads list
        thread = TerminateableThread(target=foo, name=f'Thread {i}', daemon=True)
        thread.start()
        running_threads.append(thread)

    # this assures all threads jump inside while True loop concurrently
    start_event.set()

    # wait til done
    for thread in running_threads:
        thread.join()
    print('Done...')

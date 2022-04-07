from dnn_solver.spec import Specification, get_acasxu_bounds, split_bounds
from dnn_solver.dnn_solver import DNNSolver
from archive.terminatable_thread import TerminateableThread, ThreadTerminatedError

from joblib import Parallel, delayed
import threading
import queue
import time

def worker(name, spec):
    solver = DNNSolver(name, spec)
    status = solver.solve()

    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        return status, solution, output
    return status, None, None


i = 3
j = 2
p = 2
name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet'

def test_one():
    bounds = get_acasxu_bounds(p)
    spec = Specification(p=p, bounds=bounds)
    tic = time.time()
    solver = DNNSolver(name, spec)
    print('\nRunning:', name)
    print('Property:', p)
    status = solver.solve()
    print(name, status , time.time() - tic)

    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('lower:', solver.dnn_theorem_prover.lbs_init.data)
        print('upper:', solver.dnn_theorem_prover.ubs_init.data)
        print('solution:', solution)
        print('output:', output)

def test_multithread():

    bounds = get_acasxu_bounds(p)
    splits = split_bounds(bounds, steps=5)

    q = queue.Queue()
    for i, s in enumerate(splits):
        spec = Specification(p=p, bounds=s)
        q.put(spec)

    running_threads = []

    def wrapper_target(f, q, name):
        current_thread = threading.current_thread()
        try:
            while True:
                try:
                    spec = q.get(timeout=1)
                except queue.Empty:
                    return None

                status, solution, output = f(name, spec)
                print('[+]', current_thread.name)
                print('\t- lower:', spec.lower.data)
                print('\t- upper:', spec.upper.data)
                print('\t- status:', status)
                print('\t\t- solution:', solution)
                print('\t\t- output:', output)
                q.task_done()

                if status:
                    print('clear queue')
                    while not q.empty():
                        q.get()
                    for thread in running_threads:
                        if thread.name != current_thread.name:
                            thread.terminate()
        except ThreadTerminatedError:
            # print(f'{current_thread.name} terminated')
            return None


    tic = time.time()
    for i in range(16):  # initialize some threads and add to running_threads list
        thread = TerminateableThread(
            target=wrapper_target, 
            args=(worker, q, name),
            name=f'Thread {i}', 
            daemon=True
        )
        thread.start()
        running_threads.append(thread)


    # wait til done
    for thread in running_threads:
        thread.join()

    print('done', time.time() - tic)


if __name__ == '__main__':
    test_multithread()
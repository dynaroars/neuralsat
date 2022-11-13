import numpy as np
import itertools
import torch
import time
import copy
import math
import multiprocessing
import queue
from collections import Counter

from dnn_solver.dnn_solver import DNNSolver
from abstract.crown import arguments
import settings


class Worker:

    def __init__(self, index, net, shared_queue, shared_status, dataset, spec, initial_splits=4):
        self.shared_queue = shared_queue
        self.shared_status = shared_status
        self.index = index
        self.dataset = dataset
        self.net = net
        self.initial_splits = 4
        self.timeout = 0.5
        self.count = 0
        self.spec = spec

    def get_instance(self, timeout=0.01):
        try:
            res = self.shared_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            res = None

        if res is not None:
            self.shared_status.decrease_task()

        return res

    def put_intance(self, instance):
        # print(f'Thread {self.index} puts:', instance)
        self.shared_queue.put(instance)
        self.shared_status.increase_task()

    def is_sat(self):
        if isinstance(self.shared_status.is_sat, int):
            is_sat = self.shared_status.is_sat == 1
        else:
            is_sat = self.shared_status.is_sat.value == 1
        return is_sat

    def has_error(self):
        if isinstance(self.shared_status.thread_errors, int):
            return self.shared_status.thread_errors > 0
        return self.shared_status.thread_errors.value > 0

    def run(self):
        while True:
            instance = self.get_instance()
            if self.has_error():
                continue

            if (instance is not None) and (not self.is_sat()):

                self.shared_status.update(self.index, 1)
                spec, conflict_clauses = instance
                
                # print(f'Thread {self.index} init')
                solver = DNNSolver(self.net, spec, self.dataset)
                for clause in conflict_clauses:
                    solver._solver._add_clause(clause)
                    solver._solver._generated_conflict_clauses.add(clause)
                # print(f'Thread {self.index} init done')
                # print(f'Thread {self.index}: len(conflict_clauses) =', len(list(solver._solver._generated_conflict_clauses)))
                # print(f'Thread {self.index} starts running')
                try:
                    status = solver.solve(timeout=self.timeout)
                except KeyError as e:
                    # print(f'\t\t\t\tThread {self.index} reset!! =========================================== !!!!! ')
                    continue


                except KeyboardInterrupt as e:
                    print(e)
                    self.shared_status.add_thread_error()
                    continue
                except:
                    continue
                    # raise


                if status == 'TIMEOUT':
                    # print(f'Thread {self.index} got timeout ({self.timeout}) after:', solver.dnn_theorem_prover.count, 'iterations', len(list(solver._solver.get_conflict_clauses())), 'clauses')
                    bounds = spec.get_input_property()
                    lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE, device=self.net.device)
                    upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE, device=self.net.device)
                    tic = time.time()
                    multi_bounds = self.split_multi_bounds(lower.clone(), upper.clone(), self.initial_splits)
                    # print(f'Thread {self.index} split bounds', time.time() - tic)

                    tic = time.time()
                    for l, u in multi_bounds:

                        s_in = (u-l) * torch.rand(50, self.net.n_input) + l
                        # assert torch.all(s_in >= l)
                        # assert torch.all(s_in <= u)
                        s_out = self.net(s_in)
                        for prop_mat, prop_rhs in self.spec.mat:
                            prop_mat = torch.from_numpy(prop_mat).float()
                            prop_rhs = torch.from_numpy(prop_rhs).float()
                            vec = prop_mat.matmul(s_out.t())
                            sat = torch.all(vec <= prop_rhs.reshape(-1, 1), dim=0)
                            if (sat==True).any():
                                # print(sat)
                                self.shared_status.sat()
                                self.shared_status.update(self.index, 0)
                                return

                        s = copy.deepcopy(spec)
                        s.bounds = [(li.item(), ui.item()) for li, ui in zip(l, u)]
                        self.put_intance((s, solver._solver.get_conflict_clauses()))
                    # print(f'Thread {self.index} finding cex', time.time() - tic)
                elif status == 'SAT':
                    # print(f'\t\t\t\tThread {self.index} Solved cex!! =============== !!!!!')
                    self.shared_status.sat()
                    self.shared_status.update(self.index, 0)
                    return
                else: 
                    # print(f'\t\t\t\tThread {self.index} finished:', status, 'remaining:', self.shared_status.tasks if isinstance(self.shared_status.tasks, int) else self.shared_status.tasks.value, self.is_sat())
                    pass
                    
            else:
                self.shared_status.update(self.index, 0)
                # print(f'Thread {self.index} idle')

            # time.sleep(0.5)

            if self.is_sat():
                # print(f'\t\t\t\tThread {self.index} Found cex!! =============== !!!!!')
                pass
                # return

            if sum(list(self.shared_status.threads_in_progress)) == 0 and self.shared_queue.empty():
                # print('quit', self.index)
                return

            if self.has_error():
                # print('quit', self.index)
                return
    
    def estimate_grads(self, lower, upper, steps=3):
        # print(lower.device)
        inputs = [(((steps - i) * lower + i * upper) / steps) for i in range(steps + 1)]
        diffs = torch.zeros(len(lower), dtype=settings.DTYPE, device=lower.device)

        for sample in range(steps + 1):
            pred = self.net(inputs[sample].unsqueeze(0))
            for index in range(len(lower)):
                if sample < steps:
                    l_input = [m if i != index else u for i, m, u in zip(range(len(lower)), inputs[sample], inputs[sample+1])]
                    l_input = torch.tensor(l_input, dtype=settings.DTYPE, device=lower.device).unsqueeze(0)
                    l_i_pred = self.net(l_input)
                else:
                    l_i_pred = pred
                if sample > 0:
                    u_input = [m if i != index else l for i, m, l in zip(range(len(lower)), inputs[sample], inputs[sample-1])]
                    u_input = torch.tensor(u_input, dtype=settings.DTYPE, device=lower.device).unsqueeze(0)
                    u_i_pred = self.net(u_input)
                else:
                    u_i_pred = pred
                diff = [abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)][0]
                diffs[index] += diff.sum()
        return diffs / steps

    def split_multi_bound(self, multi_bound, dim=0, d=2):
        if isinstance(d, int):
            di = d
        else:
            di = d[dim]
        new_multi_bound = []
        for idx, (lower, upper) in enumerate(multi_bound):
            d_lb = lower[dim].clone()
            d_ub = upper[dim].clone()

            d_range = d_ub-d_lb
            d_step = d_range/di
            for i in range(di):
                # print(idx, dim, len(multi_bound), d_step, d_lb, d_ub)
                lower[dim] = d_lb + i*d_step
                upper[dim] = d_lb + (i+1)*d_step
                new_multi_bound.append((lower.clone(), upper.clone()))
                # print('new lower:', new_multi_bound[-1][0])
                # print('new upper:', new_multi_bound[-1][1])
            # print()
        # print('--')
        if dim + 1 < len(upper):
            return self.split_multi_bound(new_multi_bound, dim=dim+1, d=d)
        else:
            return new_multi_bound

    def split_multi_bounds(self, lower, upper, initial_splits=10):
        if initial_splits <= 1:
            return ([(lower, upper)])

        if 1:
            grads = self.estimate_grads(lower, upper, steps=3)
            smears = (grads.abs() + 1e-6) * (upper - lower + 1e-6)
            # print(smears)
            # print(smears.argmax())
            split_multiple = initial_splits / smears.sum()
            # print(split_multiple)
            num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]
            # num_splits = [5 if i >= 5 else i for i in num_splits]
            # print(f'\t[{self.count}] num_splits 1:', num_splits)
            # exit()
            # num_splits = self.balancing_num_splits(num_splits)
            # print(num_splits)
            assert all([x>0 for x in num_splits])
        else:
            num_splits = [1] * self.net.n_input
        if not any([x>1 for x in num_splits]):
            num_splits[(upper - lower).argmax()] += 1
            # print(num_splits)
        # assert any([x>1 for x in num_splits])
        # print(num_splits)
        # exit()

        return self.split_multi_bound([(lower, upper)], d=num_splits)

    def balancing_num_splits(self, num_splits, max_batch=4):
        num_splits = np.array(num_splits)
        idx = np.argmax(num_splits)
        num_splits = [1] * len(num_splits)
        num_splits[idx] += 1
        return num_splits
        # while True:
        #     idx = np.argmin(num_splits)
        #     num_splits[idx] += 1
        #     if math.prod(num_splits) > max_batch:
        #         num_splits[idx] -= 1
        #         break
        # return num_splits.tolist()



def worker_func(worker_index, net, shared_queue, shared_status, dataset, spec):
    w = Worker(worker_index, net, shared_queue, shared_status, dataset, spec)
    w.run()

class SharedStatusMP:

    def __init__(self, n_proc):
        self.mutex = multiprocessing.Lock()
        self.threads_in_progress = multiprocessing.Array('i', [1]*n_proc)
        self.tasks = multiprocessing.Value('i', 1)
        self.thread_errors = multiprocessing.Value('i', 0)
        self.is_sat = multiprocessing.Value('i', 0)

    def update(self, index, value):
        did_lock = self.mutex.acquire()
        self.threads_in_progress[index] = value
        if did_lock:
            self.mutex.release()


    def sat(self):
        did_lock = self.mutex.acquire()
        self.is_sat.value = 1
        if did_lock:
            self.mutex.release()


    def increase_task(self):
        did_lock = self.mutex.acquire()
        self.tasks.value += 1
        if did_lock:
            self.mutex.release()

    def decrease_task(self):
        did_lock = self.mutex.acquire()
        self.tasks.value -= 1
        if did_lock:
            self.mutex.release()

    def add_thread_error(self):
        did_lock = self.mutex.acquire()
        self.thread_errors.value = 1
        if did_lock:
            self.mutex.release()


         
class SharedStatus:

    def __init__(self, n_proc):
        self.threads_in_progress = [1]*n_proc
        self.tasks = 1
        self.thread_errors = 0
        self.is_sat = 0

    def update(self, index, value):
        self.threads_in_progress[index] = value

    def increase_task(self):
        self.tasks += 1

    def decrease_task(self):
        self.tasks -= 1

    def add_thread_error(self):
        self.thread_errors = 1

    def sat(self):
        self.is_sat = 1


class DNNSolverMulti:

    def __init__(self, net, spec, dataset, initial_splits=10):

        self.net = net
        self.new_specs = []
        self.dataset = dataset

        self.spec = spec

    def solve(self):
        print('Solve multi demo')
        N_PROCS = arguments.Config["general"]["n_procs"]

        shared_queue = multiprocessing.Queue()
        shared_queue.put((self.spec, set([])))

        if N_PROCS > 1:        
            processes = []
            
            shared_status = SharedStatusMP(N_PROCS)
            
            for index in range(N_PROCS):
                p = multiprocessing.Process(target=worker_func, args=(index, self.net, shared_queue, shared_status, self.dataset, self.spec))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            n_errors = shared_status.thread_errors.value
            is_sat = shared_status.is_sat.value

        else:
            # shared_queue = 
            shared_status = SharedStatus(N_PROCS)
            worker_func(0, self.net, shared_queue, shared_status, self.dataset, self.spec)

            n_errors = shared_status.thread_errors
            is_sat = shared_status.is_sat

        
        if n_errors > 0:
            return 'UNKNOWN'
        if is_sat == 0:
            return 'UNSAT'
        return 'SAT'


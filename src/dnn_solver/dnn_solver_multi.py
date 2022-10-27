import numpy as np
import itertools
import torch
import time
import copy
import math
import multiprocessing
import queue

from dnn_solver.dnn_solver import DNNSolver
import settings

def naive_split_bounds(bounds, steps=3):
    lower = bounds['lbs']
    upper = bounds['ubs']

    bs = [(l, u) for l, u in zip(lower, upper)]
    bs = [torch.linspace(b[0], b[1], steps=steps) for b in bs]
    bs = [[torch.Tensor([b[i], b[i+1]]) for i in range(b.shape[0] - 1)] for b in bs]
    bs = itertools.product(*bs)
    splits = [{'lbs': torch.Tensor([_[0] for _ in b]),
               'ubs': torch.Tensor([_[1] for _ in b])} for b in bs]
    random.shuffle(splits)
    return splits


class Worker:

    def __init__(self, index, net, shared_queue, shared_status, dataset, initial_splits=4):
        self.shared_queue = shared_queue
        self.shared_status = shared_status
        self.index = index
        self.dataset = dataset
        self.net = net
        self.initial_splits = initial_splits

    def get_instance(self, timeout=0.01):
        try:
            res = self.shared_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            res = None
        return res

    def put_intance(self, instance):
        # print(f'Thread {self.index} puts:', instance)
        self.shared_queue.put(instance)

    def run(self):
        while True:
            # print(list(self.shared_status.threads_in_progress))
            # print(len(set(self.shared_queue)))
            instance = self.get_instance()
            if instance is not None:
                self.shared_status.update(self.index, True)
                spec, conflict_clauses = instance
                
                solver = DNNSolver(self.net, spec, self.dataset)
                for clause in conflict_clauses:
                    solver._solver._add_clause(clause)
                    solver._solver._generated_conflict_clauses.add(clause)

                # print(f'Update {self.index} to True')
                # self.shared_status[self.index] = True
                # time.sleep(1)
                # print(f'Thread {self.index}: len(conflict_clauses) =', len(list(solver._solver._generated_conflict_clauses)))
                status = solver.solve(timeout=1)
                if status == 'TIMEOUT':
                    bounds = spec.get_input_property()
                    lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE, device=self.net.device)
                    upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE, device=self.net.device)
                    multi_bounds = self.split_multi_bounds(lower.clone(), upper.clone(), self.initial_splits)
                    for l, u in multi_bounds:
                        s = copy.deepcopy(spec)
                        s.bounds = [(li.item(), ui.item()) for li, ui in zip(l, u)]
                        # self.new_specs.append(s)
                        # print(s.get_input_property())
                        self.put_intance((s, solver._solver._generated_conflict_clauses))
                else: 
                    print(f'Thread {self.index} finished:', status, solver._solver._generated_conflict_clauses)
                # self.put_intance([(i+self.index+1) for i in instance])
                # self.put_intance([(i+self.index+2) for i in instance])
            else:
                self.shared_status.update(self.index, False)
                # print(f'Update {self.index} to False')
                # self.shared_status.update(False)
                # self.shared_status[self.index] = False
                # print(f'Thread {self.index} idles')
            # time.sleep(0.5)
            if sum(list(self.shared_status.threads_in_progress)) == 0 and self.shared_queue.empty():
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
        num_splits = self.balancing_num_splits(num_splits)
        print(num_splits)
        assert all([x>0 for x in num_splits])
        return self.split_multi_bound([(lower, upper)], d=num_splits)



    def balancing_num_splits(self, num_splits, max_batch=8):
        num_splits = np.array(num_splits)
        while True:
            idx = np.argmin(num_splits)
            num_splits[idx] += 1
            if math.prod(num_splits) > max_batch:
                num_splits[idx] -= 1
                break
        return num_splits.tolist()



def worker_func(worker_index, net, shared_queue, shared_status, dataset):
    w = Worker(worker_index, net, shared_queue, shared_status, dataset)
    w.run()

class SharedStatus:

    def __init__(self, n_proc):
        # self.status = {i: True for i in range(n_proc)}
        self.mutex = multiprocessing.Lock()
        self.threads_in_progress = multiprocessing.Array('i', [1]*n_proc)
        print(list(self.threads_in_progress))
        # self.threads_in_progress.value = 0

    def update(self, index, value):
        did_lock = self.mutex.acquire()
        self.threads_in_progress[index] = value
        if did_lock:
            self.mutex.release()


class DNNSolverMulti:

    def __init__(self, net, spec, dataset, initial_splits=10):

        self.net = net
        self.new_specs = []
        self.dataset = dataset

        # bounds = spec.get_input_property()
        # self.lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE)
        # self.upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE)

        # print(self.lower)
        # print(self.upper)
        # print()
        self.spec = spec

        # self.multi_bounds = self.split_multi_bounds(lower.clone(), upper.clone(), initial_splits)

        # for l, u in self.multi_bounds:
        #     s = copy.deepcopy(spec)
        #     s.bounds = [(li.item(), ui.item()) for li, ui in zip(l, u)]
        #     self.new_specs.append(s)



    # def solve(self):
    #     for idx, spec in enumerate(self.new_specs):
    #         if idx != 5:
    #             continue
    #         solver = DNNxSolver(self.net, spec, self.dataset)
    #         print(f'[{idx}] lower:', self.multi_bounds[idx][0])
    #         print(f'[{idx}] upper:', self.multi_bounds[idx][1])
    #         tic = time.time()
    #         status = solver.solve(timeout=20)
    #         print(f'{idx}/{len(self.new_specs)}', status, time.time() - tic)
    #         break

    def solve(self):
        print('Solve multi demo')
        # exit()
        shared_queue = multiprocessing.Queue()
        shared_queue.put((self.spec, set([])))
        processes = []

        N_PROCS = 4
        shared_status = SharedStatus(N_PROCS)

        for index in range(N_PROCS):
            p = multiprocessing.Process(target=worker_func, args=(index, self.net, shared_queue, shared_status, self.dataset))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # print(shared_queue)


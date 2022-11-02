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

    def __init__(self, index, net, shared_queue, shared_status, dataset, spec, abstractor, initial_splits=4):
        self.shared_queue = shared_queue
        self.shared_status = shared_status
        self.index = index
        self.dataset = dataset
        self.net = net
        self.initial_splits = 4
        self.timeout = 0.1
        self.count = 0
        self.spec = spec
        self.abstractor = abstractor
        # print(index, id(spec))

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

    def run(self):
        while True:
            # if self.index == 0:
            #     print('number of tasks:', self.shared_status.tasks.value)
            # print(list(self.shared_status.threads_in_progress))
            # print(len(set(self.shared_queue)))
            instance = self.get_instance()
            if self.shared_status.thread_errors.value > 0:
                continue


            if (instance is not None) and (not self.is_sat()):
                # self.count += 1
                # if self.count % 20:
                #     self.timeout += 0.5

                self.shared_status.update(self.index, 1)
                spec, conflict_clauses = instance
                
                print(f'Thread {self.index} init')
                solver = DNNSolver(self.net, spec, self.dataset)
                for clause in conflict_clauses:
                    solver._solver._add_clause(clause)
                    solver._solver._generated_conflict_clauses.add(clause)
                print(f'Thread {self.index} init done')

                # print(f'Update {self.index} to True')
                # self.shared_status[self.index] = True
                # time.sleep(1)
                # print(f'Thread {self.index}: len(conflict_clauses) =', len(list(solver._solver._generated_conflict_clauses)))
                print(f'Thread {self.index} starts running')
                try:
                    status = solver.solve(timeout=self.timeout)
                except KeyError as e:
                    print(f'\t\t\t\tThread {self.index} reset!! =========================================== !!!!! ')
                    # self.timeout -= 1
                    # self.put_intance((spec, set([])))
                    continue

                    # Error
                    # src/sat_solver/custom_sat_solver.py", line 255, in _conflict_resolution
                    # conflict_clause.remove(-last_literal)

                #     bounds = spec.get_input_property()
                #     print(bounds['lbs'])
                #     print(bounds['ubs'])
                #     self.shared_status.add_thread_error()
                #     print(f'error at Thread {self.index}!!')
                #     return
                except KeyboardInterrupt as e:
                    print(e)
                    self.shared_status.add_thread_error()
                    continue
                except:
                    # self.shared_status.add_thread_error()
                    continue
                    # raise


                if status == 'TIMEOUT':
                    print(f'Thread {self.index} got timeout ({self.timeout}) after:', solver.dnn_theorem_prover.count, 'iterations', len(list(solver._solver.get_conflict_clauses())), 'clauses')
                    bounds = spec.get_input_property()
                    lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE, device=self.net.device)
                    upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE, device=self.net.device)
                    tic = time.time()
                    multi_bounds = self.split_multi_bounds(lower.clone(), upper.clone(), self.initial_splits)
                    # print(f'Thread {self.index} split bounds', time.time() - tic)

                    tic = time.time()
                    for l, u in multi_bounds:
                        # if torch.allclose(l, u):
                        #     print(l)
                        #     print(u)
                        #     raise


                        # s_in = (u-l) * torch.rand(10, self.net.n_input) + l
                        # # assert torch.all(s_in >= l)
                        # # assert torch.all(s_in <= u)
                        # s_out = self.net(s_in)
                        # for prop_mat, prop_rhs in self.spec.mat:
                        #     prop_mat = torch.from_numpy(prop_mat).float()
                        #     prop_rhs = torch.from_numpy(prop_rhs).float()
                        #     vec = prop_mat.matmul(s_out.t())
                        #     sat = torch.all(vec <= prop_rhs.reshape(-1, 1), dim=0)
                        #     if (sat==True).any():
                        #         self.shared_status.sat()
                        #     # print(sat)
                        #         self.shared_status.update(self.index, 0)
                        #         return

                        # if self.spec.check_solution(self.net((l+u)/2.0)):
                        #     self.shared_status.sat()

                        s = copy.deepcopy(spec)
                        s.bounds = [(li.item(), ui.item()) for li, ui in zip(l, u)]
                        # self.new_specs.append(s)
                        # print(s.get_input_property())
                        # self.put_intance((s, []))
                        self.put_intance((s, solver._solver.get_conflict_clauses()))
                    # print(f'Thread {self.index} finding cex', time.time() - tic)
                else: 
                    print(f'\t\t\t\tThread {self.index} finished:', status, 'remaining:', self.shared_status.tasks if isinstance(self.shared_status.tasks, int) else self.shared_status.tasks.value, self.shared_status.is_sat.value)
                    
                # self.put_intance([(i+self.index+1) for i in instance])
                # self.put_intance([(i+self.index+2) for i in instance])
            else:
                self.shared_status.update(self.index, 0)
                # print(f'Update {self.index} to False')
                # self.shared_status.update(False)
                # self.shared_status[self.index] = False
                print(f'Thread {self.index} idle')
            # time.sleep(0.5)

            if self.is_sat():
                print(f'\t\t\t\tThread {self.index} Found cex!! =============== !!!!!', 'remaining:', self.shared_status.tasks.value)
                # return

            if sum(list(self.shared_status.threads_in_progress)) == 0 and self.shared_queue.empty():
                return

            if isinstance(self.shared_status.thread_errors, int):
                if self.shared_status.thread_errors > 0:
                    print('quit thoi', self.index)
                    # if self.index == 0:
                    #     while True:
                    #         res = self.get_instance()
                    #         if res == None:
                    #             break
                    return
            else: 
                if self.shared_status.thread_errors.value > 0:
                    print('quit thoi', self.index)
                    # if self.index == 0:
                    #     while True:
                    #         res = self.get_instance()
                    #         if res == None:
                    #             break
                    return
    
    def split_multi_bounds_2(self, lower, upper, args=None):
        # print(lower)
        # print(upper)
        ls = []
        us = []
        save_mb = []
        for i in range(len(lower)):
            initial_splits = [1] * len(lower)
            initial_splits[i] += 1
            multi_bounds = self.split_multi_bound([(lower.clone(), upper.clone())], d=initial_splits)
            ls += [b[0] for b in multi_bounds]
            us += [b[1] for b in multi_bounds]
            save_mb.append(multi_bounds)

        # for i in range(len(ls)):
        #     print(i+1, ls[i])
        #     print(i+1, us[i])
        #     print()

        ls = torch.stack(ls)
        us = torch.stack(us)
        
        with torch.no_grad():
            (lbs, ubs), _, _ = self.abstractor(ls, us, assignment=None, return_hidden_bounds=True, reset_param=True)

        # for i in range(len(ls)):
        #     print(i+1, lbs[i])
        #     print(i+1, ubs[i])
        #     print()


        best_indices = []
        for prop_mat, prop_rhs in self.spec.mat:
            prop_mat = torch.from_numpy(prop_mat)
            for p in prop_mat:
                # print(p.shape, p[None])
                score = (p[None].clamp(max=0) * lbs + p[None].clamp(min=0) * ubs).sum(dim=-1)
            #     print(score.shape)
            # print(prop_mat.shape, lbs.shape)
            # # print(prop_mat)
            # score = (prop_mat.clamp(max=0) * lbs + prop_mat.clamp(min=0) * ubs).sum(dim=-1)
                best_indices.append(int(score.argmin()) // 2)
        # print(best_indices)
        # exit()
        return save_mb[Counter(best_indices).most_common(1)[0][0]]
        # exit()

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



def worker_func(worker_index, net, shared_queue, shared_status, dataset, spec, abstractor):
    w = Worker(worker_index, net, shared_queue, shared_status, dataset, spec, abstractor)
    w.run()

class SharedStatusMP:

    def __init__(self, n_proc):
        # self.status = {i: True for i in range(n_proc)}
        self.mutex = multiprocessing.Lock()
        self.threads_in_progress = multiprocessing.Array('i', [1]*n_proc)
        # print(list(self.threads_in_progress))
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
        # self.status = {i: True for i in range(n_proc)}
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

        # bounds = spec.get_input_property()
        # self.lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE)
        # self.upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE)

        # print(self.lower)
        # print(self.upper)
        # print()
        self.spec = spec
        self.deeppoly = None #deeppoly.BatchDeepPoly(net, back_sub_steps=10)


    def solve(self):
        print('Solve multi demo')
        N_PROCS = settings.N_PROCS

        shared_queue = multiprocessing.Queue()
        shared_queue.put((self.spec, set([])))

        if N_PROCS > 1:        
            processes = []
            
            shared_status = SharedStatusMP(N_PROCS)
            
            for index in range(N_PROCS):
                p = multiprocessing.Process(target=worker_func, args=(index, self.net, shared_queue, shared_status, self.dataset, self.spec, self.deeppoly))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print('Error:', shared_status.thread_errors.value)

        else:
            # shared_queue = 
            shared_status = SharedStatus(N_PROCS)
            worker_func(0, self.net, shared_queue, shared_status, self.dataset, self.spec, self.deeppoly)

            print('Error:', shared_status.thread_errors)

        # print(shared_queue)


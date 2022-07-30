'''
utilities for lp solving
'''

import time
from multiprocessing import Pool

import numpy as np

from nnenum.timerutil import Timers
from nnenum.settings import Settings

def init_worker(wfunc, serialized_star):
    'initializer parallel worker'

    #assert isinstance(serialized_star.lpi.lp, tuple)
    wfunc.star = serialized_star

def worker_func(param):
    '''worker func for parallel lp

    return tuple is (split_index, new_lb, new_ub, num_lps)
    '''

    star = worker_func.star
    num_lps = 0

    if isinstance(star.lpi.lp, tuple):
        star.lpi.deserialize()

    # lb, ub are current bounds
    i, lb, ub, sim_i, both_bounds = param

    #ub_first = -lb < ub
    ub_first = sim_i < 0

    # recompute bounds
    num_lps += 1
    
    if ub_first:
        # try upper bound first
        ub = star.minimize_output(i, maximize=True)
    else:
        lb = star.minimize_output(i)

    #if lb >= 0 or ub <= 0:
    if lb < -Settings.SPLIT_TOLERANCE and ub > Settings.SPLIT_TOLERANCE:
        if both_bounds or (ub_first and lb == -np.inf) or (not ub_first and ub == np.inf):
            # for enumeration, we only need a single bound since the simulation is the witness for the other side

            # try the other side
            num_lps += 1
            
            if ub_first:
                lb = star.minimize_output(i)
            else:
                ub = star.minimize_output(i, maximize=True)

            assert lb - 1e-1 < sim_i < ub + 1e-1, f"sim[{i}]={sim_i} was not between bounds: {lb, ub}"

    return (i, lb, ub, num_lps)

def update_bounds_lp(layer_bounds, star, sim, split_indices, depth, check_cancel_func=None, both_bounds=False):
    '''update the passed in bounds using an lp solver (if two-sided)
    
    split_indices may be None
    check_cancel_func can raise an expection to cancel

    returns split_indices
    '''

    n = Settings.NUM_LP_PROCESSES

    if Settings.PARALLEL_ROOT_LP:
        p = Settings.NUM_PROCESSES

        if depth >= 30:
            n = 1 # we don't have 1 billion processes, run single threaded
        else:
            num_threads = 2**depth # estimate of number of threads
            n = p // num_threads

    # eth_check.py mnist_0.3 2 fails half the time with multithreadding, this detects it
    # memory corruption?
    #print(f".lputil WARNING: checking feas with {star.lpi.get_num_rows()} rows and {star.lpi.get_num_cols()} cols")
    #if not star.lpi.is_feasible():
    #    print("infeasible")
    #    print(star.lpi)

    if n <= 1:
        rv = update_bounds_lp_serial(layer_bounds, star, sim, split_indices, check_cancel_func, both_bounds)
    else:
        rv = update_bounds_lp_parallel(layer_bounds, star, sim, split_indices, n, check_cancel_func, both_bounds)

    return rv

def update_bounds_lp_parallel(layer_bounds, star, sim, split_indices, num_processes,
                              check_cancel_func=None, both_bounds=False):
    '''
    parallel version of update_bounds_lp
    '''

    start = time.perf_counter()

    Timers.tic('update_bounds_lp_parallel')
    assert len(sim) == layer_bounds.shape[0]

    if split_indices is None:
        num_neurons = star.a_mat.shape[0]
        split_indices = range(num_neurons)

    params = []

    for i in split_indices:
        lb, ub = layer_bounds[i]
        assert lb < 0 < ub, f"bounds for {i} were not two-sided: {lb}, {ub}"

        param = i, lb, ub, sim[i], both_bounds
        params.append(param)

    ####### start
    
    star.lpi.serialize()

    init_arg = (worker_func, star)
    new_splits = []
    total_lps = 0

    with Pool(initializer=init_worker, initargs=init_arg, processes=num_processes) as pool:

        for i, res in enumerate(pool.imap_unordered(worker_func, params)):
            if Settings.NUM_PROCESSES == 1 and Settings.PRINT_OUTPUT:
                print(f'\rParallel LP Progress: {round(100 * i / len(params), 1)}%', end='', flush=True)

            i, lb, ub, num_lps = res
            total_lps += num_lps

            layer_bounds[i, 0] = lb
            layer_bounds[i, 1] = ub

            if i % 2 == 0 and check_cancel_func is not None:
                check_cancel_func() # raises exception to cancel

            if lb < -Settings.SPLIT_TOLERANCE and ub > Settings.SPLIT_TOLERANCE:
                new_splits.append(i)

        pool.close() # do we need this? we're in a manager

    star.lpi.deserialize()
    #############

    new_splits = np.array(new_splits)
    Timers.toc('update_bounds_lp_parallel')

    if Settings.NUM_PROCESSES == 1 and Settings.PRINT_OUTPUT:
        diff = max(1e-6, time.perf_counter() - start)
        avg = diff / total_lps

        print(f"\nTotal time in update_bounds_lp: {round(diff, 2)}, num lp: {total_lps}, lp avg ms: {round(avg, 4)}")

    return new_splits        

def update_bounds_lp_serial(layer_bounds, star, sim, split_indices, check_cancel_func=None, both_bounds=False):
    '''
    single-threaded version of update_bounds_lp
    '''

    #start = time.perf_counter()

    Timers.tic('update_bounds_lp')
    assert len(sim) == layer_bounds.shape[0]
    new_splits = []

    if split_indices is None:
        num_neurons = star.a_mat.shape[0]
        split_indices = range(num_neurons)

    for index_in_list, i in enumerate(split_indices):

        #ratio = index_in_list / len(split_indices)
        #diff = time.perf_counter() - start
        
        #if ratio > 0:
        #    tot = diff / ratio
        #    print(f"solving lp {index_in_list} / {len(split_indices)}, total time estimate: {tot}")
        
        
        if index_in_list % 2 == 0 and check_cancel_func is not None:
            check_cancel_func() # raises exception to cancel

        if not Settings.EAGER_BOUNDS and new_splits:
            new_splits.append(i) # non-eager computation
            continue
            
        lb, ub = layer_bounds[i]

        assert lb < 0 < ub, f"bounds for {i} were not two-sided: {lb}, {ub}"

        #ub_first = -lb < ub
        ub_first = sim[i] < 0

        # recompute bounds
        if ub_first:
            # try upper bound first
            layer_bounds[i, 1] = ub = star.minimize_output(i, maximize=True)
        else:
            layer_bounds[i, 0] = lb = star.minimize_output(i)

        #if lb >= 0 or ub <= 0:
        if lb >= -Settings.SPLIT_TOLERANCE or ub <= Settings.SPLIT_TOLERANCE:
            # branch was rejected, done
            continue

        if both_bounds or (ub_first and lb == -np.inf) or (not ub_first and ub == np.inf):
            # for enumeration, we only need a single bound since the simulation is the witness for the other side

            # try the other side
            if ub_first:
                layer_bounds[i, 0] = lb = star.minimize_output(i)
            else:
                layer_bounds[i, 1] = ub = star.minimize_output(i, maximize=True)

            assert lb - 1e-1 < sim[i] < ub + 1e-1, f"sim[{i}]={sim[i]} was not between bounds: {lb, ub}"

            # branch was rejected, done
            if lb >= -Settings.SPLIT_TOLERANCE or ub <= Settings.SPLIT_TOLERANCE:
                continue

        new_splits.append(i)

    new_splits = np.array(new_splits)

    Timers.toc('update_bounds_lp')

    #print(f"total time in update_bounds_lp: {time.perf_counter() - start}")

    return new_splits

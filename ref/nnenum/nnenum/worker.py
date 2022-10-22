'''
Part of the enumeration process splits work among multiple worker processes. This is their implementation.

Stanley Bak
Feb 2020
'''

import math
import random
import time

import numpy as np

from nnenum.timerutil import Timers
from nnenum.overapprox import do_overapprox_rounds, make_prerelu_sims, OverapproxCanceledException
from nnenum.settings import Settings
from nnenum.util import Freezable, to_time_str
from nnenum.network import nn_unflatten, nn_flatten

from nnenum.prefilter import LpCanceledException

class Worker(Freezable):
    'local data for a worker process'

    def __init__(self, shared, priv):

        self.shared = shared
        self.priv = priv

        self.freeze_attrs()

    def has_timeout(self):
        'was a timeout reached?'

        return time.perf_counter() - self.priv.start_time > Settings.TIMEOUT
        
    def main_loop(self):
        'main worker loop'

        should_exit = False

        while not should_exit:
            # check if finished
            if self.priv.ss and self.priv.ss.is_finished(self.shared.network):
                self.finished_star() # this sets self.priv.ss to None

                if self.priv.work_list and Settings.BRANCH_MODE in [Settings.BRANCH_EGO, Settings.BRANCH_EGO_LIGHT]:
                    self.priv.work_list[-1].should_try_overapprox = True

            timer_name = Timers.stack[-1].name if Timers.stack else None
            
            try: # catch lp timeout
                if self.priv.ss and not self.has_timeout():
                    first_overapprox = len(self.priv.ss.branch_tuples) == 0
                    
                    self.consider_overapprox()

                    if self.priv.worker_index == 0 and first_overapprox:
                        self.shared.finished_initial_overapprox.value = 1

                    # overapproximation intersection with violation can cause splits to be eliminated
                    if self.priv.ss and self.priv.ss.prefilter.output_bounds.branching_neurons.size == 0:
                        self.priv.ss.propagate_up_to_split(self.shared.network, self.priv.start_time)

                if self.priv.ss and not self.has_timeout():
                    start_time = time.perf_counter()
                    self.advance_star()
                    diff = time.perf_counter() - start_time
                    self.add_branch_str(f"advance_star {round(diff * 1000, 3)} ms")

            except LpCanceledException:
                while Timers.stack and Timers.stack[-1].name != timer_name:
                    Timers.toc(Timers.stack[-1].name)

                self.timeout()

            # pop queue before updating shared variables so it doesn't look like there's no work if queue is nonempty
            if self.priv.ss is None:
                
                if self.priv.work_list: # pop from local
                    self.priv.ss = self.priv.work_list.pop()
                    print('[+] Pop:', self.priv.ss)
                    
                else: # pop from global (shared)

                    if self.priv.worker_index == 0 or self.shared.finished_initial_overapprox.value:
                        self.priv.ss = self.shared.get_global_queue(timeout=0.01)

                    if self.priv.ss is not None:
                        print('[+] Pop:', self.priv.ss)
                        
                        # make sure we tell other people we have work now
                        self.priv.shared_update_urgent = True

                        if self.priv.fulfillment_requested_time is not None:
                            diff = time.perf_counter() - self.priv.fulfillment_requested_time
                            self.priv.fulfillment_requested_time = None
                            self.priv.total_fulfillment_time += diff
                            self.priv.total_fulfillment_count += 1

            # shuffle (optional)
            if Settings.SHUFFLE_TIME is not None and time.perf_counter() > self.priv.next_shuffle_time:
                self.shuffle_work() # todo: evaluate if this helps

            should_exit = self.update_shared_variables()
            self.print_progress()

        Timers.tic('post_loop')
        self.update_final_stats()
        self.clear_remaining_work()
        Timers.toc('post_loop')

    def add_branch_str(self, label):
        '''add the branch string to the tuples list (if we're saving it)'''

        if self.priv.branch_tuples_list is not None:
            self.priv.branch_tuples_list.append(f'{self.priv.ss.branch_str()} ({label})')

    def consider_overapprox(self):
        '''conditionally run overapprox analysis

        this may set self.priv.ss to None if overapprox is safe

        returns is_safe (False does not mean unsafe, just that safe cannot be proven or timeout)
        '''

        do_overapprox = False
        is_safe = False
        concrete_io_tuple = None
        ss = self.priv.ss
        network = self.shared.network
        spec = self.shared.spec

        assert ss.remaining_splits() > 0

        print(' ----- consider approx -----')

        if Settings.BRANCH_MODE == Settings.BRANCH_OVERAPPROX:
            do_overapprox = True
        elif Settings.BRANCH_MODE in [Settings.BRANCH_EGO, Settings.BRANCH_EGO_LIGHT]:
            do_overapprox = ss.should_try_overapprox

        # todo: experiment moving this after single-zono overapprox
        if do_overapprox and Settings.SPLIT_IF_IDLE and self.exists_idle_worker():
            do_overapprox = False

        if do_overapprox:
            # todo: experiment global timeout vs per-round timeout
            start = time.perf_counter()

            def check_cancel_func():
                'worker cancel func. can raise OverapproxCanceledException'

                if self.shared.should_exit.value:
                    raise OverapproxCanceledException(f'shared.should_exit was true')

                #if Settings.SPLIT_IF_IDLE and self.exists_idle_worker():
                #    print("cancel idle")
                #    raise OverapproxCanceledException('exists idle worker')

                now = time.perf_counter()

                if now - self.priv.start_time > Settings.TIMEOUT:
                    raise OverapproxCanceledException('timeout exceeded')

                if now - start > Settings.OVERAPPROX_LP_TIMEOUT:
                    raise OverapproxCanceledException('lp timeout exceeded')

            timer_name = 'do_overapprox_rounds'
            Timers.tic(timer_name)

            # compute simulation first (and make sure it's safe)
            prerelu_sims = make_prerelu_sims(ss, network)
                        
            if prerelu_sims is None:
                concrete_io_tuple = None
            else:
                sim_out = prerelu_sims[len(network.layers)]

                if spec.is_violation(sim_out):
                    sim_in_flat = ss.prefilter.simulation[0]
                    sim_in = ss.star.to_full_input(sim_in_flat)

                    # run through complete network in to out before counting it
                    sim_out = network.execute(sim_in)
                    sim_out = nn_flatten(sim_out)

                    if spec.is_violation(sim_out):
                        concrete_io_tuple = [sim_in, sim_out]

                        if Settings.PRINT_OUTPUT:
                            print("\nPre-overapproximation simulation found a confirmed counterexample.")
                            print(f"\nUnsafe Base Branch: {self.priv.ss.branch_str()} (Mode: {Settings.BRANCH_MODE})")

                        self.found_unsafe(concrete_io_tuple)
                        self.add_branch_str('CONCRETE UNSAFE')

            if concrete_io_tuple is None:
                # sim was safe, proceed with overapproximation

                try:
                    gen_limit = max(self.priv.max_approx_gen, Settings.OVERAPPROX_MIN_GEN_LIMIT)

                    if Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER is None:
                        gen_limit = np.inf

                    num_branches = len(ss.branch_tuples)
                    if num_branches > Settings.OVERAPPROX_NEAR_ROOT_MAX_SPLITS:
                        otypes = Settings.OVERAPPROX_TYPES
                    else:
                        otypes = Settings.OVERAPPROX_TYPES_NEAR_ROOT

                    print('[+] Abstraction:', ss)
                    res = do_overapprox_rounds(ss, network, spec, prerelu_sims, check_cancel_func, gen_limit,
                                               overapprox_types=otypes)
                    print('\t- safe:', ss, res.is_safe)
                    if res.is_safe:
                        print(ss.star.lpi)

                    if res.concrete_io_tuple is not None:
                        if Settings.PRINT_OUTPUT:
                            print("\nviolation star found a confirmed counterexample.")
                            print(f"\nUnsafe Base Branch: {self.priv.ss.branch_str()} (Mode: {Settings.BRANCH_MODE})")

                        self.found_unsafe(res.concrete_io_tuple)
                        self.add_branch_str('CONCRETE UNSAFE')
                    else:
                        is_safe = res.is_safe
                        safe_str = "safe" if is_safe else "unsafe"
                        self.add_branch_str(f"{safe_str} {res}")

                        if not is_safe:
                            ss.should_try_overapprox = False

                        if Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER is not None:
                            if is_safe:
                                new_max = Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER * res.get_max_gens()
                                self.priv.max_approx_gen = max(self.priv.max_approx_gen, new_max)
                            else:
                                self.priv.max_approx_gen = 0 # reset limit

                except OverapproxCanceledException as e:
                    # fix timers
                    while Timers.stack and Timers.stack[-1].name != timer_name:
                        Timers.toc(Timers.stack[-1].name)

                    print('\t- stat: cancelled')

                    self.add_branch_str(f'{str(e)}')
                    self.priv.max_approx_gen = 0 # reset limit

                    ss.should_try_overapprox = False

            Timers.toc(timer_name)

            ##### post overapproximation processing
            
            if Settings.PRINT_BRANCH_TUPLES:
                print(self.priv.branch_tuples_list[-1])

            if is_safe or concrete_io_tuple is not None:
                # done with this branch
                 
                if Settings.RESULT_SAVE_POLYS:
                    self.save_poly(ss)

                self.priv.ss = None
                self.priv.finished_approx_stars += 1

                # local stats that get updated in update_shared_variables
                self.priv.update_stars += 1
                self.priv.update_work_frac += ss.work_frac
                self.priv.update_stars_in_progress -= 1
                    
                if not self.priv.work_list:
                    # urgently update shared variables to try to get more work
                    self.priv.shared_update_urgent = True
                    self.priv.fulfillment_requested_time = time.perf_counter()

        return is_safe
        
    def shuffle_work(self):
        'shuffle work'

        Timers.tic('shuffle')
        
        if self.priv.worker_index == 0:
            # print queues
            qsize = self.shared.more_work_queue.qsize()

        #self.priv.next_shuffle_step *= 2 # exponential backoff
        self.priv.next_shuffle_time += self.priv.next_shuffle_step

        global_work = []

        while True:
            ss = self.shared.get_global_queue(timeout=0.01)

            if ss is None:
                break

            global_work.append(ss)

        if self.priv.ss:
            i = self.priv.worker_index
            my = len(self.priv.work_list)
            print(f".{i}: my work size {my}", flush=True)

            self.shared.put_queue(self.priv.ss)
            self.priv.num_offloaded += 1

            #self.priv.work_list.append(self.priv.ss)
            self.priv.ss = None

        self.priv.work_list += global_work

        # shuffle remaining work and put it all into the queue
        random.shuffle(self.priv.work_list)

        #for ss in self.priv.work_list:
        #    self.shared.put_queue(ss)
        #    self.priv.num_offloaded += 1

        #self.priv.work_list = []
        #self.priv.shared_update_urgent = True
        #self.priv.fulfillment_requested_time = time.perf_counter()

        Timers.toc('shuffle')

    def exists_idle_worker(self):
        'do idle workers (with no work) exist?'

        Timers.tic('exists_idle_worker')

        rv = False

        # checking qsize here slows things down

        for i, size in enumerate(self.shared.heap_sizes):
            if i != self.priv.worker_index:
                if size == 0:
                    rv = True
                    break

        Timers.toc('exists_idle_worker')

        return rv

    def update_shared_variables(self):
        '''update shared variables among workers periodically

        returns should_exit
        '''

        should_exit = False

        if self.shared.should_exit.value:
            self.priv.shared_update_urgent = True

        # don't need to update shared varaibles every iteration as it can take nontrivial time
        now = time.perf_counter()

        if self.priv.shared_update_urgent or (self.priv.work_list and now > self.priv.next_shared_var_update):
            Timers.tic('update_shared_variables')
            
            windex = self.priv.worker_index

            #######################################
            Timers.tic('get mutex')
            should_block = self.priv.shared_update_urgent
            did_lock = self.shared.mutex.acquire(should_block)
            Timers.toc('get mutex')

            if did_lock:
                self.priv.shared_update_urgent = False

            self.priv.next_shared_var_update = now + Settings.UPDATE_SHARED_VARS_INTERVAL

            # can be non-blocking
            if did_lock:
                # update in progress counts
                self.shared.stars_in_progress.value += self.priv.stars_in_progress
                self.priv.stars_in_progress = 0

                # update heap info
                self.shared.heap_sizes[windex] = len(self.priv.work_list)

                if self.priv.ss:
                    self.shared.heap_sizes[windex] += 1
                    
                num_zeros = 0

                for i, size in enumerate(self.shared.heap_sizes):
                    if i != self.priv.worker_index:
                        if size == 0:
                            num_zeros += 1
                            self.shared.heap_sizes[i] = 0 # temp marker so other people don't give them work
                    
                #heap_sizes = list(self.shared.heap_sizes) # was comprehension

                # update finished counts (these get set in finalized_star)
                self.shared.finished_stars.value += self.priv.update_stars
                self.shared.finished_work_frac.value += self.priv.update_work_frac
                self.shared.stars_in_progress.value += self.priv.update_stars_in_progress

                self.shared.incorrect_overapprox_count.value += self.priv.incorrect_overapprox_count
                self.shared.incorrect_overapprox_time.value += self.priv.incorrect_overapprox_time

                self.priv.update_stars = 0
                self.priv.update_work_frac = 0
                self.priv.update_stars_in_progress = 0
                self.priv.incorrect_overapprox_count = 0
                self.priv.incorrect_overapprox_time = 0

                # check if completed
                if self.is_finished_with_lock():
                    should_exit = True

                self.shared.mutex.release()
                #########################################

                if not should_exit:
                    Timers.tic('load_balancing')
                    self.do_load_balancing(num_zeros)
                    Timers.toc('load_balancing')

            Timers.toc('update_shared_variables')

        return should_exit

    def timeout(self):
        '''a timeout occured'''

        ##############################
        self.shared.mutex.acquire()
        self.shared.had_timeout.value = 1
        self.shared.mutex.release()
        ##############################
        self.priv.shared_update_urgent = True

    def print_progress(self):
        'periodically print progress (worker 0 only)'

        if self.priv.worker_index == 0:
            now = time.perf_counter()

            if self.priv.last_print_time is None:
                self.priv.last_print_time = now - Settings.PRINT_INTERVAL - 1 # force a print at start

            cur_time = now - self.priv.start_time

            if cur_time >= Settings.TIMEOUT:
                self.timeout()

            if Settings.PRINT_OUTPUT and Settings.PRINT_PROGRESS and \
               now - self.priv.last_print_time > Settings.PRINT_INTERVAL:
                Timers.tic("print_progress")
                
                # print stats
                self.priv.last_print_time = now
                self.priv.num_prints += 1

                ##############################
                self.shared.mutex.acquire()
                
                in_progress = self.shared.stars_in_progress.value
                finished = self.shared.finished_stars.value

                layers = list(self.shared.cur_layers)
                neurons = list(self.shared.cur_neurons)

                qsize = self.shared.more_work_queue.qsize()

                finished_frac = self.shared.finished_work_frac.value

                self.shared.mutex.release()
                ##############################
                elapsed = now - self.priv.start_time
                time_str = to_time_str(elapsed)

                status = ""

                for layer, neuron in zip(layers, neurons):
                    if status:
                        status += " "

                    status += f"{layer}-{neuron}".rjust(7)

                status += "       "
                total_stars = in_progress + finished

                eta = '-'

                delta_time = now - self.priv.start_time

                if finished_frac > 0:
                    estimated_total = delta_time / finished_frac
                    eta = max(estimated_total - elapsed, 0)
                    eta = to_time_str(eta)

                # don't divide by 0
                expected_stars = round(1 if finished_frac < 1e-9 else finished / finished_frac)

                print(f"({time_str}) Q: {qsize}, Sets: {finished}/{total_stars} " + \
                      f" ({round(finished_frac * 100, 3)}%) ETA: {eta} (expected {expected_stars} stars)   ", end="\r")

                log_prints = math.log(self.priv.num_prints, 2)
                
                if abs(log_prints - round(log_prints)) < 1e-7:
                    # save snapshot
                    print("") # newline
                
                Timers.toc("print_progress")

    def is_finished_with_lock(self):
        '''is the computation finished (should workers exit?)

        call with mutex locked
        '''

        rv = self.shared.stars_in_progress.value == 0

        if not rv:
            rv = self.shared.result.found_confirmed_counterexample.value == 1

        if not rv and self.shared.had_exception.value == 1:

            if Settings.PRINT_OUTPUT:
                print(f"Worker {self.priv.worker_index} quitting due to exception in some worker")
                
            self.priv.had_exception = 1
            rv = True

        if not rv and self.shared.had_timeout.value == 1:
            rv = True

        if rv and not self.shared.should_exit.value:
            self.shared.should_exit.value = 1

        return rv
                        
    def clear_remaining_work(self):
        'sometimes we quit early, make sure the work queue is empty so processes exit as expected'

        # force an update
        self.priv.shared_update_urgent = True
        self.update_shared_variables()

        if self.priv.ss is not None or self.priv.stars_in_progress > 0:
            #######################################
            self.shared.mutex.acquire()

            if self.priv.ss:
                self.shared.stars_in_progress.value -= 1
                self.shared.unfinished_stars.value += 1
                self.priv.ss = None

            self.shared.stars_in_progress.value += self.priv.stars_in_progress
            self.priv.stars_in_progress = 0
                
            self.shared.mutex.release()
            ########################################

        should_exit = False
        count = len(self.priv.work_list)

        while not should_exit:
            
            # pop as many as possible
            while True:
                ss = self.shared.get_global_queue(False, skip_deserialize=True)

                if ss:
                    count += 1
                else:
                    break

            ########################################
            self.shared.mutex.acquire()
            self.shared.stars_in_progress.value -= count
            self.shared.unfinished_stars.value += count
            count = 0

            if self.shared.stars_in_progress.value <= 0:
                assert self.shared.stars_in_progress.value == 0, \
                    f"stars in progress should be 0, was {self.shared.stars_in_progress.value}"
                should_exit = True

            self.shared.mutex.release()
            ##########################################

            # don't busy wait
            if not should_exit:
                time.sleep(0.01)

    def update_final_stats(self):
        'all processes finished, update global stats'

        self.shared.mutex.acquire()

        self.shared.finished_approx_stars.value += self.priv.finished_approx_stars

        self.shared.num_lps.value += self.priv.num_lps
        self.shared.num_lps_enum.value += self.priv.num_lps_enum
        
        self.shared.num_offloaded.value += self.priv.num_offloaded

        for tindex, timer_name in enumerate(Settings.RESULT_SAVE_TIMERS):
            timer_list = Timers.top_level_timer.get_children_recursive(timer_name)
            
            for t in timer_list:
                self.shared.timer_secs[tindex] += t.total_secs
                self.shared.timer_counts[tindex] += t.num_calls

        self.shared.mutex.release()

    def do_load_balancing(self, num_zeros):
        '''balance work across threads using a work-stealing strategy
        '''

        if self.priv.work_list:
            #print(f". num_zeros: {num_zeros}")
            if num_zeros > 0:
                while num_zeros > 0 and len(self.priv.work_list) > 1 or \
                  (len(self.priv.work_list) > 0 and self.priv.ss is not None):
                    # push some work onto the queue

                    # min item is heaviest (closest to root)
                    # heaviest will be first item on list
                    new_ss = self.priv.work_list.pop(0)

                    self.priv.num_offloaded += 1
                    num_zeros -= 1

                    self.shared.put_queue(new_ss)

    def save_star(self, ss):
        '''save the lp_star to the result

        this has the effect of serializing the current star's lpi if multithreaded
        '''

        ss.star.lpi.serialize()
        self.shared.result.stars.append(ss.star)

    def save_poly(self, ss):
        'save the polygon verts for the current, finished star into result.polys'

        Timers.tic('save_poly')

        xdim, ydim = Settings.RESULT_SAVE_POLYS_DIMS

        # save polygon
        verts = ss.star.verts(xdim, ydim, epsilon=Settings.RESULT_SAVE_POLYS_EPSILON)
        self.shared.result.polys.append(verts)
        
        Timers.toc('save_poly')

    def find_concrete_io(self, star, branch_tuples):
        'try to find a concrete input and output in star, that explores the passed-in branch_tuples'

        assert Settings.FIND_CONCRETE_COUNTEREXAMPLES

        Timers.tic('find_concrete_io')
        rv = None

        # solve lp to get the input/output
        res = star.minimize_vec(None, return_io=True)

        if res is not None:
            cinput, _ = res

            # try to confirm the counter-example
            full_cinput_flat = star.to_full_input(cinput).astype(star.a_mat.dtype)
            
            exec_output, exec_branch_list = self.shared.network.execute(full_cinput_flat, save_branching=True)
            exec_output = nn_flatten(exec_output)

            if branch_list_in_branch_tuples(exec_branch_list, branch_tuples):
                rv = full_cinput_flat, exec_output
            else:
                #print(". weakly-tested code: couldn't confirm countereample... tightening constraints")

                # try to make each of the constraints a little tighter
                star_copy = star.copy()

                rhs_original = star_copy.lpi.get_rhs()
                rhs = rhs_original.copy()

                # tighten by this factor
                tighten_factor = 1e-16

                while tighten_factor < 1e16:
                    
                    # tighten the constraints a little
                    for i, val in enumerate(rhs_original):
                        rhs[i] = val - tighten_factor

                    star_copy.lpi.set_rhs(rhs)
                    
                    res = star_copy.minimize_vec(None, return_io=True, fail_on_unsat=False)

                    if res is None:
                        # infeasible
                        break

                    cinput, _ = res

                    full_cinput_flat = star.to_full_input(cinput)
                    full_cinput = nn_unflatten(full_cinput_flat, self.shared.network.get_input_shape())
                    exec_output, exec_branch_list = self.shared.network.execute(full_cinput, save_branching=True)
                    exec_output = nn_flatten(exec_output)

                    if branch_list_in_branch_tuples(exec_branch_list, branch_tuples):
                        rv = full_cinput_flat, exec_output
                        break

                    # for next loop, tighten even more
                    tighten_factor *= 10

        Timers.toc('find_concrete_io')
                        
        return rv

    def found_unsafe(self, concrete_io_tuple):
        '''found a concrete counter-example, update shared variables.

        concrete_io_tuple may be None, in the case of unconfirmed counterexamples
        '''

        if self.shared.result.found_confirmed_counterexample.value == 0:
            #########################
            Timers.tic('update_shared')
            self.shared.mutex.acquire()

            self.shared.result.found_counterexample.value = 1

            if concrete_io_tuple is not None:
                self.shared.result.found_confirmed_counterexample.value = 1
                self.shared.should_exit.value = True

                for i, val in enumerate(concrete_io_tuple[0]):
                    self.shared.result.cinput[i] = val

                for i, val in enumerate(concrete_io_tuple[1]):
                    self.shared.result.coutput[i] = val

            self.shared.mutex.release()
            Timers.toc('update_shared')
            #########################
    
    def finished_star(self):
        'finished with a concrete star state'

        Timers.tic('finished_star')

        ss = self.priv.ss
        assert ss is not None
        concrete_io_tuple = None
        violation_star = None

        # update enumeration num lps before checking
        self.priv.num_lps_enum += ss.star.num_lps

        if Settings.RESULT_SAVE_POLYS:
            self.save_poly(self.priv.ss)

        spec = self.shared.spec

        if spec is not None and spec.zono_might_violate_spec(ss.prefilter.zono):
            violation_star = self.shared.spec.get_violation_star(ss.star, safe_spec_list=ss.safe_spec_list)

            # check if it's a confirmed counter-example
            # it's okay to do this unconditionally, as when a confirmed counter-example is found, we will quit
            if violation_star is not None and Settings.FIND_CONCRETE_COUNTEREXAMPLES:
                concrete_io_tuple = self.find_concrete_io(violation_star, ss.branch_tuples)

                if Settings.PRINT_OUTPUT:
                    if concrete_io_tuple:
                        print(f"\nWorker {self.priv.worker_index} found a confirmed counterexample (concrete)")
                        print(f"\nUnsafe Branch: {self.priv.ss.branch_str()} (Branch Mode: {Settings.BRANCH_MODE})")
                    else:
                        print(f"\nWorker {self.priv.worker_index} found a counterexample, but it wasn't " + \
                               "reproduced by a concrete execution.")
        # accumulate stats
        self.priv.finished_stars += 1
        self.priv.num_lps += ss.star.num_lps

        # branch_str
        self.add_branch_str(f"concrete {'UNSAFE' if violation_star is not None else 'safe'}")

        # do this last as it will serialize the star's lpi if multithreaded
        if Settings.RESULT_SAVE_STARS:
            self.save_star(self.priv.ss)

        self.priv.ss = None

        # local stats that get updates in update_shared_variables
        self.priv.update_stars += 1
        self.priv.update_work_frac += ss.work_frac
        self.priv.update_stars_in_progress -= 1

        if not self.priv.work_list:
            # urgently update shared variables to try to get more work
            self.priv.shared_update_urgent = True
            self.priv.fulfillment_requested_time = time.perf_counter()

        if Settings.PRINT_BRANCH_TUPLES:
            print(self.priv.branch_tuples_list[-1])

        if violation_star is not None:
            self.found_unsafe(concrete_io_tuple)

        Timers.toc('finished_star')

    def advance_star(self):
        '''advance current star (self.priv.ss)

        A precondition to this is that ss is already at the next split point.

        The logic for this is:

        1. do split, creating new_star
        2. propagate up to next split with ss
        3. propagate up to next split with new_star
        4. save new_star to remaining work
        '''

        Timers.tic('advance')

        ss = self.priv.ss
        network = self.shared.network
        spec = self.shared.spec

        if not ss.is_finished(network):
            new_star = ss.do_first_relu_split(network, spec, self.priv.start_time)

            ss.propagate_up_to_split(network, self.priv.start_time)

            if new_star: # new_star can be null if it wasn't really a split (copy prefilter)
                new_star.propagate_up_to_split(network, self.priv.start_time)

                # note: new_star may be done... but for expected branching order we still add it
                self.priv.stars_in_progress += 1
                print('[+] Append:', new_star)
                self.priv.work_list.append(new_star)

        Timers.toc('advance')

def branch_list_in_branch_tuples(branch_list, branch_tuples):
    'does the passed in concrete-execution (branch_list) go down the same branches as the star (branch_tuples)?'

    rv = True

    for layer, neuron, branch_type in branch_tuples:

        #print(f"layer {layer}, neuron {neuron}, star branch type {branch_type}")

        exec_type = branch_list[layer][neuron]

        #print(f"exec_type: {exec_type}")

        if isinstance(exec_type, list):
            rv = branch_type in exec_type
        else:
            rv = branch_type == exec_type

        if not rv:
            break

    return rv

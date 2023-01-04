from input_split.solver import Solver as InputSplitSolver
from util.spec.spec_vnnlib import SpecVNNLIB
from core.solver.smt_solver import SMTSolver
from util.misc.logger import logger
from attack.attack import Attacker
import arguments

import time

class NeuralSAT:

    def __init__(self, net, specs):
        self.net = net
        self.raw_specs = specs

        # counter-example
        self._assignment = None 

        # adv attack
        self.attacker = Attacker(net, specs)

        # create multiple specs from DNF spec
        self.raw_specs = self._preprocess_spec(self.raw_specs)


    def _preprocess_spec(self, raw_specs):
        new_raw_specs = []
        for spec in raw_specs:
            bounds = spec[0]
            for i in spec[1]:
                new_raw_specs.append((bounds, [i]))
        return new_raw_specs

    def solve(self, timeout=1000):
        # pass
        start_time = time.perf_counter()
        stat = arguments.ReturnStatus.UNKNOWN

        if arguments.Config['attack']:
            if self.check_adv_pre():
                return arguments.ReturnStatus.SAT

        for idx, spec in enumerate(self.raw_specs):
            spec_start_time = time.perf_counter()
            vnnlib_spec = SpecVNNLIB(spec)

            logger.info(f'Spec {idx+1}/{len(self.raw_specs)} ({vnnlib_spec.mat[0][0].tolist()})')

            # try input splitting
            input_split_solver = InputSplitSolver(self.net, vnnlib_spec)
            stat= input_split_solver.solve()
            if stat == arguments.ReturnStatus.SAT: 
                self._assignment = input_split_solver.get_assignment()
                return stat
            
            if stat == arguments.ReturnStatus.UNSAT:
                logger.info(f'Spec {idx+1}/{len(self.raw_specs)} stat={stat} time={time.perf_counter() - spec_start_time:.02f} remain={timeout - (time.perf_counter() - start_time):.02f}')
                continue
            

            # try hidden splitting
            smt_solver = SMTSolver(self.net, vnnlib_spec)
            remain_time = timeout - (time.perf_counter() - start_time)
            if remain_time < 0:
                return arguments.ReturnStatus.TIMEOUT

            stat = smt_solver.solve(timeout=remain_time)

            logger.info(f'Spec {idx+1}/{len(self.raw_specs)} stat={stat} time={time.perf_counter() - spec_start_time:.02f} remain={timeout - (time.perf_counter() - start_time):.02f}')

            if stat in [arguments.ReturnStatus.SAT, arguments.ReturnStatus.UNKNOWN]:
                return stat
        return stat


    def get_assignment(self):
        return self._assignment


    def check_adv_pre(self):
        is_attacked, self._assignment = self.attacker.run()
        return is_attacked
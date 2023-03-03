from core.input_solver.input_solver import InputSolver
from util.spec.spec_vnnlib import SpecVNNLIB
from core.solver.smt_solver import SMTSolver
from util.misc.logger import logger
from attack.attack import Attacker, ShrinkAttacker
import arguments

import util.network.read_onnx
import time
import pdb
DBG = pdb.set_trace

from beartype import beartype

class NeuralSAT:
    
    @beartype
    def __init__(self, net, raw_specs: list) -> None:
        self.net = net
        self.raw_specs = raw_specs
        # create multiple specs from DNF spec
        self.processed_specs = self._preprocess_spec(raw_specs)

        # adv attack
        self.attacker = Attacker(net, raw_specs)

        # counter-example
        self._assignment = None 

    def _preprocess_spec(self, raw_specs: list) -> list:
        processed_specs = []
        for spec in raw_specs:
            bounds = spec[0]
            for i in spec[1]:
                processed_specs.append((bounds, [i]))
        return processed_specs

    def solve(self, timeout: float = 1000) -> arguments.ReturnStatus:
        start_time = time.perf_counter()
        return_status = []

        # pre-verifying attack
        if arguments.Config['attack']:
            if self.check_adv_pre():
                return arguments.ReturnStatus.SAT

        for idx, spec in enumerate(self.processed_specs):
            spec_start_time = time.perf_counter()
            vnnlib_spec = SpecVNNLIB(spec)

            logger.info(f'Spec {idx+1}/{len(self.processed_specs)} ({vnnlib_spec.mat[0][0].tolist()})')

            # check timeout
            remain_time = timeout - (time.perf_counter() - start_time)
            if remain_time < 0:
                return arguments.ReturnStatus.TIMEOUT

            # try input splitting
            if arguments.Config['input_split']:
                input_split_solver = InputSolver(self.net, vnnlib_spec)
                stat = input_split_solver.solve()

                logger.info(f'Spec {idx+1}/{len(self.processed_specs)} '
                            f'stat={stat} time={time.perf_counter() - spec_start_time:.02f} '
                            f'remain={timeout - (time.perf_counter() - start_time):.02f}')

                if stat in [arguments.ReturnStatus.SAT, arguments.ReturnStatus.TIMEOUT]: 
                    self._assignment = input_split_solver.get_assignment()
                    return stat

                if stat == arguments.ReturnStatus.UNSAT:
                    return_status.append(stat)
                    continue

                if stat == arguments.ReturnStatus.UNKNOWN:
                    # do something if unknown, e.g., skip input splitting, do hidden splitting
                    pass

            # check timeout
            remain_time = timeout - (time.perf_counter() - start_time)
            if remain_time < 0:
                return arguments.ReturnStatus.TIMEOUT

            # try hidden splitting
            smt_solver = SMTSolver(self.net, vnnlib_spec)
            stat = smt_solver.solve(timeout=remain_time)

            logger.info(f'Spec {idx+1}/{len(self.processed_specs)} '  
                        f'stat={stat} time={time.perf_counter() - spec_start_time:.02f} '
                        f'remain={timeout - (time.perf_counter() - start_time):.02f}')

            if stat in [arguments.ReturnStatus.SAT, arguments.ReturnStatus.TIMEOUT]:
                self._assignment = smt_solver.get_assignment()
                return stat

            return_status.append(stat)

            # post-verifying attack
            if return_status[-1] == arguments.ReturnStatus.UNKNOWN:
                logger.info(f'Spec {idx+1}/{len(self.processed_specs)} '
                            f'Post-verifying attack '
                            f'remain={timeout - (time.perf_counter() - start_time):.02f}')
                if self.shrink_attack([spec]):
                    return arguments.ReturnStatus.SAT

        if arguments.ReturnStatus.UNKNOWN in return_status:
            return arguments.ReturnStatus.UNKNOWN
        
        return arguments.ReturnStatus.UNSAT

    def get_assignment(self) -> dict:
        return self._assignment

    def check_adv_pre(self) -> bool:
        is_attacked, self._assignment = self.attacker.run()
        logger.info(f"Pre-verifying attack {'successfully' if is_attacked else 'failed'}")
        return is_attacked

    def shrink_attack(self, spec, timeout=None) -> bool:
        shrink_attacker = ShrinkAttacker(self.net, spec)
        is_attacked, self._assignment = shrink_attacker.run(timeout)
        logger.info(f"Post-verifying attack {'successfully' if is_attacked else 'failed'}")
        return is_attacked

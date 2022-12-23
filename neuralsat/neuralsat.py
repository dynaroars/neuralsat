from util.spec.spec_vnnlib import SpecVNNLIB
from core.solver.smt_solver import SMTSolver
from attack.attack import Attacker
from util.misc.logger import logger
import arguments

import time

class NeuralSAT:

    def __init__(self, net, specs):
        self.net = net
        self.raw_specs = specs

        # adv attack
        self.attacker = Attacker(net, specs)

        # create multiple specs from DNF spec
        if True: #TODO: fixme: not always True
            self.raw_specs = self._preprocess_spec(self.raw_specs)


    def _preprocess_spec(self, raw_specs):
        new_raw_specs = []        
        bounds = raw_specs[0][0]
        for i in raw_specs[0][1]:
            new_raw_specs.append((bounds, [i]))
        return new_raw_specs

    def solve(self, timeout=1000):
        # pass
        start_time = time.perf_counter()
        stat = arguments.ReturnStatus.UNKNOWN

        if self.check_adv_pre():
            return arguments.ReturnStatus.SAT

        # exit()

        for idx, spec in enumerate(self.raw_specs):
            vnnlib_spec = SpecVNNLIB(spec)
            smt_solver = SMTSolver(self.net, vnnlib_spec)
            remain_time = timeout - (time.perf_counter() - start_time)
            if remain_time < 0:
                return arguments.ReturnStatus.TIMEOUT

            stat = smt_solver.solve(timeout=remain_time)

            msg = f'Spec {idx} ({vnnlib_spec.mat[0][0][0].tolist()}) stat={stat}'
            logger.info(msg)
            if stat in [arguments.ReturnStatus.SAT, arguments.ReturnStatus.UNKNOWN]:
                return stat
        return stat


    def get_assignment(self):
        return self._assignment


    def check_adv_pre(self):
        is_attacked, self._assignment = self.attacker.run()
        return is_attacked
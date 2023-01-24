from attack.pgd_attack import PGDAttack, PGDWhiteBox
from attack.random_attack import RandomAttack
from util.spec.spec_vnnlib import SpecVNNLIB
from util.misc.logger import logger

import random

class Attacker:

    def __init__(self, net, raw_specs):
        self.net = net
        self.raw_specs = raw_specs
        
        self.attackers = []
        self.attackers += [PGDAttack(net, s, mode=m) for s in raw_specs for m in ['diversed_PGD', 'diversed_GAMA_PGD', 'PGD']]
        self.attackers += [PGDWhiteBox(net, SpecVNNLIB(s)) for s in raw_specs]
        self.attackers += [RandomAttack(net, SpecVNNLIB(s)) for s in raw_specs]
 

    def run(self):
        return self._attack()


    def _attack(self):
        for atk in self.attackers:
            seed = random.randint(0, 1000)
            atk.manual_seed(seed)
            # print('Running attacker:', atk)
            is_attacked, adv = atk.run()

            msg = f'{str(atk):<50} success={is_attacked}'
            logger.info(msg)

            if is_attacked:
                return is_attacked, adv
        return False, None



class ShrinkAttacker:

    def __init__(self, net, raw_specs):

        self.attackers = [PGDAttack(net, s, mode=m) for s in raw_specs for m in ['diversed_PGD', 'diversed_GAMA_PGD', 'PGD']]


    def run(self, timeout):
        print('Shrink Attacker', timeout)
        for i in range(100):
            print('[+] Iteration:', i)
            for atk in self.attackers:
                seed = random.randint(0, 1000)
                atk.manual_seed(seed)
                is_attacked, adv = atk.run(mutate=True)

                if is_attacked:
                    return is_attacked, adv
        return False, None
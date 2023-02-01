import logging
import torch


class ReturnStatus:

    UNSAT   = 'UNSAT'
    SAT     = 'SAT'
    UNKNOWN = 'UNKNOWN'
    TIMEOUT = 'TIMEOUT'
    RESTART = 'RESTART'


class Configuration:

    def __init__(self):
        
        self.seed = 0
        self.dtype = torch.float32
        self.device = 'cpu'
        self.batch = 1

        self.max_hidden_branch = 100000 # if exceed, return unknown
        self.max_input_branch  = 100000 # if exceed, return unknown

        self.early_stop = True # stop when all branches are verified (for hidden splitting)
        self.print_progress = True  # print remaining unverified branches
        self.attack = False
        self.pre_verify_mip_refine = False

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


# global configuration
Config = Configuration()

if __name__ == '__main__':
    print(Config['dtype'])
    
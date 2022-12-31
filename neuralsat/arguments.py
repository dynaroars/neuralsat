import logging
import torch

class ReturnStatus:

    UNSAT   = 'UNSAT'
    SAT     = 'SAT'
    UNKNOWN = 'UNKNOWN'
    TIMEOUT = 'TIMEOUT'



class Configuration:

    def __init__(self):
        
        self.seed = 0
        self.dtype = torch.float32
        self.device = 'cpu'
        
        self.batch = 1024
        self.attack = False
        self.logging_level = logging.INFO

        self.max_branch = 50000 # if exceed, return unknown

        self.early_stop = True # stop when all branches are verified
        self.print_progress = True # print remaining domains

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

# global configuration
Config = Configuration()

if __name__ == '__main__':
    print(Config['dtype'])
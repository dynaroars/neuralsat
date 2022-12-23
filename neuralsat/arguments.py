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
        
        self.batch = 500

        self.max_branch = 50000 # if exceed, return unknown

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

# global configuration
Config = Configuration()

if __name__ == '__main__':
    print(Config['dtype'])
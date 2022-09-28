import random
import torch

DEBUG = False

DTYPE = torch.float32

DECISION = 'MIN_BOUND' # 'RANDOM/MAX_BOUND/MIN_BOUND/KW/GRAD/BABSR'

SEED = random.randint(0, 10000) if DECISION == 'RANDOM' else None
print('SEED:', SEED)

HEURISTIC_DEEPZONO = False
HEURISTIC_DEEPPOLY = True

FALSIFICATION_TIMEOUT = 0.01

BENCHMARKS = ['acasxu', 'cifar2020', 'mnistfc']#, 'oval21', 'nn4sys', 'eran', 'marabou-cifar10']

"""
Scripts used for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import time
import random
import sys

import numpy as np
import torch
from shutil import copyfile
import gurobipy as grb
from data_loader.onnx_parser import ONNXParser

from src.algorithm.verinet import VeriNet
from src.algorithm.verinet_util import Status
from src.algorithm.verification_objectives import ArbitraryObjective
from src.util.logger import get_logger
from src.util.config import *
from src import vnnlib

random_seed = 0
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

benchmark_logger = get_logger(
    LOGS_LEVEL, __name__, "../../logs/", "benchmark_log")

if __name__ == "__main__":
    assert len(sys.argv) == 5
    model_path = sys.argv[1]
    vnnlib_path = sys.argv[2]
    result_path = sys.argv[3]
    timeout = int(float(sys.argv[4]))

    # Get the "Academic license" print from gurobi at the beginning
    grb.Model()

    model, input_nodes, output_nodes = ONNXParser(model_path).to_pytorch()
    if os.path.isfile(result_path):
        copyfile(result_path, result_path + ".bak")

    a = vnnlib.read_vnnlib_simple(vnnlib_path, input_nodes, output_nodes)
    assert len(a) == 1
    input_bounds, constraints = a[0]
    input_bounds = np.array(input_bounds)
    objectiveMatrix = []
    objectiveBias = []
    for weights, bias in constraints:
        objectiveMatrix.append(weights)
        objectiveBias.append(bias)
    objectives = np.concatenate(
        [np.array(objectiveMatrix), -np.array(objectiveBias)[:, :, np.newaxis]], axis=2)

    with open(result_path, 'w', buffering=1) as f:

        solver = VeriNet(model,
                         gradient_descent_max_iters=5,
                         gradient_descent_step=1e-1,
                         gradient_descent_min_loss_change=1e-2,
                         max_procs=None)
        start = time.time()
        objective = ArbitraryObjective(
            objectives, input_bounds, output_size=output_nodes)
        status = solver.verify(objective, timeout=timeout, no_split=False,
                               gradient_descent_intervals=5, verbose=False)
        if status == Status.Safe:
            f.write("holds")
        elif status == Status.Unsafe:
            f.write("violated")
        else:
            f.write("run_instance_timeout")

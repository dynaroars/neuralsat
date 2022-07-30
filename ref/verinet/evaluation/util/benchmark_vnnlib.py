
"""
Script for running vnnlib benchmarks (formated as in vnncomp'21).

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import time

from tqdm import tqdm

from verinet.parsers.onnx_parser import ONNXParser
from verinet.parsers.vnnlib_parser import VNNLIBParser
from verinet.verification.verinet import VeriNet
from verinet.verification.verifier_util import Status


def parse_instances_csv(instances_csv_path: str):

    """
    Parses the verification query csv file.

    Args:
        instances_csv_path:
            The path of the benchmark csv file.
    """

    queries = []

    with open(instances_csv_path, "r") as f:

        for line in f:
            if len(line.split(",")) != 3:
                continue

            model_name, prop, timeout = line.split(",")
            queries.append((model_name, prop, float(timeout)))

    return queries


def benchmark(result_path: str, benchmark_path: str, instances_csv_path: str, input_shape: tuple,
              transpose_fc_weights: bool = False, precision: int = 32):

    """
    Runs the benchmarks.

    Args:
        result_path:
            The path of the result file.
        benchmark_path:
            The directory containing the benchmarks.
        instances_csv_path:
            The path of the csv file containing the instances.
        input_shape:
            The shape of the models input.
        transpose_fc_weights:
            If true, weights of fc layers are transposed.
        precision:
            The floating point precision for the torch model.
    """

    solver = VeriNet(use_gpu=False)
    instances = parse_instances_csv(instances_csv_path)

    with open(result_path, "w",  buffering=1) as result_file:

        for model_name, prop, timeout in tqdm(instances, desc="Verifying instance"):

            if model_name[-3:] == ".gz":
                model_name = model_name[:-3]

            onnx_parser = ONNXParser(os.path.join(benchmark_path, model_name),
                                     transpose_fc_weights=transpose_fc_weights, use_64bit=precision == 64)
            model = onnx_parser.to_pytorch()
            model.eval()
            vnnlib_parser = VNNLIBParser(os.path.join(benchmark_path, prop))
            objectives = vnnlib_parser.get_objectives_from_vnnlib(model, input_shape=input_shape)

            start_time = time.time()

            branches = 0
            max_depth = 0

            for objective in objectives:
                status = solver.verify(objective, timeout - (time.time() - start_time))

                branches += solver.branches_explored
                max_depth = max(max_depth, solver.max_depth)

                if (time.time() - start_time) > timeout:
                    status = Status.Undecided
                    break

                if status != Status.Safe:
                    break

            result_file.write(f"Network_{model_name}, Property: {prop}, Status: {status}, "
                              f"branches explored: {branches}, max depth: {max_depth}, "
                              f"time: {time.time() - start_time:.2f} seconds\n")

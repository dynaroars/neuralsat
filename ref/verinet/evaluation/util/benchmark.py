

"""
Scripts used for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import time

import numpy as np
import torch
from tqdm import tqdm
from shutil import copyfile

from verinet.verification.verinet import VeriNet
from verinet.verification.verifier_util import Status
from verinet.verification.objective import Objective
from verinet.parsers.onnx_parser import ONNXParser
from verinet.util.logger import get_logger
from verinet.util.config import CONFIG

benchmark_logger = get_logger(CONFIG.LOGS_LEVEL_VERINET, __name__, "../../logs/", "benchmark_log")


# noinspection PyArgumentList,PyShadowingNames,PyCallingNonCallable
def run_benchmark(images: np.array,
                  epsilons: list,
                  model_path: str,
                  mean: float,
                  std: float,
                  timeout: int,
                  result_path: str,
                  targets: np.array = None,
                  use_gpu: bool = True,
                  max_procs: int = None
                  ):

    """
    Runs benchmarking for networks.

    Args:
        images:
            The images used for benchmarking, should be NxM where N is the number of
            images and M is the number of pixels for FC networks and
            NxChannelsxHeightxWidth for convolutional networks.
        epsilons:
            A list with the epsilons (maximum pixel change).
        model_path:
            The path where the nnet model is stored.
        mean:
            The normalising mean of the images.
        std:
            The normalising std of the images.
        timeout:
            The maximum time in settings before timeout for each image.
        result_path:
            The path where the results are stored.
        targets:
            The correct classes for the input, if None the predictions are used as
            correct classes.
        use_gpu:
            Determines whether to use Cuda.
        max_procs:
            The maximum number of procs to use.
    """

    model_onnx = ONNXParser(model_path)
    model = model_onnx.to_pytorch()

    # noinspection PyUnresolvedReferences
    model.eval()

    batch_size = images.shape[0]

    if targets is None:
        torch_images = torch.Tensor(((images.reshape(batch_size, -1)/255)-mean)/std).to(device=model.device)
        targets = model(torch_images)[0].argmax(dim=1).detach().cpu().numpy()

    if os.path.isfile(result_path):
        copyfile(result_path, result_path + ".bak")

    with open(result_path, 'w', buffering=1) as f:

        benchmark_logger.info(f"Starting benchmarking with timeout: {timeout},  model path: {model_path}")
        f.write(f"Benchmarking with:"
                f"Timeout {timeout} seconds \n" +
                f"Model path: {model_path} \n\n")

        solver = VeriNet(use_gpu=use_gpu, max_procs=max_procs)

        for eps in epsilons:

            safe = []
            unsafe = []
            undecided = []
            underflow = []

            benchmark_logger.info(f"Starting benchmarking with epsilon: {eps}")
            f.write(f"Benchmarking with epsilon = {eps}: \n\n")
            solver_time = 0

            for i in tqdm(range(len(images)), desc="Processing images"):

                data_i = images[i]/255

                torch_norm_data_i = torch.Tensor((data_i.reshape((1, *data_i.shape))-mean)/std).to(device=model.device)

                # Test that the data point is classified correctly
                pred_i = model(torch_norm_data_i)[0].argmax(dim=1).detach().cpu().numpy()[0]
                if pred_i != targets[i]:
                    f.write(f"Final result of input {i}: Skipped,correct_label: {targets[i]}, predicted: {pred_i}\n")
                    continue

                # Create input bounds
                input_bounds = np.zeros((*data_i.shape, 2), dtype=np.float32)
                input_bounds[..., 0] = (np.clip((data_i - eps), 0, 1) - mean) / std
                input_bounds[..., 1] = (np.clip((data_i + eps), 0, 1) - mean) / std

                # Run verification
                start = time.time()

                objective = Objective(input_bounds, output_size=10, model=model)
                out_vars = objective.output_vars
                for j in range(objective.output_size):
                    if j != int(targets[i]):
                        # noinspection PyTypeChecker
                        objective.add_constraints(out_vars[j] <= out_vars[int(targets[i])])

                status = solver.verify(objective=objective, timeout=timeout)

                f.write(f"Final result of input {i}: {status}, branches explored: {solver.branches_explored}, "
                        f"max depth: {solver.max_depth}, time spent: {time.time()-start:.3f} seconds\n")
                solver_time += time.time() - start

                if status == Status.Safe:
                    safe.append(i)
                elif status == Status.Unsafe:
                    unsafe.append(i)
                elif status == Status.Undecided:
                    undecided.append(i)
                elif status == Status.Underflow:
                    benchmark_logger.warning(f"Underflow for image {i}")
                    underflow.append(i)

            f.write("\n")
            f.write(f"Time spent in solver: {solver_time}\n")
            f.write(f"Total number of images verified as safe: {len(safe)}\n")
            f.write(f"Safe images: {safe}\n")
            f.write(f"Total number of images verified as unsafe: {len(unsafe)}\n")
            f.write(f"Unsafe images: {unsafe}\n")
            f.write(f"Total number of images timed-out: {len(undecided)}\n")
            f.write(f"Timed-out images: {undecided}\n")
            f.write(f"Total number of images with underflow: {len(underflow)}\n")
            f.write(f"Underflow images: {underflow}\n")
            f.write("\n")

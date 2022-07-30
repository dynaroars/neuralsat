"""
Small script for benchmarking the MNIST networks

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from evaluation.util.benchmark import run_benchmark
from verinet.parsers.input_data_parser import load_mnist_human_readable


if __name__ == "__main__":

    epsilons = [0.02, 0.03, 0.05]
    timeout = 1800

    mean = 0
    std = 1

    num_images = 25

    img_dir = f"../../resources/images/mnist"
    model_path = "../../resources/models/onnx/mnist-net_256x4.onnx"
    result_path = f"../benchmark_results/mnist-net_256x4.txt"

    if not os.path.isdir("../benchmark_results"):
        os.mkdir("../benchmark_results")

    run_benchmark(images=load_mnist_human_readable(img_dir, list(range(1, num_images + 1))).reshape(num_images, -1),
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  use_gpu=False)

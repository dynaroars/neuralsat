"""
Small script for benchmarking the MNIST networks

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from evaluation.util.benchmark import run_benchmark
from verinet.parsers.input_data_parser import load_images_eran


if __name__ == "__main__":

    epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    timeout = 1800

    mean = 0.1307
    std = 0.3081

    num_images = 100

    img_dir = f"../../resources/images/mnist_test.csv"
    model_path = "../../resources/models/onnx/ffnnSIGMOID__PGDK_w_0.1_6_500.onnx"
    result_path = f"../benchmark_results/mnist_sigmoid.txt"

    if not os.path.isdir("../benchmark_results"):
        os.mkdir("../benchmark_results")

    images, targets = load_images_eran(img_dir, image_shape=(28, 28))

    run_benchmark(images=images.reshape(num_images, -1),
                  targets=targets,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  use_gpu=False)

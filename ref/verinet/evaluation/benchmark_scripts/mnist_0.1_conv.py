"""
Small script for benchmarking.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from evaluation.util.benchmark import run_benchmark
from verinet.parsers.input_data_parser import load_images_eran

if __name__ == "__main__":

    epsilons = [0.1]
    timeout = 1800

    mean = 0.1307
    std = 0.3081

    num_images = 100

    img_dir = "../../resources/images/mnist_test.csv"
    model_path = "../../resources/models/onnx/mnist_0.1.onnx"
    result_path = "../benchmark_results/mnist_0.1.txt"

    if not os.path.isdir("../benchmark_results"):
        os.mkdir("../benchmark_results")

    images, labels = load_images_eran(img_csv=img_dir, image_shape=(28, 28))
    images = images.reshape(num_images, 1, 28, 28)

    run_benchmark(images=images,
                  targets=labels,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  use_gpu=True)

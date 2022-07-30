"""
Small script for benchmarking.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np

from evaluation.util.benchmark import run_benchmark
from verinet.parsers.input_data_parser import load_images_eran

if __name__ == "__main__":

    epsilons = [2/255]
    timeout = 1800

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((1, 3, 1, 1))

    num_images = 100

    img_dir = "../../resources/images/cifar10_test.csv"
    model_path = "../../resources/models/onnx/cifar10_2_255.onnx"
    result_path = "../benchmark_results/cifar10_2_255.txt"

    if not os.path.isdir("../benchmark_results"):
        os.mkdir("../benchmark_results")

    images, labels = load_images_eran(img_csv=img_dir)
    images = images.reshape(100, 32, 32, 3).transpose(0, 3, 1, 2)

    run_benchmark(images=images,
                  targets=labels,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  use_gpu=True)

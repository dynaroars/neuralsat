from verinet.util.config import CONFIG

CONFIG.NUM_ITER_OPTIMISED_RELAXATIONS = 3
CONFIG.INDIRECT_HIDDEN_MULTIPLIER = 0.5
CONFIG.INDIRECT_INPUT_MULTIPLIER = 0.75

from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/cifar2020.txt"
    benchmark_path = "../../resources/benchmarks/cifar2020/"
    instances_csv_path = "../../resources/benchmarks/cifar2020/cifar2020_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(3, 32, 32))

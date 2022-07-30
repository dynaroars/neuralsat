from verinet.util.config import CONFIG

CONFIG.PRECISION = 64

from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/eran.txt"
    benchmark_path = "../../resources/benchmarks/eran/"
    instances_csv_path = "../../resources/benchmarks/eran/eran_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(1, 28, 28))

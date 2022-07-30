from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/oval21.txt"
    benchmark_path = "../../resources/benchmarks/oval21/"
    instances_csv_path = "../../resources/benchmarks/oval21/oval21_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(3, 32, 32))

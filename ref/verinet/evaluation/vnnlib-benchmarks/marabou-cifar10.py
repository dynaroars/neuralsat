from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/marabou-cifar10.txt"
    benchmark_path = "../../resources/benchmarks/marabou-cifar10/"
    instances_csv_path = "../../resources/benchmarks/marabou-cifar10/marabou-cifar10_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(32, 32, 3), transpose_fc_weights=True)

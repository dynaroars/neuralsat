from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/cifar10_resnet.txt"
    benchmark_path = "../../resources/benchmarks/cifar10_resnet/"
    instances_csv_path = "../../resources/benchmarks/cifar10_resnet/cifar10_resnet_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(3, 32, 32))

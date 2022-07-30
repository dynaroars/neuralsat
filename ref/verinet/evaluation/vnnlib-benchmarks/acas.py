from verinet.util.config import CONFIG

CONFIG.NUM_ITER_OPTIMISED_RELAXATIONS = 1
CONFIG.INDIRECT_HIDDEN_MULTIPLIER = 0.75
CONFIG.INDIRECT_INPUT_MULTIPLIER = 0.75


from evaluation.util.benchmark_vnnlib import benchmark

if __name__ == '__main__':

    result_path = f"../vnn21_results/acas.txt"
    benchmark_path = "../../resources/benchmarks/acasxu/"
    instances_csv_path = "../../resources/benchmarks/acasxu/acasxu_instances.csv"

    benchmark(result_path, benchmark_path, instances_csv_path, input_shape=(5,), transpose_fc_weights=True)

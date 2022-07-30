import os


def run_sgd_trained_experiment(gpu_id, cpu_list):
    os.system("mkdir -p ./results/sgd8")
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpu_list} " \
              f"python tools/cifar_bound_comparison.py " \
              f"./networks/cifar_sgd_8px.pth 0.01960784313 ./results/sgd8 --from_intermediate_bounds"
    print(command)
    os.system(command)


def run_madry_trained_experiment(gpu_id, cpu_list):
    os.system("mkdir -p ./results/madry8")
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpu_list} " \
              f"python tools/cifar_bound_comparison.py " \
              f"./networks/cifar_madry_8px.pth 0.04705882352 ./results/madry8 --from_intermediate_bounds"
    print(command)
    os.system(command)


if __name__ == "__main__":
    gpu_id = 0
    cpu_list = "0-3"

    run_madry_trained_experiment(gpu_id, cpu_list)
    run_sgd_trained_experiment(gpu_id, cpu_list)

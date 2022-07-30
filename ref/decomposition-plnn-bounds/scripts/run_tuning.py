import os

def run():

    gpu_id = 0
    cpus = "0-3"
    iters = 100

    define_linear_approximation = False

    images = [0, 50, 100, 150, 200]
    adam_algorithms = [  # algo, beta1, inlr, finlr
        ("planet-adam", 0.9, 1e-3, 1e-4),
        ("planet-adam", 0.9, 1e-2, 1e-4),
        ("dj-adam", 0.9, 1e-2, 1e-4),
        ("dj-adam", 0.9, 1e-3, 1e-4),
    ]
    prox_algorithms = [  # algo, momentum, ineta, fineta
        ("proxlp", 0.0, 1e2, 1e2),
        ("proxlp", 0.0, 5e1, 1e2),
        ("proxlp", 0.3, 1e1, 5e2),
    ]

    define_linear_approximation_string = "--define_linear_approximation" if define_linear_approximation else ""

    for img in images:

        for algo, beta1, inlr, finlr in adam_algorithms:

            # have adam run for more as it's faster
            adam_iters = int(iters * 2.6) if algo == "dj-adam" else int(iters * 1.6)

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/cifar_runner.py " \
                f"~/data/cifar_experiments/cifar_sgd_8px.pth 0.01960784313 --algorithm {algo} --out_iters {adam_iters} --img_idx" \
                f" {img} --init_step {inlr} --fin_step {finlr} --beta1 {beta1} {define_linear_approximation_string}"
            print(command)
            os.system(command)

        for algo, momentum, ineta, fineta in prox_algorithms:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/cifar_runner.py " \
                      f"~/data/cifar_experiments/cifar_sgd_8px.pth 0.01960784313 --algorithm {algo} --out_iters {iters} " \
                      f"--img_idx {img} --eta {ineta} --feta {fineta} --prox_momentum {momentum} " \
                      f"{define_linear_approximation_string}"
            print(command)
            os.system(command)


if __name__ == "__main__":
    run()

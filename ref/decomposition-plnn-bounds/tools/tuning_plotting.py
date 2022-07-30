import torch
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from tools.plot_utils import custom_plot


def plot_tuning():

    define_linear_approximation = False

    images = [0, 50, 100, 150, 200]
    adam_algorithms = [
        ("planet-adam", 0.9, 1e-3, 1e-4),
        ("planet-adam", 0.9, 1e-2, 1e-4),
        ("dj-adam", 0.9, 1e-2, 1e-4),
        ("dj-adam", 0.9, 1e-3, 1e-4),
    ]
    prox_algorithms = [
        # algo, momentum, ineta, fineta
        ("proxlp", 0.0, 1e2, 1e2),
        ("proxlp", 0.0, 5e1, 1e2),
        ("proxlp", 0.3, 1e1, 5e2),
    ]

    algorithm_name_dict = {
        "planet-adam": "ADAM",
        "planet-auto-adagrad": "AdaGrad",
        "planet-auto-adam": "Autograd's ADAM",
        "proxlp": "Proximal",
        "jacobi-proxlp": "Jacobi Proximal",
        "dj-adam": "Dvijotham ADAM"
    }

    lin_approx_string = "" if not define_linear_approximation else "-allbounds"

    fig_idx = 0
    for img in images:
        folder = "./timings_cifar/"
        color_id = 0
        for algo, beta1, inlr, finlr in adam_algorithms:
            adam_name = folder + f"timings-img{img}-{algo},istepsize:{inlr},fstepsize:{finlr},beta1:{beta1}{lin_approx_string}.pickle"
            nomomentum = " w/o momentum" if beta1 == 0 else ""

            adam = torch.load(adam_name, map_location=torch.device('cpu'))
            custom_plot(fig_idx, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                        None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\alpha \in$" + f"[{inlr}, {finlr}]" + nomomentum,
                        dotted="-", xlog=False,
                        ylog=False, color=colors[color_id])
            color_id += 1

        for algo, momentum, ineta, fineta in prox_algorithms:

            acceleration_string = ""
            if algo != "jacobi-proxlp":
                acceleration_string += f"-mom:{momentum}"
            prox_name = folder + f"timings-img{img}-{algo},eta:{ineta}-feta:{fineta}{acceleration_string}{lin_approx_string}.pickle"

            acceleration_label = ""
            if momentum:
                acceleration_label += f"momentum {momentum}"

            prox = torch.load(prox_name, map_location=torch.device('cpu'))
            custom_plot(fig_idx, prox.get_last_layer_time_trace(), prox.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                        None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\eta \in$" + f"[{ineta}, {fineta}], " +
                                  f"{acceleration_label}",
                        dotted="-", xlog=False, ylog=False, color=colors[color_id])
            color_id += 1
        fig_idx += 1


if __name__ == "__main__":

    plot_tuning()
    plt.show()

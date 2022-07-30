import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    filenames_verinet = ["VeriNet/mnist_100_imgs_48_relu.txt", "VeriNet/mnist_100_imgs_100_relu.txt",
                         "VeriNet/mnist_100_imgs_1024_relu.txt"]
    filenames_neurify = ["neurify/neurify_mnist_48.txt", "neurify/neurify_mnist_100.txt",
                         "neurify/neurify_mnist_1024.txt"]

    verinet_times = np.zeros(1350)
    verinet_status = np.zeros(1350, dtype=int)
    neurify_times = np.zeros(1350)
    neurify_status = np.zeros(1350, dtype=int)

    i = 0
    for file in filenames_verinet:
        with open(file, "r") as f:
            for line in f:

                line_arr = line.split(" ")

                if line_arr[0] != "Final":
                    continue

                if line_arr[5] == "Status.Safe,":
                    verinet_times[i] = float(line_arr[-2])
                    verinet_status[i] = 0

                elif line_arr[5] == "Status.Unsafe,":
                    verinet_times[i] = float(line_arr[-2])
                    verinet_status[i] = 1

                elif line_arr[5] == "Status.Undecided," or line_arr[5] == "Status.Underflow,":
                    verinet_times[i] = 3600
                    verinet_status[i] = 2
                i += 1

    i = 0
    for file in filenames_neurify:
        with open(file, "r") as f:
            for line in f:

                line_arr = line.split(" ")

                if line_arr[0] != "Final":
                    continue

                if line_arr[5] == "Safe,":
                    neurify_times[i] = float(line_arr[-1])
                    neurify_status[i] = 0

                elif line_arr[5] == "Unsafe,":
                    neurify_times[i] = float(line_arr[-1])
                    neurify_status[i] = 1

                elif line_arr[5] == "Undecided," or line_arr[5] == ",":
                    neurify_times[i] = 3600
                    neurify_status[i] = 2

                elif line_arr[5] == "Manually":
                    neurify_times[i] = np.NAN
                    neurify_status[i] = 3

                elif line_arr == "Underflow,":
                    neurify_times[i] = np.NAN
                    neurify_status[i] = 3

                i += 1

    verified_safe = np.argwhere(((neurify_status == 0) + (verinet_status == 0)) * (neurify_status != 3))
    verified_unsafe = np.argwhere(((neurify_status == 1) + (verinet_status == 1)) * (neurify_status != 3))

    plt.figure()
    plt.axis([0.005, 3600, 0.005, 3600])
    plt.xscale("log")
    plt.yscale("log")
    x = np.linspace(0, 3600, 1000)
    plt.plot(np.linspace(0, 3700), np.linspace(0, 3700), "black", x, x/10, "g:", x, x/100, "y--", x, x/1000, "m-.")
    plt.legend(["0x speedup", "10x speedup", "100x speedup", "1000x speedup"])
    plt.scatter(neurify_times[verified_safe], verinet_times[verified_safe], c="blue", marker="o")
    plt.scatter(neurify_times[verified_unsafe], verinet_times[verified_unsafe], c="red", marker="+")
    plt.legend(["0x speedup", "10x speedup", "100x speedup", "1000x speedup", "Safe", "Unsafe"])
    plt.title("Verification times")
    plt.ylabel("VeriNet time (seconds)")
    plt.xlabel("Neurify time (seconds)")

    plt.show()

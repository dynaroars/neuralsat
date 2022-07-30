
import os
import sys


import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import onnxruntime as onnxrun


# noinspection PyShadowingNames
def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = True) -> tuple:

    """
    Loads the mnist data.

    Args:
        data_dir:
            The directory to store the full MNIST dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(mnist_test, batch_size=10000,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(mnist_test, batch_size=10000)

    images, labels = next(iter(loader_test))

    num_selected = 0
    selected_images, selected_labels = [], []

    sess1 = onnxrun.InferenceSession("./mnist-net_256x2.onnx")
    sess2 = onnxrun.InferenceSession("./mnist-net_256x4.onnx")
    sess3 = onnxrun.InferenceSession("./mnist-net_256x6.onnx")
    sessions = [sess1, sess2, sess3]

    i = -1
    while num_selected < num_imgs:

        i += 1
        correctly_classified = True

        for sess in sessions:
            input_name = sess.get_inputs()[0].name
            result = np.argmax(sess.run(None, {input_name: images[i].numpy().reshape(1, 784, 1)})[0])

            if result != labels[i]:
                correctly_classified = False
                break

        if not correctly_classified:
            continue

        num_selected += 1
        selected_images.append(images[i])
        selected_labels.append(labels[i])

    return selected_images, selected_labels


def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:

    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((img - eps), 0, 1)
    bounds[..., 1] = torch.clip((img + eps), 0, 1)

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; Mnist property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (<= Y_{i} Y_{label}))\n")
        f.write("))")


def create_instances_csv(num_props: int = 15, path: str = "mnistfc_instances.csv"):

    """
    Creates the instances_csv file.

    Args:
        num_props:
            The number of properties.
        path:
            The path of the csv file.
    """

    nets = ["mnist-net_256x2.onnx",
            "mnist-net_256x4.onnx",
            "mnist-net_256x6.onnx"]

    props = [f"prop_{i}_0.03.vnnlib" for i in range(num_props)]
    props += [f"prop_{i}_0.05.vnnlib" for i in range(num_props)]

    with open(path, "w") as f:

        for net in nets:
            timeout = 120 if net == "mnist-net_256x2.onnx" else 300
            for prop in props:

                if net == nets[-1] and prop == props[-1]:
                    f.write(f"{net},{prop},{timeout}")
                else:
                    f.write(f"{net},{prop},{timeout}\n")


if __name__ == '__main__':

    num_images = 15
    epsilons = [0.03, 0.05]

    try:
        torch.random.manual_seed(int(sys.argv[1]))
    except (IndexError, ValueError):
        raise ValueError("Expected seed (int) to be given as command line argument")

    images, labels = load_data(num_imgs=num_images, random=True)

    for eps in epsilons:
        for i in range(num_images):

            image, label = images[i], labels[i]
            input_bounds = create_input_bounds(image, eps)

            spec_path = f"prop_{i}_{eps:.2f}.vnnlib"

            save_vnnlib(input_bounds, label, spec_path)

    create_instances_csv()

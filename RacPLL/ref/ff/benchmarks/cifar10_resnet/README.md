A ResNet Benchmark on CIFAR-10 for Neural Network Verification
-----

We propose a new set of benchmarks of residual networks (ResNet) on CIFAR-10
for neural network verification in this repository.

Currently, most networks evaluated in the literature are feedforward NNs, and
many tools are hardcoded to handle feedforward networks only. To make neural
network verification more useful in practical scenarios, we advocate that tools
should handle more general architectures, and ResNet is the first step towards
this goal. We hope this can provide some incentives for the community to
develop better tools.

**Model details**: We provided two adversarially trained ResNet models on CIFAR-10 with the following structures:

- ResNet-2B with 2 residual blocks: 5 convolutional layers + 2 linear layers
- ResNet-4B with 4 residual blocks: 9 convolutional layers + 2 linear layers

The ONNX format networks are available in the [onnx](onnx/) folder, and PyTorch models
can be found in the [pytorch_model](pytorch_model) folder. PyTorch model definitions
are available [here](pytorch_model/resnet.py).

Since this is one of the first benchmarks using ResNet in VNN-COMP, we provide a
relatively small network (ResNet-2B, with 2 residual blocks). We also include a deeper and
wider network (ResNet-4B, with 4 residual blocks) to test the scalability of tools.
Networks are trained using adversarial training with L∞ perturbation
epsilon (2/255). We report basic model performance numbers below:

| Model      | # ReLUs | Clean acc. |  PGD acc. <br> ε=2/255  |  PGD acc. <br> ε=1/255 | CROWN/DeepPoly <br> verified acc. <br> ε=2/255 | CROWN/DeepPoly <br> verified acc. <br> ε=1/255 |
|------------|---------|------------|-----------------|----------------|-----------------------------------|-----------------------------------|
| ResNet-2B  |   6244  |    69.25%  |      54.82%     |      62.24%    |   26.88%                          |   57.16%                          |
| ResNet-4B  |  14436  |    77.20%  |      61.41%     |      69.75%    |    0.24%                          |   23.28%                          |

Since the models are trained using adversarial training, it also poses a
challenge for many verifiers - the CROWN/DeepPoly verified accuracy (as a
simple baseline) is much lower than PGD accuracy, and we hope this benchmark
can motivate researchers in the community to develop stronger tools that can
make this gap smaller. In this year, to reduce difficulty, we reduce `ε` 
to `1/255` for the ResNet-4B model.

**Data Format**: The input images should be normalized using mean and std
computed from CIFAR-10 training set. The perturbation budget is element-wise,
eps=2/255 on unnormalized images and clipped to the [0, 1] range. We provide
`cifar_eval.py` as a simple PyTorch example of loading data (e.g., data
preprocessing, channel ordering etc).

**Data Selection**: We propose to randomly select 48 images from the test set
for the ResNet-2B and 24 images for the ResNet-4B.  The images are classified
correctly and cannot be attacked by a 100-step PGD attack with 5 random
restarts.  For each image, we specify the property that the groundtruth label
is always larger than all the other labels within L∞ perturbation `ε=2/255` on
input for ResNet-2B and `ε=1/255` for ResNet-4B. The `ε` for ResNet-4B is
smaller to reduce its difficulty, and it can be increased in future
evaluations. The Per-example timeout is set to 5 minutes and the overall
runtime is guaranteed to be less than 6 hours.

See instructions below for generating test images with a script, and some example
properties are in the `vnnlib_properties_pgd_filtered` folder.


**Generating properties**: To generate properties random images 
and verification properties (i.e., the property is that the true label is larger
than all the other labels under perturbation) that are classified correctly and
are also robust against 100-step PGD attacks with 5 random restarts, please run:

```bash
cd pytorch_model
python generate_properties_pgd.py --seed <seed> --device [cpu|gpu]
```

**Citation:** If you use our ResNet benchmarks in your research, please kindly cite our paper:

```
@article{wang2021betacrown,
  title={Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}
```

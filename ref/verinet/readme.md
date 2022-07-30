# VeriNet

The VeriNet toolkit is a state-of-the-art sound and complete symbolic 
interval propagation based toolkit for verification of neural networks. VeriNet 
won second place overall and was the most performing among toolkits not using GPUs in the [2nd 
international verification of neural networks competition](https://arxiv.org/pdf/2109.00498.pdf).
VeriNet is devloped at the 
[Verification of Autonomous Systems (VAS) group, Imperial College London](https://vas.doc.ic.ac.uk).

## Relevant Publications. 

VeriNet is developed as part of the following publications:

[Efficient Neural Network Verification via Adaptive Refinement and Adversarial Search](https://ecai2020.eu/papers/384_paper.pdf)  

[DEEPSPLIT: An Efficient Splitting Method for Neural Network Verification via Indirect Effect Analysis](https://www.ijcai.org/proceedings/2021/351)

This version of VeriNet subsumes the VeriNet toolkit publised in the first 
paper and the DeepSplit toolkit published in the second paper.

## Installation:

All dependencies can be installed via Pipenv. 

VeriNet depends on the Xpress solver, which can solve smaller problems without a 
license; however, larger problems (networks with more than approximately 5000 nodes) 
require a license. Free academic licenses can be obtained at: 
https://content.fico.com/l/517101/2018-06-10/3fpbf

We recommend installing the developer dependencies for some extra optimisations during 
the loading of onnx models. These can be installed by running pipenv with the --dev option: 
$pipenv install --dev.

## Usage:

### Models: 

VeriNet supports loading models in onnx format or custom models created 
with the VeriNetNN class, a subclass of torch.nn.module. 

#### Loading onnx models:

Onnx models can be loaded as follows:

```python
from verinet.parsers.onnx_parser import ONNXParser

onnx_parser = ONNXParser(onnx_model_path, input_names=("x",), transpose_fc_weights=False, use_64bit=False)
model = onnx_parser.to_pytorch()
model.eval()
```

The first argument is the path of the onnx file; input_names should be a tuple containing the input-variable name 
as stored in the onnx model; if transpose_fc_weights is true the weight matrices of fully-connected layers are 
transposed; if use_64bit is true the parameters of the model are stored as torch.DoubleTensors instead of 
torch.FloatTensors. 

#### Custom models: 

The following is a simple example of a VeriNetNN model with two inputs, one 
FC layer, one ReLU layer and 2 outputs:

```python
import torch.nn as nn
from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode

nodes = [VeriNetNNNode(idx=0, op=nn.Identity(), connections_from=None, connections_to=[1]),
         VeriNetNNNode(idx=1, op=nn.nn.Linear(2, 2)(), connections_from=[0], connections_to=[2]),
         VeriNetNNNode(idx=2, op=nn.ReLU(), connections_from=[1], connections_to=[3]),
         VeriNetNNNode(idx=3, op=nn.Identity(), connections_from=[2], connections_to=None)]

model = VeriNetNN(nodes)
```

VeriNetNN takes as input a list of nodes (note that 'nodes' here do not correspond to 
neurons, each node may have multiple neurons) where each node has the following parameters:
 
* idx: A unique node-index sorted topologically wrt the connections.  
* op: The operation performed by the node, all operations defined in 
verinet/neural_networks/custom_layers.py as well as nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Linear, 
nn.Conv2d, nn.AvgPool2d, nn.Identity, nn.Reshape, nn.Transpose and nn.Flatten are supported. 
* connections_from: A list of which nodes' outputs are used as input in this node. Note 
that more than one output in a single node (residual connections) is only support for 
nodes with the AddDynamic op as defined in custom_layers.py. 
* connections_to: A list of which nodes' input depend on this node's output corresponding 
to connections_from. 

The first and last layer should be nn.Identity nodes. BatchNorm2d and MaxPool2d operations
can be implemented by saving the model to onnx and reloading as the onnx parser 
automatically attempts to convert these to equivalent Conv2d and ReLU layers. 

### Verification Objectives:

VeriNet supports verification objectives in the VNN-COMP'21 vnnlib format and 
custom objectives. 

#### Vnnlib:

VeriNet supports vnnlib files formated as described in the following discussion:
https://github.com/stanleybak/vnncomp2021/issues/2. The files can be loaded as follows:

```python
from verinet.parsers.vnnlib_parser import VNNLIBParser

vnnlib_parser = VNNLIBParser(vnnlib_path)
objectives = vnnlib_parser.get_objectives_from_vnnlib(model, input_shape)
```

The vnnlib_path parameter should be the path of the vnnlib file, model is the VeriNetNN
model as discussed above while the input shape is a tuple describing the shape of the
models input without batch-dimension (e.g. (784, ) for flattened MNIST (1, 28, 28) for 
MNIST images and (3, 32, 32) for CIFAR-10 Images). 

#### Custom objectives:

The following is an example of how a custom verification objective for classification
problems can be encoded (correct output larger than all other outputs): 

```python
from verinet.verification.objective import Objective

objective = Objective(input_bounds, output_size=10, model=model)
out_vars = objective.output_vars
for j in range(objective.output_size):
    if j != correct_output:
        # noinspection PyTypeChecker
        objective.add_constraints(out_vars[j] <= out_vars[correct_output])
```

Here input bounds is an array of shape (*network_input_shape, 2) where network_input_shape
is the input shape of the network (withut batch dimension) and the last dimension 
contains the lower bounds at position 0 and upper bounds at position 1. 

Note that the verification objective encodes what it means for the network to be Safe/Robust. 
And-type constraints can be encoded by calling objective.add_constraints for each and-clause, 
while or-type constraints can be encoded with '|' (e.g. ```(out_vars[0] < 1) | (out_vars[0] < out_vars[1]))```. 

### Verification: 

After defining the model and objective as described above, verification is performed by
using the VeriNet class as follows: 

```python
from verinet.verification.verinet import VeriNet

solver = VeriNet(use_gpu=True, max_procs=None)
status = solver.verify(objective=objective, timeout=3600)
```
 
The parameters of VeriNet, use_gpu and max_procs, determines whether to use the GPU 
and the maximum number of processes (max_procs = None automatically determines the 
number of processes depending on the cores available). 

The parameters in solver.verify correspond to the objective as discussed above and 
the timeout in seconds. Note that is recommended to keep solver alive instead of creating 
a new object every call to reduce overhead. 

After each verification run the number of branches explored and maximum depth reached 
are stored in solver.branches_explored and solver.max_depth, respectively. If the objective
is determined to be unsafe/not-robust, a counter example is stored in solver.counter_example.

At the end of each run, status will be either Status.Safe, Status.Unsafe, Status.Undecided or 
Status.Underflow. Safe means that the property is robust, Unsafe that a counter example 
was found, undecided that the solver timed-out and underflow that an error was encountered, 
most likely due to floating point precision. 
 
### Advanced usage:

#### Environment variables:

The .env file contains some environment variables that are automatically enabled 
if pipenv is used, if pipenv is not used make sure to export these variables. 

#### Config file:

The config file in verinet/util/config.py contains several advanced settings. Of 
particular interest are the following:

* PRECISION: (32 or 64) The floating point precision used in SIP. Note that this 
does not affect the precision of the model itself, which can be adjusted in the 
ONNXParser as discussed above. 
* MAX_ESTIMATED_MEM_USAGE: The maximum estimated memory usage acceptable in SIP. Can
be reduced to reduce the memory requirements at the cost of computational performance. 
* USE_SSIP and STORE_SSIP_BOUNDS: Performs a pre-processing using a lower cost 
SIP-variant. Should be enabled if the input-dimensionality is significantly smaller 
than the size of the network (e.g. less than 50 input nodes with more than 10k Relu nodes).

### Contact:

verinet-CO@groups.imperial.ac.uk

### Authors:

Patrick Henriksen: patrick@henriksen.as  
Alessio Lomuscio. 

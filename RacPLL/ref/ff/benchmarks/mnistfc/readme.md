## Fully-connected MNIST Benchmarks. 

This benchmark set consists of 3 fully-connected networks with 2, 4 and 6 
layers and 256 ReLU nodes in each layer.  

For each network we test 15 images with l_infty < eps pertubations 
using eps = 0.03 and eps = 0.05.  

#### Generating vnnlib specifications:

Run: 
```bash
$ generate.py 'seed'
```
The command line argument 'seed' is an integer value used to draw random 
images from the MNIST test set. The script
generates the vnnlib properties as well as the mnistfc_instances.csv
file. 

#### Expected accuracies:

The networks have the following test-set accuracies:

mnist-net_256x2.onnx:  0.980799  
mnist-net_256x4.onnx:  0.976400  
mnist-net_256x6.onnx:  0.968900
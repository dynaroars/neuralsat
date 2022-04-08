
acasxu
---------
This benchmark contains -
   45 fully connected networks with 6 hidden layers and 5 inputs , 5 outputs and 5 ReLU nodes in each layer.
   10 properties
   10 instances out of 450(45\*10) instances  are considered as ACAS-HARD

ACASXU-HARD: 

net 4-6, prop 1

net 4-8, prop 1

net 3-3, prop 2

net 4-2, prop 2

net 4-9, prop 2

net 5-3, prop 2

net 3-6, prop 3

net 5-1, prop 3

net 1-9, prop 7

net 3-3, prop 9

mnistfc
---------
This benchmark contains -
      3 fully-connected networks with 2, 4 and 6 layers and 256 ReLU nodes in each layer.
      15 images with l_infty < eps pertubations using eps = 0.03 and eps = 0.05.


cifar0-resnet
-------------

This benchmark contains two adversarially trained residual networks (ResNet) models on CIFAR-10 with the following structures:

   ResNet-2B with 2 residual blocks: 5 convolutional layers + 2 linear layers
   ResNet-4B with 4 residual blocks: 9 convolutional layers + 2 linear layers

   Activation function : ReLU

   48 images from the test set for the ResNet-2B and 24 images for the ResNet-4B.

   L∞ perturbation ε=2/255 on input for ResNet-2B and ε=1/255 for ResNet-4B. 


marabou-cifar10
-----------------
This folder contains -
     targeted robustness query for three convolutional ReLU networks trained on the CIFAR10 dataset.

Details are available in marabou-cifar10/info.txt 

cifar-2020
---------------

This benchmark contains-
     cifar-10 convolutional networks
     The epsilon values of the L_oo ball for the CIFAR10 networks are 2/255 and 8/255. 

   Actvation function used : ReLU

nn4sys
-----------
This benchmark is for verifying learned indexes of databases. To meet the high precision requirement,
the indexing is completed by two stages. 

   In the first stage, index space is divided into N pieces. A network called v1-network predicts the index of the
   key roughly to know in which piece the index locates.

   Then in the second state, the corresponding v2-network that is specifically trained for this piece predicts the acurate index of the key.


In this benchmark, the v1-network is a 5 layer fully connected network, with 100 neurons each hidden layer. 
Two different settings for the v2-network. 
      1. N=100, each v2-network is 2 layer fully connected network, with 600 neurons in the hidden layer.
      2. N=1000, each v2-network is 2 layer fully connected network, with 6 neurons in the hidden layer.

All the layers use ReLU activation except the last layer.
            
           

**Note : All networks are big in size, hence stored in .gz format. Please unzip all networks before run using -
           
         gzip -d <.gz file>
         
**Note : Replace all .gz file path by .onnx filepath in nn4sys_instances.csv 


oval21
---------
This benchmark set includes 3 ReLU-based convolutional networks, provided in the nets folder,
 which were robustly trained using [1] against l_inf perturbations of radius eps=2/255 on CIFAR10.


verivital
------------
The benchmarks consist of two MNIST classifiers, one with a maxpooling layer and the other with an average pooling layer.

For both the networks we suggest 20(randomly selected) images:

for the average pooling network with a perturbation radii of 0.02 and 0.04 and a timeout of 5 minutes.
for the max pooling network with a perturbation radii of 0.004 and a timeout of 7 minutes.
The network expects the input image to be a Tensor of size (Nx1x28x28)[i.e NCHW] and normalized between [0,1].

test
----------
The folder contains acasxu networks 1-6 (unsat), 1-7 (sat).
The properties correspond to property 3 of acasxu benchmark

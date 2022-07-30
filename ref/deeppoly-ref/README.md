# RIAI 2020 Course Project
This repository contains the code for reproducing the results of our project in the 2020 Reliable and Interpretable Artificial Intelligence (RIAI) class at ETH Zurich. The task of this year’s RIAI course project was to use [DeepPoly](https://dl.acm.org/doi/abs/10.1145/3290354) in order to build a verifier that is as precise and scalable as possible while keeping the soundness property. For evaluation, ten different trained neural networks were given together with two test cases each. The networks were trained on the MNIST dataset. Please find our project report [here](Project_Report.pdf).

## Folder structure
In the directory `code` you can find 5 files. 
File `deeppoly.py` contains a simple python implementation of DeepPoly for `Linear`, `ReLU`, and `Convolutional` layers.
File `networks.py` contains encoding of fully connected and convolutional neural network architectures as PyTorch classes.
File `verifier.py` contains a template of verifier. Loading of the stored networks and test cases is already implemented in `main` function. 
File `optimizer.py`contains our optimizer.
File `utils.py` contains some helper functions.

In folder `mnist_nets` you can find 10 neural networks (7 fully connected and 3 convolutional). These networks are loaded using PyTorch in `verifier.py`.
In folder `test_cases` you can find 10 subfolders. Each subfolder is associated with one of the networks, using the same name. In a subfolder corresponding to a network, you can find 2 test cases for this network.  The file `gt.py` contains the ground truth for each test case.

## Install dependencies

```bash
$ pip install -r requirements.txt
```

## Running the verifier

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `fc1, fc2, fc3, fc4, fc5, fc6, fc7, conv1, conv2, conv3`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To evaluate the verifier on all networks and sample test cases you can use the evaluation script.
You can run this script using the following commands:

```bash
$ bash evaluate
```

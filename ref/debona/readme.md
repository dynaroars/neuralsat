# Debona

This toolkit is a fork of [VeriNet](https://vas.doc.ic.ac.uk/papers/20/ecai20-HL.pdf). For the original source code, please refer to https://vas.doc.ic.ac.uk/software/neural/.

It implements an improvement by computing independent upper and lower symbolic bounds, instead of requiring them to be parallel to each other.
This idea has been described in [Debona: Decoupled Boundary Network Analysis for Tighter Bounds and Faster Adversarial Robustness Proofs [C. Brix, T. Noll, 2020]](https://arxiv.org/abs/2006.09040) but was independently previously published in [An Abstract Domain for Certifying Neural Networks [G. Singh et al., 2019]](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf).


## Installation

The installation can be done by running `install_tool.sh`.
As the last step, you have to enter a valid Gurobi licence key. You may also provide it later by changing into the `src` directory and running `pipenv run grbgetkey [KEY]`.  
*This will deinstall older python versions*, so execute it on a separate system or inside a docker container!

## Supported Architectures

Currently, only fully connected layers are supported.
However, Both ReLU and s-shaped non-linear activation functions can be used.

## Authors
Underlying toolkit (VeriNet):  
Patrick Henriksen: ph818@ic.ac.uk  
Alessio Lomuscio

Modifications:  
Christopher Brix: brix@cs.rwth-aachen.de  
Thomas Noll: noll@cs.rwth-aachen.de

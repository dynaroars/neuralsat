# ADMM-NN-Reachability
Codes for finite-step reachability analysis of neural network dynamical systems presented in the paper *One-shot Reachability Analysis of Neural Network Dynamical Systems, Shaoru Chen, Victor M. Preciado, Mahyar Fazlyab, 2021, submitted*. This paper compares the one-shot method, which over-approximates the output of the k-step composition of neural network dynamics directly, and the recursive method, which apply the one-step over-approximation iteratively for k-steps, in finite-step reachable set over-approximation. Figures of the paper can be generated in [examples/cartpole](https://github.com/ShaoruChen/ADMM-NN-Reachability/tree/main/examples/cartpole) and the Alternate Direction Method of Multipliers (ADMM) on neural network over-approximation is implemented in [nn_reachability/ADMM](https://github.com/ShaoruChen/ADMM-NN-Reachability/blob/main/nn_reachability/ADMM.py).

The package [pympc](https://github.com/TobiaMarcucci/pympc) by Tobia Marcucci is included in this repo. for polyhedron operation and plotting. [LiRPA](https://github.com/KaidiXu/auto_LiRPA) is used for generating the fast linear bounds. 
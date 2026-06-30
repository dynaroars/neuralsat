"""
A simple example for bounding neural network outputs using LP/MIP solvers.

Auto_LiRPA supports constructing LP/MIP optimization formulations (using
Gurobi).  This example uses LP to solve all intermediate layer bounds and
final layer bounds, reflecting the setting in the paper "A Convex
Relaxation Barrier to Tight Robustness Verification of Neural Networks".
This is sometimes referred to as the LP-Full setting. This is in general,
very slow; alpha-CROWN is generally recommended to compute intermediate
layer bound rather than LP.

Example usage: python lp_full.py --index 0 --norm 2.0 --perturbation 1.0

Here `--index` is the dataset index (MNIST in this example), `--norm` is
the Lp perturbation norm used and `--perturbation` is the magnitude of
the perturbation added to model input.
"""

import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators import BoundLinear, BoundConv
import gurobipy as grb
import time
import numpy as np
import argparse

# Help function for generating output matrix. This function used for 
# generating matrix C to calculate the margin between true class and 
# the other classes.
def build_C(label, classes):
    """
    label: shape (B,). Each label[b] in [0..classes-1].
    Return:
        C: shape (B, classes-1, classes).
        For each sample b, each row is a "negative class" among [0..classes-1]\{label[b]}.
        Puts +1 at column=label[b], -1 at each negative class column.
    """
    device = label.device
    batch_size = label.size(0)
    
    # 1) Initialize
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    
    # 2) All class indices
    # shape: (1, K) -> (B, K)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 3) Negative classes only, shape (B, K-1)
    # mask out the ground-truth
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    
    # 4) Scatter +1 at each sample’s ground-truth label
    #    shape needed: (B, K-1, 1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    
    # 5) Scatter -1 at each row’s negative label
    #    We have (B, K-1) negative labels. For row j in each sample b, neg_cls[b, j] is that row’s negative label
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    # shape: (B, K-1)
    
    # We can do advanced indexing:
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    
    return C
    
parser = argparse.ArgumentParser()
parser.add_argument('--index', default=0, type=int, help='Index of data example (from MNIST dataset).')
parser.add_argument('--norm', default='inf', type=str, help='Input perturbation norm.')
parser.add_argument('--perturbation', default=0.05, type=float, help='Input perturbation magnitude.')
parser.add_argument('--lr', default=0.5, type=float, help='Learning rate for alpha_crown.')
parser.add_argument('--iteration', default=30, type=int, help='Iterations for alpha_crown.')
args = parser.parse_args()

## Step 1: Define computational graph by implementing forward()
# You can create your own model here.
model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
# Optionally, load the pretrained weights.
checkpoint = torch.load('./models/spectral_NOR_MLP_B.pth', weights_only=True)
model.load_state_dict(checkpoint)

## Step 2: Prepare dataset.
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())

n_classes = 10
image = test_data.data[args.index].to(torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
true_label = torch.tensor([test_data.targets[args.index]])

## Step 3: Define perturbation.
eps = args.perturbation
norm = float(args.norm)
# The upper bound and lower bound of mnist dataset is [0,1],
# replace the bounds if using other dataset.
if norm == float('inf'):
    x_U = None
    x_L = None
else:
    x_U = torch.ones_like(image)
    x_L = torch.zeros_like(image)
ptb = PerturbationLpNorm(norm = norm, eps = eps, x_U = x_U, x_L = x_L)
print(f'Verification of MNIST data index {args.index} with L{args.norm} perturbation of {args.perturbation}\n')
# Here we only use one image as input.
image = BoundedTensor(image, ptb)
print('Running LP-Full with LPs for all intermediate layers...')
start_time = time.time()

## Step 4: Compute the bounds of different methods.
# For CROWN/alpha-CROWN, we use the compute_bounds() method.
# For LP and MIP, we use the build_solver_module() method.
interm_bounds = {}
lirpa_model = BoundedModule(model, image, device=image.device)
# Store the output shape for each layer first
for node in lirpa_model.nodes():
    # For each intermediate layers, we first set their bound to be infinity as placeholder.
    if hasattr(node, 'output_shape'):
        interm_lb = torch.full(node.output_shape, -float('inf'))
        interm_ub = torch.full(node.output_shape, float('inf'))
        interm_bounds[node.name] = [interm_lb, interm_ub]

# C is the specification matrix (groundtruth - target class).
C = build_C(true_label, classes=n_classes)
# Here we assume that the last node is the model output, and we start from intermdiate layers first.
# Technically, here we need a topological sort of all model nodes if the computation graph is general.
for node in lirpa_model.nodes():
    # For simplicity, we assume the model contains linear, conv, and ReLU layers.
    # We need to calculate the preactivation bounds before each ReLU layer, which are the bounds for linear of conv layers.
    if isinstance(node, (BoundLinear, BoundConv)):
        interm_lb = torch.full(node.output_shape, -float('inf'))
        interm_ub = torch.full(node.output_shape, float('inf'))
        if node.is_final_node:
            print(f'Solving LPs for final layer bounds...')
            # Last node, all intermediate layer bounds have been obtained.
            # For last node, we need to use the specification matrix C to calculate the bounds on groundtruth - target labels.
            solver_vars = lirpa_model.build_solver_module(model_type='lp', x=(image,), final_node_name=node.name, interm_bounds=interm_bounds, C=C)
            lirpa_model.solver_model.setParam('OutputFlag', 0)
            final_lb = torch.empty(n_classes-1)
            final_ub = torch.empty(n_classes-1)
            for i in range(n_classes-1):
                print(f'Solving class {i}...')
                # Now you can define objectives based on the variables on the output layer.
                # And then solve them using gurobi. Here we just output the lower and upper
                # bounds for each output neuron.
                # Solve upper bound.
                lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MAXIMIZE)
                lirpa_model.solver_model.optimize()
                # If the solver does not terminate, you will get a NaN.
                if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                    final_ub[i] = lirpa_model.solver_model.objVal
                # Solve lower bound.
                lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MINIMIZE)
                lirpa_model.solver_model.optimize()
                if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                    final_lb[i] = lirpa_model.solver_model.objVal
        else:
            print(f'Solving LPs for layer {node.name} intermediate layer bounds...')
            # Solve intermediate layer bounds, one by one.
            solver_vars = lirpa_model.build_solver_module(model_type='lp', x=(image,), final_node_name=node.name, interm_bounds=interm_bounds)
            lirpa_model.solver_model.setParam('OutputFlag', 0)
            # For linear layer, the solver_vars shape is: (neurons).
            if isinstance(node, BoundLinear):
                for i, var in enumerate(solver_vars):
                    lirpa_model.solver_model.setObjective(var, grb.GRB.MAXIMIZE)
                    lirpa_model.solver_model.optimize()
                    if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                        interm_ub[0][i] = lirpa_model.solver_model.objVal
                    # Solve lower bound.
                    lirpa_model.solver_model.setObjective(var, grb.GRB.MINIMIZE)
                    lirpa_model.solver_model.optimize()
                    if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                        interm_lb[0][i] = lirpa_model.solver_model.objVal
            # For convolutional layer, the solver_vars shape is (channel, out_w, out_h).
            elif isinstance(node, BoundConv):
                for i,channel in enumerate(solver_vars):
                    for j, row in enumerate(channel):
                        for k, var in enumerate(row):
                            lirpa_model.solver_model.setObjective(var, grb.GRB.MAXIMIZE)
                            lirpa_model.solver_model.optimize()
                            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                interm_ub[0][i][j][k] = lirpa_model.solver_model.objVal
                            # Solve lower bound.
                            lirpa_model.solver_model.setObjective(var, grb.GRB.MINIMIZE)
                            lirpa_model.solver_model.optimize()
                            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                interm_lb[0][i][j][k] = lirpa_model.solver_model.objVal
            interm_bounds[node.name] = [interm_lb, interm_ub]
        print(f'Finished solving layer {node.name} with {len(solver_vars)} neurons')
end_time = time.time()
lp_time = end_time - start_time
print(f'LP-Full time: {lp_time}\n')

lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': args.iteration, 'lr_alpha': args.lr}})
start_time = time.time()
print(f'Running alpha-CROWN with {args.iteration} iterations and learning rate of {args.lr}...')
crown_lb, crown_ub = lirpa_model.compute_bounds(x=(image, ), C=C, method='alpha-CROWN')
end_time = time.time()
alpha_crown_time = end_time - start_time
print(f'alpha-CROWN time: {alpha_crown_time}')

# Step 5: output the final results of each method.
print(f'\nResults for dataset index: {args.index}')
print(f'LP-Full bounds:')
for i in range(n_classes-1):
    if i == true_label.item():
        label = i + 1
    else:
        label = i
    print('{l:8.3f} <= f_{k} - f_{j} <= {u:8.3f}'.format(
        k=true_label.item(), j=label, l=final_lb[i].item(), u=final_ub[i].item()))

# Alpha-CROWN should achieve similar results as LP full but without running any LPs.
print(f'\nalpha-CROWN bounds:')
for i in range(n_classes-1):
    if i == true_label.item():
        label = i + 1
    else:
        label = i
    print('{l:8.3f} <= f_{k} - f_{j} <= {u:8.3f}'.format(
        k=true_label.item(), j=label, l=crown_lb[0][i].item(), u=crown_ub[0][i].item()))
print(f'alpha-CROWN bounds and LP-full bounds should be close for Linf norm; '
      'adjust the number of iterations and learning rate when necessary.\n')

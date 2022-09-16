import torch.nn as nn
import torchvision
import torch
import time
import os

from src.bound_general import BoundedModule
from src.bound_tensor import BoundedTensor
from src.perturbations import PerturbationLpNorm

torch.manual_seed(0)

## Step 1: Define computational graph by implementing forward()
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,32),
        nn.ReLU(),
        nn.Linear(32,32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model

model = mnist_model()

# checkpoint = torch.load(os.path.join(os.path.dirname(__file__),"pretrain/mnist_a_adv.pth"), map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

# For illustration we only use 2 image from dataset
N = 2
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
image = image.to(torch.float32) / 255.0

print(image.sum())

lirpa_model = BoundedModule(model, torch.empty_like(image))
eps = 0.3
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)


image = BoundedTensor(image, ptb)

pred = lirpa_model(image)

# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', ]:
# for method in ['backward (CROWN)', ]:
for method in ['CROWN-Optimized (alpha-CROWN)']:
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    print(method.split()[0])
    print('lower:', lb)
    print('upper:', ub)
    print('lower:', lb.sum())
    print('upper:', ub.sum())
    print()
    exit()

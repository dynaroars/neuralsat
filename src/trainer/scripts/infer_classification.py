import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import argparse
import torch
import sys
import os

from models.vit.vit import *


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


@torch.no_grad()
def test(test_loader, model, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout, disable=False)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X).argmax(-1)
        batch_acc = y.eq(y_pred).sum() / len(y)
        running_metric += batch_acc.item() * X.size(0)
    return running_metric / len(test_loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args


def main():
    args = parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        download=True,
        transform=transform,
        train=False,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    checkpoint_path = os.path.join(args.save_dir, f'{args.output_name}.pt')
    assert os.path.exists(checkpoint_path), f'{checkpoint_path=}'

    model = eval(args.output_name)()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    model.to(args.device)
    model.eval()
    print(model)

    params = get_model_params(model)
    print(f'{params=}')
    
    test_acc = test(
        test_loader=test_loader, 
        model=model, 
        device=args.device,
    )

    print('Test Accuracy:', test_acc)

if __name__ == '__main__':
    main()

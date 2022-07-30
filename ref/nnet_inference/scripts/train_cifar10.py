import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(train_loader, model, criterion, optimizer, device='cpu'):
    """train function"""
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        Y_rec = model(X)
        loss = criterion(Y_rec, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        pbar.set_description(f'[Training iter {batch_id + 1}/{len(train_loader)}]'
                             f' batch_loss={loss.item():.03f}')
    return running_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_loader, model, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X).argmax(-1)

        batch_acc = y.eq(y_pred).sum() / len(y)

        running_metric += batch_acc.item() * X.size(0)
        pbar.set_description(f'[Validation iter {batch_id + 1}/{len(test_loader)}]'
                             f' batch_acc={batch_acc.item():.03f}')
    return running_metric / len(test_loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--model', default='cnn')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


class CIFAR10FNN(nn.Module):
    def __init__(self, n_classes):
        super(CIFAR10FNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, n_classes):
        super(CIFAR10CNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 6, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(6, 12, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(12, 16, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 1028),
            nn.ReLU(),
            nn.Linear(1028, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        transform=transform,
        download=True)
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        transform=transform,
        download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == 'fnn':
        model = CIFAR10FNN(n_classes=len(train_set.classes))
    elif args.model == 'cnn':
        model = CIFAR10CNN(n_classes=len(train_set.classes))
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # train
    print('\n\n[Training]')
    for epoch in range(args.max_epoch):
        print(f'[Epoch {epoch + 1} / {args.max_epoch}]')
        train_epoch_loss = train(train_loader, model, criterion, optimizer, args.device)
        print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
              f'train_loss={train_epoch_loss:.4f}')

        val_epoch_acc = test(test_loader, model, args.device)
        print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
              f'val_acc={val_epoch_acc:.4f}')
        scheduler.step(val_epoch_acc)

    # save
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_file = os.path.join(args.save_dir, f'cifar10_{args.model}.pt')
    torch.save(model.state_dict(), model_save_file)


if __name__ == '__main__':
    main()

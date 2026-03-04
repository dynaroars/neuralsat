import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from contextlib import suppress
import torch.nn as nn
from tqdm import tqdm
import torchvision
import argparse
import torch
import sys
import os

from ..timm.utils import AverageMeter, random_seed, CheckpointSaver
from ..timm.scheduler import create_scheduler_v2

from ..models.rnn.mnist import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def train(epoch, train_loader, model, criterion, optimizer, scheduler, amp_autocast=suppress(), device='cpu'):
    """train function"""
    model.train()

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout, disable=True)
    num_updates = epoch * len(train_loader)
    losses_m = AverageMeter()
    
    
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)
        
        with amp_autocast:
            Y_pred = model(X)
            loss = criterion(Y_pred, y)

        losses_m.update(loss.item(), X.size(0))

        # clean
        optimizer.zero_grad()
        loss.backward(create_graph=second_order)
        optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        
        running_loss += loss.item() * X.size(0)
        pbar.set_description(
            f'[Training iter {batch_id + 1}/{len(train_loader)}]'
            f' batch_loss={loss.item():.03f}'
        )
        
        if scheduler is not None:
            scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
    return running_loss / len(train_loader.dataset)



@torch.no_grad()
def test(test_loader, model, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout, disable=True)
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
    parser.add_argument('--output_name', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--data_dir', default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--save_dir', default=os.path.join(ROOT_DIR, 'weights'))
    parser.add_argument('--model', type=str, default='rnn', choices=['gru', 'lstm'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--saver', action='store_true')

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args


def main():
    args = parse_args()
    
    random_seed(args.seed)
    
    output_dir = f'{args.save_dir}/{args.output_folder}/{args.output_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_class = torchvision.datasets.MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = dataset_class(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True)
    
    test_set = dataset_class(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_name = args.output_name
    model = eval(model_name)()
    print(model)
    model.to(args.device)

    params = get_model_params(model)
    print(f'{params=}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        num_epochs=args.max_epoch,
        min_lr=1e-5,
        cooldown_epochs=10,
        warmup_epochs=10,
        warmup_lr=1e-5,
    )
    amp_autocast = torch.amp.autocast(args.device)
    if args.saver:
        saver = CheckpointSaver(
            model=model, 
            optimizer=optimizer, 
            args=args, 
            model_ema=None, 
            amp_scaler=None,
            checkpoint_dir=output_dir, 
            recovery_dir=output_dir, 
            decreasing=False, 
            max_history=5
        )
    else:
        saver = None

    # train
    # print('\n\n[Training]')
    pbar = tqdm(range(num_epochs), desc=f'{model_name}')
    for epoch in pbar:
        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        # print(f'\n[Epoch {epoch + 1} / {num_epochs}] {lr=:.07f}')
        train_loss = train(
            epoch=epoch,
            train_loader=train_loader, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer,
            scheduler=scheduler,
            amp_autocast=amp_autocast,
            device=args.device,
        )
        # print(f'[Epoch {epoch + 1} / {num_epochs}] train_loss={train_epoch_loss}')

        val_acc = test(
            test_loader=test_loader, 
            model=model, 
            device=args.device,
        )
        # print(f'[Epoch {epoch + 1} / {num_epochs}] val_acc={val_epoch_acc:.4f}')

        if scheduler is not None:
            # step LR for next epoch
            scheduler.step(epoch + 1, val_acc)
        
        if saver:
            best_acc, best_epoch = saver.save_checkpoint(epoch, metric=val_acc)
            pbar.set_postfix(loss=train_loss, acc=val_acc, lr=lr, best_acc=best_acc, best_epoch=best_epoch)
        else:
            pbar.set_postfix(loss=train_loss, acc=val_acc, lr=lr)
        
    # save
    model.eval()
    model_save_file = os.path.join(output_dir, f'{args.output_name}.pt')
    torch.save(model.state_dict(), model_save_file)
    print('Accuracy:', val_acc)

if __name__ == '__main__':
    main()

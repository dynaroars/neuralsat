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

from timm.utils import NativeScaler, AverageMeter, random_seed, CheckpointSaver
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import create_dataset, create_loader, FastCollateMixup
from timm.models import model_parameters, create_model
from timm.scheduler import create_scheduler_v2

from models.resnet.resnet import *
from models.vit.vit import *

loss_fn_p = torch.nn.CrossEntropyLoss()

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    # Set requires_grad attribute of the images tensor
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to calculate gradients
    loss.backward()
    
    # Collect the sign of the gradients
    sign_data_grad = images.grad.data.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * sign_data_grad
    
    # Return the perturbed image
    return perturbed_image


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def train(epoch, train_loader, model, criterion, optimizer, loss_scaler, scheduler, amp_autocast=suppress(), device='cpu', adv_train=False):
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
            Y_rec = model(X)
            loss = criterion(Y_rec, y)

        losses_m.update(loss.item(), X.size(0))

        # clean
        optimizer.zero_grad()
        
        if loss_scaler is not None:
            loss_scaler(
                loss, 
                optimizer,
                clip_grad=0.1, 
                clip_mode='norm',
                parameters=model_parameters(model, exclude_head='agc' in 'norm'),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            optimizer.step()


        # # adv
        # if adv_train:
        #     optimizer.zero_grad()
        #     perturbed_images = fgsm_attack(model, loss_fn_p, X, y, epsilon=0.01)
        #     Y_perturb = model(perturbed_images)
        #     loss_p = loss_fn_p(Y_perturb, y)
        #     loss_p.backward()
        #     optimizer.step()
        
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
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--saver', action='store_true')

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args


def main():
    args = parse_args()
    
    random_seed(args.seed)
    
    output_dir = f'{args.save_dir}/{args.output_folder}/{args.output_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset.endswith('mnist'):
        input_shape = (1, 1, 28, 28)
        dataset_class = torchvision.datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset.endswith('cifar10'):
        input_shape = (1, 3, 32, 32)
        dataset_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    else:
        raise ValueError(args.dataset)
    
    num_classes = 10
    
    # train_set = dataset_class(
    #     root=args.data_root,
    #     train=True,
    #     transform=transform,
    #     download=True)
    
    # test_set = dataset_class(
    #     root=args.data_root,
    #     train=False,
    #     transform=transform,
    #     download=True)

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    mixup_args = dict(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=num_classes
    )
    collate_fn = FastCollateMixup(**mixup_args)
    
    
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_root,
        split='train', 
        is_training=True,
        batch_size=args.batch_size, 
        repeats=0,
        download=True,
    )
    
    dataset_eval = create_dataset(
        args.dataset, 
        root=args.data_root,
        split='eval', 
        is_training=False, 
        batch_size=args.batch_size,
        download=True,
    )

    print(dataset_train)
    print(dataset_eval)
    
    train_loader = create_loader(
        dataset_train,
        input_size=input_shape[1:],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        re_split=False,
        scale=(0.8, 1.0),
        ratio=(3./4., 4./3.),
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        num_aug_splits=0,
        interpolation='random',
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        num_workers=8,
        distributed=False,
        collate_fn=collate_fn,
        pin_memory=False,
        use_multi_epochs_loader=False,
    )
    
    test_loader = create_loader(
        dataset_eval,
        input_size=input_shape[1:],
        batch_size=1 * args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation='bicubic',
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        num_workers=8,
        distributed=False,
        crop_pct=1.0,
        pin_memory=False,
    )
    
    
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # if args.model == 'vit':
        # from models.vit import get_model
        # weights = args.infer
        # model = get_model(
        #     input_shape=input_shape,
        #     depth=4, 
        #     num_heads=4, 
        #     patch_size=14, 
        #     embed_dim=256, 
        #     weights=weights,
        # )
        # if weights:
        #     model.to(args.device)
        #     val_epoch_acc = test(test_loader, model, args.device)
        #     print(f'val_acc={val_epoch_acc:.4f}')
        #     exit()
        
    model_name = args.output_name
    
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_connect_rate=None,
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path=None)
    
    model.to(args.device)
    print(model)

    params = get_model_params(model)
    print(f'{params=}')
    
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(args.device)
    criterion = SoftTargetCrossEntropy().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5 / 5e-4, total_iters=10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10])
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        num_epochs=args.max_epoch,
        min_lr=1e-5,
        cooldown_epochs=10,
        warmup_epochs=10,
        warmup_lr=1e-5,
    )
    loss_scaler = NativeScaler()
    amp_autocast = torch.amp.autocast(args.device)
    # print(loss_scaler)
    # exit()
    if args.saver:
        saver = CheckpointSaver(
            model=model, 
            optimizer=optimizer, 
            args=args, 
            model_ema=None, 
            amp_scaler=loss_scaler,
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
            loss_scaler=loss_scaler, 
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
    # torch.onnx.export(
    #     model.cpu(),
    #     torch.zeros(input_shape),
    #     os.path.join(args.save_dir, f'{args.output_name}.onnx'),
    #     verbose=False,
    #     opset_version=12,
    # )
    print('Accuracy:', val_acc)

if __name__ == '__main__':
    main()

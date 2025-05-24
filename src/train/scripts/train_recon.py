from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import torchvision
import argparse
import random
import torch
import yaml
import os

from models.vae.discriminator import Discriminator
from models.vae.vae_naive import get_model
from models.vae.lpips import LPIPS
from models.vae.vae import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    print(f'{dataset_config=}\n')
    print(f'{autoencoder_config=}\n')
    print(f'{train_config=}\n')
    
    model_ckpt_name = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    model_ckpt = model_ckpt_name + '.pth'
    
    disc_ckpt_name = os.path.join(train_config['task_name'], train_config['vae_discriminator_ckpt_name'])
    disc_ckpt = disc_ckpt_name + '.pth'
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VAE(
        dataset_config=config['dataset_params'],
        model_config=config['autoencoder_params'],
    ).to(device)
    print(model)
    
    if os.path.exists(model_ckpt):
        model.load_state_dict(torch.load(model_ckpt))
    
    params = get_model_params(model)
    print(f'{params=}')
    
    # print(model)
    # shape = (1, 3, 224, 224)
    # model = get_model(shape).to(device)

    # x = torch.randn(shape)
    # y = model(x)
    # print(f'{x.shape=} {y.shape=}')
    
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    # ])
    
    
    if args.dataset == 'mnist':
        dataset_class = torchvision.datasets.MNIST
        dataset_args = {'train': True, 'download': True}
    
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
    elif args.dataset == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
        dataset_args = {'train': True, 'download': True}
    
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
        
    elif args.dataset == 'imagenet':
        dataset_class = torchvision.datasets.ImageNet
        dataset_args = {'split': 'val'}
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # Resize the shortest side to 256 pixels
            torchvision.transforms.CenterCrop(dataset_config['im_size']),  # Center crop the image to 224x224 pixels
            torchvision.transforms.ToTensor()  # Convert image to a PyTorch tensor
        ])
        
    else:
        raise ValueError(args.dataset)
    
    im_dataset = dataset_class(
        root='data',
        transform=transform,
        **dataset_args
    )
    
    
    data_loader = DataLoader(
        im_dataset,
        batch_size=train_config['autoencoder_batch_size'],
        shuffle=True,
    )

    # Create output directories
    os.makedirs(train_config['task_name'], exist_ok=True)
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    if os.path.exists(disc_ckpt):
        discriminator.load_state_dict(torch.load(disc_ckpt))
        
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    pbar = tqdm(range(num_epochs), desc=f'VAE {params=} ({os.path.basename(args.config_path)})')
    for epoch_idx in pbar:
        recon_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for (im, _) in data_loader:
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            output = model(im)
            # print(im.shape, output.shape)
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                # print(f'{output.shape=} {save_input.shape=} {save_output.shape=}')
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                os.makedirs(os.path.join(train_config['task_name'], 'vae_autoencoder_samples'), exist_ok=True)
                # img.save(os.path.join(train_config['task_name'],'vae_autoencoder_samples', f'current_autoencoder_sample_{img_save_count}.png'))
                img.save(os.path.join(train_config['task_name'],'vae_autoencoder_samples', f'current_autoencoder_sample.png'))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = recon_loss 

            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device))
                assert not torch.isnan(disc_fake_loss)
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            # print(
            #     f'Epoch : {epoch_idx + 1}/{num_epochs} | Recon Loss : {np.mean(recon_losses):.4f} | Perceptual Loss : {np.mean(perceptual_losses):.4f} | '
            #     f'G Loss : {np.mean(gen_losses):.4f} | D Loss {np.mean(disc_losses):.4f}'
            # )
            pbar.set_postfix(recon=np.mean(recon_losses), perceptual=np.mean(perceptual_losses), gen=np.mean(gen_losses), disc=np.mean(disc_losses))
        else:
            pbar.set_postfix(recon=np.mean(recon_losses), perceptual=np.mean(perceptual_losses))
            # print(f'Epoch: {epoch_idx + 1}/{num_epochs} | Recon Loss : {np.mean(recon_losses):.4f} | Perceptual Loss : {np.mean(perceptual_losses):.4f}')
        
        if not epoch_idx % 100:
            # model.eval()
            torch.save(model.state_dict(), model_ckpt)
            torch.save(model.state_dict(), model_ckpt_name + f'_epoch_{epoch_idx}.pth')
            
            torch.save(discriminator.state_dict(), disc_ckpt)
            
            
            # model.train()
            
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='config/mnist.yaml', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    args = parser.parse_args()
    train(args)

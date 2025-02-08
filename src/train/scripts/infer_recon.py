import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from models.vae.vae import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    config = yaml.safe_load(open(args.config))
    print(config)
    
    train_config = config['train_params']
    model_ckpt = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])

    dataset_class = torchvision.datasets.CIFAR10
    dataset_args = {'train': True, 'download': True}

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    im_dataset = dataset_class(
        root='data',
        transform=transform,
        **dataset_args
    )
    
    num_images = 16
    ngrid = 4 
    
    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][0][None] for idx in idxs]).float()
    ims = ims.to(device)
    
    model = VAE(
        dataset_config=config['dataset_params'],
        model_config=config['autoencoder_params'],
    ).to(device)
    model.eval()
    print(f'Loading {model_ckpt=}')
    model.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=True))
    
    with torch.no_grad():
        # encoded_output = model.encode(ims)
        recon_output = model(ims)
        recon_output = torch.clamp(recon_output, -1., 1.)
        recon_output = (recon_output + 1) / 2
        ims = (ims + 1) / 2

        recon_grid = make_grid(recon_output.cpu(), nrow=ngrid)
        input_grid = make_grid(ims.cpu(), nrow=ngrid)
        recon_grid = torchvision.transforms.ToPILImage()(recon_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        recon_grid.save(os.path.join(train_config['task_name'], 'recon_samples.png'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VAE inference')
    parser.add_argument('--config', dest='config', required=True, type=str)
    args = parser.parse_args()
    infer(args)

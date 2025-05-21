import torch.nn as nn
import torch

def build_layers(layer_configs, transpose=False):
    layers = []
    for config in layer_configs:
        if transpose:
            layers.append(nn.ConvTranspose2d(**config))
        else:
            layers.append(nn.Conv2d(**config))
        layers.append(nn.ReLU(inplace=True))
    return layers


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params = }')
    return total_params


class Unflatten(nn.Module):
    
    def __init__(self, start_dim, shape):
        super(Unflatten, self).__init__()
        self.shape = tuple(shape)
        self.start_dim = start_dim
        
    def forward(self, x):
        assert self.start_dim == 1
        return x.view(x.size(0), *self.shape)
    
    def __repr__(self):
        return f'Unflatten(start_dim={self.start_dim}, shape={self.shape})'

        
class CustomVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder_configs, decoder_configs):
        super(CustomVAE, self).__init__()
        
        encoder_layers = build_layers(encoder_configs, transpose=False)
        decoder_layers = build_layers(decoder_configs, transpose=True)[:-1]

        with torch.no_grad():
            dummy = torch.randn(input_shape)
            enc_output = nn.Sequential(*encoder_layers)(dummy)
            enc_flattened_size = enc_output.flatten(1).size(1)
            enc_unflattened_size = enc_output.size()[1:]

        encoder_layers = encoder_layers + [nn.Flatten(1), nn.Linear(enc_flattened_size, latent_dim)]
        decoder_layers = [nn.Linear(latent_dim, enc_flattened_size), Unflatten(1, enc_unflattened_size), nn.ReLU(inplace=True)] + decoder_layers + [nn.Flatten(1)]

        # self.encoder = nn.Sequential(*encoder_layers)
        # self.decoder = nn.Sequential(*decoder_layers)
        
        self.layers = nn.ModuleList([
            nn.Sequential(*encoder_layers),
            nn.Sequential(*decoder_layers),
        ])
        
        
    # def forward(self, x):
    #     z = self.encoder(x)
    #     recon_x = self.decoder(z)
    #     return recon_x.flatten(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def get_model(input_shape, latent_dim=64):
    
    in_channels = input_shape[1]
    enc_configs = [
        {'in_channels': in_channels, 'out_channels': 32, 'kernel_size': 3, 'stride': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1},
        # {'in_channels': 128, 'out_channels': 256, 'kernel_size': 2, 'stride': 1},
    ]
    
    dec_configs = [
        # {'in_channels': 256, 'out_channels': 128, 'kernel_size': 2, 'stride': 1},
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 1},
        {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3, 'stride': 1},
        {'in_channels': 32, 'out_channels': in_channels, 'kernel_size': 3, 'stride': 1},
    ]

    vae = CustomVAE(
        input_shape=input_shape,
        encoder_configs=enc_configs, 
        decoder_configs=dec_configs,
        latent_dim=latent_dim, 
    )
    
    get_model_params(vae)
    
    return vae
    
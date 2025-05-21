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

        
class VAEClassifier(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder_configs, decoder_configs):
        super(VAEClassifier, self).__init__()
        
        encoder_layers = build_layers(encoder_configs, transpose=False)
        decoder_layers = build_layers(decoder_configs, transpose=True)

        with torch.no_grad():
            dummy = torch.randn(input_shape)
            enc_output = nn.Sequential(*encoder_layers)(dummy)
            enc_flattened_size = enc_output.flatten(1).size(1)
            enc_unflattened_size = enc_output.size()[1:]

        encoder_layers = encoder_layers + [nn.Flatten(1), nn.Linear(enc_flattened_size, latent_dim)]
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = [nn.Linear(latent_dim, enc_flattened_size), Unflatten(1, enc_unflattened_size), nn.ReLU(inplace=True)] + decoder_layers
        self.decoder = nn.Sequential(*decoder_layers)
        
        with torch.no_grad():
            dec_output = self.decoder(self.encoder(dummy))
            dec_flattened_size = dec_output.flatten(1).size(1)
            
        classifier = nn.Sequential(*[
            nn.Flatten(1),
            nn.ReLU(),
            nn.Linear(dec_flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ])
        
        decoder_layers += classifier
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x


def get_vae_model(input_shape, latent_dim):
    
    enc_configs = [
        {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        # {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 0},
    ]
    
    dec_configs = [
        {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 0},
        # {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 64, 'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 0},
    ]

    vae = VAEClassifier(
        input_shape=input_shape,
        encoder_configs=enc_configs, 
        decoder_configs=dec_configs,
        latent_dim=latent_dim, 
    )
    
    return vae
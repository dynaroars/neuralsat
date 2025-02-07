import torch
import torch.nn as nn

    
class VAEOld(nn.Module):
    
    def __init__(self, dataset_config, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.resample = model_config['resample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1], f'{self.mid_channels[0]=} {self.down_channels[-1]=}'
        assert self.mid_channels[-1] == self.down_channels[-1]
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(dataset_config['im_channels'], self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(in_channels=self.down_channels[i],
                                                 out_channels=self.down_channels[i + 1],
                                                 down_sample=self.resample,
                                                 num_layers=self.num_down_layers))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(in_channels=self.mid_channels[i], 
                                              out_channels=self.mid_channels[i + 1],
                                              num_layers=self.num_mid_layers))
        
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, 2*self.z_channels, kernel_size=1)
        ####################################################
        
        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(in_channels=self.mid_channels[i], 
                                              out_channels=self.mid_channels[i - 1],
                                              num_layers=self.num_mid_layers))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(in_channels=self.down_channels[i], 
                                               out_channels=self.down_channels[i - 1],
                                               up_sample=self.resample,
                                               num_layers=self.num_up_layers))
        
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], dataset_config['im_channels'], kernel_size=3, padding=1)
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        # out = self.encoder_norm_out(out)
        # out = nn.SiLU()(out)
        out = out.relu()
        out = self.encoder_conv_out(out)
        # print(out.shape)
        out = self.pre_quant_conv(out)
        # return out
        mean, logvar = torch.chunk(out, 2, dim=1)
        return mean + logvar
        # return mean, out
        std = torch.exp(0.5 * logvar)
        sample = mean + std * 0.1 # torch.randn_like(mean).to(x)
        return sample, out
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_layers:
            out = up(out)
        out = out.relu()
        out = self.decoder_conv_out(out)
        return out

    # def forward(self, x):
    #     z, encoder_output = self.encode(x)
    #     out = self.decode(z)
    #     return out, encoder_output


    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out



class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """
    
    def __init__(self, in_channels, out_channels, down_sample, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, 
                              out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()
        
    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else out_channels, 
                              out_channels, 
                              kernel_size=3, 
                              stride=1,
                              padding=1),
                    nn.ReLU(),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
    
    def forward(self, x):
        out = x
 
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
        
        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, up_sample, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) if self.up_sample else nn.Identity()
        
    def forward(self, x):
        # Upsample
        x = self.up_sample_conv(x)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
        return out


import torch

class ReshapeConv(torch.nn.Module):
    def __init__(self, in_img_dim, out_img_dim, in_channels, out_channels, layer):
        super(ReshapeConv, self).__init__()
        self.in_img_dim = in_img_dim
        self.out_img_dim = out_img_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = layer

    def forward(self, x):
        out = self.layer(x.view(1, self.in_channels, self.in_img_dim, self.in_img_dim))
        return torch.flatten(out)
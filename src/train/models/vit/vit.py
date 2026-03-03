import torch.nn.functional as F
from torch import Tensor
from torch import nn
import torch
import os


try:
    from ..models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        """
        Fallback wrapper in case timm isn't installed
        """
        return func
    
class ReduceMean(nn.Module):
    
    def forward(self, x):
        return x.mean(dim=1)


class BatchNorm(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


class PatchEmbedding(nn.Module):
    
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(emb_size))
        self.emb_size = emb_size
        
    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        x = self.projection(x)
        cls_tokens = torch.zeros(batch, 1, self.emb_size, device=x.device) + self.cls_token
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model = 123, num_heads = 6, dropout = 0., expansion = 1, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0, f"{d_model=} must be divisible by {num_heads=}"

        self.d_model = d_model 
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(d_model, d_model * expansion)
        self.key = nn.Linear(d_model, d_model * expansion)
        self.value = nn.Linear(d_model, d_model * expansion)
        self.out = nn.Linear(d_model * expansion, d_model)
        self.expansion = expansion

        # self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.scale

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model * self.expansion)
        attn_output = self.out(attn_output)

        return attn_output


class ResidualAdd(nn.Module):
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        # print(f'{x.shape=} {x.shape[1] * x.shape[2]}')
        return x


class FeedForwardBlock(nn.Sequential):
    
    def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            # nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 3,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                # BatchNorm(emb_size),
                MultiHeadAttention(emb_size, expansion=forward_expansion, **kwargs),
                # nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                # BatchNorm(emb_size),
                FeedForwardBlock(emb_size=emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                # nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    
    def __init__(self, depth: int = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            ReduceMean(),
            # nn.BatchNorm1d(emb_size),
            nn.Linear(emb_size, n_classes),
        )



class ViTLite(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 num_heads: int = 1,
                 patch_size: int = 2,
                 emb_size: int = 4,
                 depth: int = 1,
                 n_classes: int = 2,
                 forward_expansion: int = 1,
                 *args, 
                 **kwargs):

        super().__init__()
        assert depth >= 2
        first_block = nn.Sequential(
            PatchEmbedding(in_channels, patch_size, emb_size), 
            TransformerEncoder(emb_size=emb_size, num_heads=num_heads, forward_expansion=forward_expansion, **kwargs))
        last_block = nn.Sequential(
            TransformerEncoder(emb_size=emb_size, num_heads=num_heads, forward_expansion=forward_expansion, **kwargs), 
            ClassificationHead(emb_size, n_classes))
        middle_blocks = [TransformerEncoder(emb_size=emb_size, num_heads=num_heads, forward_expansion=forward_expansion, **kwargs) for _ in range(depth-2)]
        
        layers = [first_block, *middle_blocks, last_block]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            # print(f'{x.shape=}')
            x = layer(x)
        return x

@register_model
def vit_3_32(*args, **kwargs):
    return ViTLite(
        in_channels=3,
        depth=3,
        emb_size=32,
        patch_size=8,
        num_heads=2,
        n_classes=10,
        forward_expansion=2,
        *args, 
        **kwargs
    )
    
    
if __name__ == "__main__":
    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params = }')
        return total_params

    # model = vit_toy(in_channels=1, depth=3)
    model = vit_3_32()
    x = torch.randn(2, 3, 32, 32)
    print(model)
    
    get_model_params(model)
    
    y = model(x)
    print(x.shape, y.shape)
    exit()
    
    model.eval()
    output_name = f'weights/vit_toy_new.onnx'    
    print('[+] Exporting ONNX')
    torch.onnx.export(
        model,
        x,
        output_name,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    
    os.system(f'onnxsim "{output_name}" "{output_name}"')
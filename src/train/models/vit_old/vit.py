import torch.nn as nn
import torch
import os

from .vit_utils import TransformerEmbedder, TransformerEncoderLayer, TransformerClassifier
from .vit_utils import Tokenizer

try:
    from ..models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        """
        Fallback wrapper in case timm isn't installed
        """
        return func
                
class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 num_layers=14,
                 num_heads=6,
                 head_dim=16,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 activation='relu',
                 *args, **kwargs):
        super(ViTLite, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be divisible by patch size ({kernel_size})"
        
        tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            n_conv_layers=1,
            conv_bias=True,
        )
        seq_len = tokenizer.sequence_length(n_channels=n_input_channels, height=img_size, width=img_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                tokenizer, 
                TransformerEmbedder(
                    sequence_length=seq_len,
                    embedding_dim=embedding_dim, 
                    positional_embedding=positional_embedding, 
                ),
                TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=num_heads,
                    head_dim=head_dim,
                    dim_feedforward=int(embedding_dim * mlp_ratio), 
                    dropout=dropout,
                    attention_dropout=attention_dropout, 
                    activation=activation,
                )
            ),
            *[
                TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=num_heads,
                    head_dim=head_dim,
                    dim_feedforward=int(embedding_dim * mlp_ratio), 
                    dropout=dropout,
                    attention_dropout=attention_dropout, 
                    activation=activation,
                ) 
                for _ in range(num_layers - 2)
            ],
            nn.Sequential(
                TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=num_heads,
                    head_dim=head_dim,
                    dim_feedforward=int(embedding_dim * mlp_ratio), 
                    dropout=dropout,
                    attention_dropout=attention_dropout, 
                    activation=activation,
                ),
                TransformerClassifier(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes
                )
            )
            
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def _vit_lite_relu(num_layers, num_heads, head_dim, mlp_ratio, embedding_dim,
              positional_embedding='learnable', activation='relu',
              kernel_size=4, *args, **kwargs):
    model = ViTLite(num_layers=num_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    positional_embedding=positional_embedding,
                    activation=activation,
                    *args, **kwargs)

    return model


@register_model
def vit_toy(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=2, 
        kernel_size=2,
        embedding_dim=4, 
        num_heads=1,
        head_dim=4,
        mlp_ratio=1, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )


@register_model
def vit_medium(img_size=32, positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=2, 
        kernel_size=16,
        embedding_dim=48, 
        num_heads=3,
        head_dim=16,
        mlp_ratio=1, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )   
    
    
if __name__ == "__main__":
    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params = }')
        return total_params

    # model = vit_toy(n_input_channels=1)
    model = vit_medium(n_input_channels=1)
    x = torch.randn(2, 1, 32, 32)
    print(model)
    
    get_model_params(model)
    
    y = model(x)
    print(y.shape)
    
    model.eval()
    output_name = f'weights/vit_toy_old.onnx'    
    
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
    print(f'[+] Exporting ONNX: {output_name=}')

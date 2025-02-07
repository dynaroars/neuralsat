import torch.nn as nn
import torch

from .vit_utils import TransformerEmbedder, TransformerEncoderLayer, TransformerClassifier
from .vit_utils import Tokenizer, TransformerClassifierOld

try:
    from timm.models.registry import register_model
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
        if num_layers == 1:
            classifier = TransformerClassifierOld(
                sequence_length=seq_len,
                embedding_dim=embedding_dim,
                seq_pool=False,
                dropout=dropout,
                attention_dropout=attention_dropout,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_classes=num_classes,
                positional_embedding=positional_embedding,
                activation=activation,
            )
            
            self.layers = nn.ModuleList([nn.Sequential(tokenizer, classifier)])
        else:
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
                    for ii in range(num_layers - 2)
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


def _vit_lite_sigmoid(num_layers, num_heads, mlp_ratio, embedding_dim,
              positional_embedding='learnable', activation='sigmoid',
              kernel_size=4, *args, **kwargs):
    model = ViTLite(num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    positional_embedding=positional_embedding,
                    activation=activation,
                    *args, **kwargs)

    return model


@register_model
def vit_10_8_128_3_32_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=10, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=32,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )
    
 
@register_model
def vit_12_8_128_3_32_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=12, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=32,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )
    
 
@register_model
def vit_12_8_128_2_32_1(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=12, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=2,
        head_dim=32,
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
def vit_8_8_128_3_32_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=8, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=32,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )
    
    
@register_model
def vit_6_8_128_3_32_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=6, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=32,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )
    
@register_model
def vit_10_8_128_3_16_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=10, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=16,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )   
    
    
    
@register_model
def vit_12_8_128_3_16_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=12, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=16,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )   
    
   
@register_model
def vit_12_8_128_2_16_1(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=12, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=2,
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
     
    
@register_model
def vit_8_8_128_3_16_2(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=8, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=16,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )   
    
@register_model
def vit_toy(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=5, 
        kernel_size=8,
        embedding_dim=128, 
        num_heads=3,
        head_dim=16,
        mlp_ratio=2, 
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )


@register_model
def vit_4_16_32_2_8_1(img_size=32, positional_embedding='none', num_classes=10, *args, **kwargs):
    return _vit_lite_relu(
        num_layers=4, 
        kernel_size=16,
        embedding_dim=32, 
        num_heads=2,
        head_dim=8,
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

    model = vit_toy(n_input_channels=1)
    x = torch.randn(2, 1, 32, 32)
    print(model)
    
    get_model_params(model)
    
    y = model(x)
    print(y.shape)
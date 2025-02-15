import torch.nn.functional as F
import torch.nn as nn
import torch


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def subtraction_gaussian_kernel_torch(q, k):
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:], device=q.device)
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:], device=q.device) @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)

def stable_softmax(x, dim):
    # Subtract the max value from each element to prevent overflow
    x_max = torch.max(x, dim=dim, keepdim=True)[0]  # Compute max per row if x is a matrix
    exp_x = torch.exp(x - x_max)  # Subtract max and compute exp
    softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)  # Normalize to get probabilities
    return softmax_x

    
class Tokenizer(nn.Module):
    def __init__(self, kernel_size, stride, padding,
                 n_conv_layers=1, n_input_channels=3, n_output_channels=64,
                 in_planes=64, conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + [in_planes for _ in range(n_conv_layers - 1)] + [n_output_channels]
        self.conv_layers = nn.Sequential(*[
            nn.Conv2d(
                n_filter_list[i], 
                n_filter_list[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, 
                bias=conv_bias,
            )
            for i in range(n_conv_layers)
        ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.conv_layers(x).flatten(2).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            

class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, head_dim=32, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // self.num_heads
        inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.attn_drop = nn.Dropout(attention_dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Linear(inner_dim, dim)
        # self.proj_drop = nn.Dropout(projection_dropout)

        # self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(dim, inner_dim, bias=False)
        # self.attn_act = LearnableSigmoid(17) #nn.Sigmoid()

    def forward(self, x):
        B, N, _ = x.shape
        H = self.num_heads
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q: B, N, H * D -> B H N D
        q = self.to_q(x).view(B, N, H, -1).permute(0, 2, 1, 3) 
        k = self.to_k(x).view(B, N, H, -1).permute(0, 2, 1, 3)
        v = self.to_v(x).view(B, N, H, -1).permute(0, 2, 1, 3)
        
        # print(f'{q.shape=}')
        # q, k, v = map(lambda t: rearrange(t, 'B N (H D) -> B H N D', H=H), [q, k, v])
        # print(f'{q.shape=}')
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(1, attn.shape)
        # attn = attn.softmax(dim=-1)
        attn = stable_softmax(attn, dim=-1)
        
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = self.attn_act(attn)
        
        # attn = subtraction_gaussian_kernel_torch(q, k.transpose(-2, -1)) * self.scale
        # attn = torch.exp(-attn / 2)
        # attn = self.attn_drop(attn)

        x = (attn @ v)
        x = rearrange(x, 'B H N D -> B N (H D)')
        
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class LayerNormNoVar(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNormNoVar, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = x - u
        return self.weight * x + self.bias



class LearnableSigmoid(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LearnableSigmoid, self).__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        # print(f'{x.shape=} {self.bias.shape=}')
        # exit()
        x = 1 + torch.exp(-(x + self.bias))
        return 1/x



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, head_dim=32, dim_feedforward=2048, dropout=0.1, attention_dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNormNoVar(d_model)
        # self.pre_norm = nn.BatchNorm1d(d_model)
        self.self_attn = Attention(dim=d_model, 
                                   num_heads=nhead,
                                   head_dim=head_dim,
                                   attention_dropout=attention_dropout, 
                                   projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNormNoVar(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.dropout2 = nn.Dropout(dropout)

        assert activation in ['relu', 'sigmoid', 'tanh'], f'Unsupport {activation=}'
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.self_attn(self.pre_norm(src))
        # src = src + self.self_attn(src)
        src = self.norm1(src)
        src = src + self.linear2(self.activation(self.linear1(src)))
        return src
    
class TransformerClassifierOld(nn.Module):
    def __init__(self,
                 seq_pool=False,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 positional_embedding='sine',
                 activation='relu',
                 sequence_length=None,
                 n_channels=3):
        super().__init__()
        positional_embedding = positional_embedding if positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.n_channels = n_channels

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim), requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False)
        else:
            self.positional_emb = None

        # self.dropout = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads,
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                attention_dropout=attention_dropout, 
                activation=activation,
            )
            for i in range(num_layers)
        ])
        self.norm = LayerNormNoVar(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm, LayerNormNoVar)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.1)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        print('cac')
        pe = torch.FloatTensor([
            [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] 
            for p in range(n_channels)
        ])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class TransformerEmbedder(nn.Module):

    def __init__(self, embedding_dim=768, positional_embedding='sine', sequence_length=None):
        super().__init__()
        
        assert positional_embedding in ['sine', 'learnable', 'none']
        positional_embedding = positional_embedding
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length+1, embedding_dim), requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length+1, embedding_dim), requires_grad=False)
        else:
            self.positional_emb = None

        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
            raise

        cls_token = self.class_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            raise
            x += self.positional_emb

        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNormNoVar, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.1)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([
            [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] 
            for p in range(n_channels)
        ])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


   
class TransformerClassifier(nn.Module):

    def __init__(self, embedding_dim=768, num_classes=1000):
        super().__init__()
        
        self.norm = LayerNormNoVar(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.norm(x)
        x = x[:, 0]
        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm, LayerNormNoVar)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.1)


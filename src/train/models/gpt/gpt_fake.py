from torch.nn import functional as F
import torch.nn as nn
import torch
import math

LayerNorm = nn.LayerNorm

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config, **kwargs):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"{config.n_embd=} must be divisible by {config.n_head=}"

        self.n_embd = config.n_embd 
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.out = nn.Linear(config.n_embd, config.n_embd)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_head, self.head_dim)
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

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd)
        attn_output = self.out(attn_output)

        return attn_output

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = nn.ReLU(),
            # dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):

    """ GPT Language Model """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Linear(1, config.n_embd),
            # drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, x):
        # device = idx.device
        # b, t = idx.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        # print('idx shape:', idx.shape)
        # tok_emb = self.transformer.wte(idx).unsqueeze(1) # token embeddings of shape (b, t, n_embd)
        # print('tok_emb shape:', tok_emb.shape)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        # x = tok_emb
        # print('embedding shape:', x.shape)
        for block in self.transformer.h:
            x = block(x)
            # print('block output shape:', x.shape)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # print('logits shape:', logits.shape)
        # if we are given some desired targets also calculate the loss
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits

if __name__ == '__main__':
    # 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    # 'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    # 'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    # 'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    
    class Config:
        pass
    
    config = Config()
    config.vocab_size = 10
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    
    model = GPT(config)
    
    x = torch.randint(0, config.vocab_size, (1, 1)).float()
    print(x.shape)
    logits, loss = model(x)
    print(logits.shape, loss)
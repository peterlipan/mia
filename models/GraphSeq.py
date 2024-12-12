import torch
import torch.nn as nn
from einops import rearrange


class LineaEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super(LineaEmbedding, self).__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class Transformer(nn.Module):
    def __init__(self, n_regions, embed_dim, n_classes, max_len=1000, n_heads=8, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.embedding = LineaEmbedding(n_regions, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.attention1 = nn.MultiheadAttention(embed_dim, n_heads, drop_rate, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim, n_heads, drop_rate, batch_first=True)
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.contrast_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, 64, bias=False)
        )

        # self.num_attn = nn.Parameter(torch.ones(1, num_dim) / num_dim, requires_grad=True)
        # self.str_attn = nn.Parameter(torch.ones(1, str_num) / str_num, requires_grad=True)
        # self.str_oh_exp = torch.tensor(str_oh_exp, requires_grad=False)
        # self.imputation = nn.Parameter(torch.zeros(1, phe_dim), requires_grad=True)
    

    def forward(self, x):
        # x: [B, T, V, N]
        # num_fea: [B, n_num]; str_fea: [B, n_str]
        B, T, V, N = x.shape
        # [B, T, V, N] -> [B V, T, N]
        x = rearrange(x, 'b t v n -> (b v) t n')
        x = self.embedding(x) # [B V, T, C]
        x = x + self.positional_encoding[:, :T]
        x = self.attention1(x, x, x)[0] # [B V, T, C]
        x = self.norm(x)
        x = self.attention2(x, x, x)[0]
        x = self.norm(self.pool(x.transpose(1, 2)).squeeze(-1)) # [B V, C]

        logits = self.classifier(x) # [B V, n_cls]
        con_fea = self.contrast_head(x)
        con_fea = rearrange(con_fea, '(b v) c -> b v c', b=B, v=V)
    
        return logits, con_fea

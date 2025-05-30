import math
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from einops import rearrange
from .utils import ModelOutputs
from torch.utils.checkpoint import checkpoint
from .Sparsemax import Sparsemax


class LineaEmbedding(nn.Module):
    def __init__(self, n_regions, embed_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(n_regions, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.BatchNorm1d(n_regions)

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out)
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class MultiheadChannelAttention(nn.Module):
    def __init__(self, d_model, n_head=1, d_x=64, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.d_x = d_x
        self.n_head = n_head
        self.softmax = Sparsemax(dim=-1)
        
        # Use a single projection matrix for Q, K, V
        self.w_qkv = nn.Linear(d_model, n_head * d_x * 3, bias=bias)
        self.fc = nn.Linear(n_head * d_x, d_model, bias=bias)

    
    def ScaledDotProductChannelAttention(self, query, key, value):
        dx = query.size()[-1]
        scores = query.transpose(-2, -1).matmul(key) / math.sqrt(dx)
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        return value.matmul(attention), attention

    def forward(self, x):
        
        d_x, n_head = self.d_x, self.n_head
        sz_b, len_q, _ = x.size()

        residual = x
        
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(sz_b, len_q, n_head, d_x)
        k = k.view(sz_b, len_q, n_head, d_x)
        v = v.view(sz_b, len_q, n_head, d_x)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # b x n x lq x dx
        q, attn = self.ScaledDotProductChannelAttention(q, k, v) # b x n x lq x dx

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dx)
        q= self.dropout(self.fc(q))

        q += residual
        q = self.norm(q)
        
        return q, attn


class MultiheadTokenAttention(nn.Module):
    def __init__(self, d_model, n_head=1, d_x=64, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.d_x = d_x
        self.n_head = n_head
        
        # Use a single projection matrix for Q, K, V
        self.w_q = nn.Linear(d_model, n_head * d_x, bias=bias)
        self.w_k = nn.Linear(d_model, n_head * d_x, bias=bias)
        self.w_v = nn.Linear(d_model, n_head * d_x, bias=bias)
        self.fc = nn.Linear(n_head * d_x, d_model, bias=bias)
    
    def ScaledDotProductTokenAttention(self, query, key, value, attn_mask=None):
        """
        Computes scaled dot-product attention for tokens.
        """
        d_x = query.size(-1)  # Dimension of each head
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_x)  # (batch, n_head, seq_len_q, seq_len_k)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))  # Apply attention mask
        
        attention = F.softmax(scores, dim=-1)  # Normalize over the last dimension (seq_len_k)
        attention = self.dropout(attention)  # Apply dropout to attention weights
        
        return attention.matmul(value), attention  # (batch, n_head, seq_len_q, d_x)

    def forward(self, query, key=None, value=None, attn_mask=None):
        """
        Forward pass for token attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model) (optional, defaults to `query` for self-attention)
            value: Value tensor of shape (batch_size, seq_len_k, d_model) (optional, defaults to `query` for self-attention)
            attn_mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k)
        
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        d_x, n_head = self.d_x, self.n_head
        sz_b, len_q, _ = query.size()  # Batch size, query sequence length, embedding dimension

        # If key and value are not provided, use self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        # Linear projections for Q, K, V
        q = self.w_q(query).view(sz_b, len_q, n_head, d_x)
        k = self.w_k(key).view(sz_b, key.size(1), n_head, d_x)  # len_k = key.size(1)
        v = self.w_v(value).view(sz_b, value.size(1), n_head, d_x)  # len_k = value.size(1)

        # Transpose for multi-head attention: (batch, seq_len, n_head, d_x) -> (batch, n_head, seq_len, d_x)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Compute scaled dot-product attention
        q, attn = self.ScaledDotProductTokenAttention(q, k, v, attn_mask)  # (batch, n_head, seq_len_q, d_x)

        # Concatenate heads and project back to d_model
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # (batch, seq_len_q, n_head * d_x)
        q = self.dropout(self.fc(q))  # Final linear projection

        # Add residual connection and apply LayerNorm
        q += query
        q = self.norm(q)
        
        return q, attn


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, cs1, cs2):
        gate_values = self.gate(torch.cat([cs1, cs2], dim=-1))
        return gate_values * cs1 + (1 - gate_values) * cs2


class GridEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner=1024, n_head=8, d_x=64, dropout=0.1, downsample=None,
                 spatial_attention=True, temporal_attention=True):
        super().__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.cross_attention = spatial_attention and temporal_attention
        self.spat_attn = MultiheadChannelAttention(d_model, n_head, d_x, dropout) if spatial_attention else None
        self.temp_attn = MultiheadTokenAttention(d_model, n_head, d_x, dropout) if temporal_attention else None
        self.cross_atten1 = MultiheadTokenAttention(d_model, n_head, d_x, dropout) if self.cross_attention else None
        self.cross_atten2 = MultiheadTokenAttention(d_model, n_head, d_x, dropout) if self.cross_attention else None
        
        # Add feedforward network
        self.ffn = PositionwiseFeedForward(d_model, d_inner, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.downsample = downsample
        self.use_checkpoint = True
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.fuse = GatedFusion(d_model) if self.cross_attention else None
    
    def _forward(self, x, spat_attn=None, temp_attn=None):
        x_trans = x.transpose(1, 2)  # [B, C, L]
        x = self.batch_norm(x_trans).transpose(1, 2)  # [B, L, C]

        if self.downsample is not None:
            x = self.downsample(x)

        # spatial attention
        if self.spatial_attention:
            xs, spat_attn_score = self.spat_attn(x)
        
        # temporal attention
        if self.temporal_attention:
            xt, temp_attn_score = self.temp_attn(x)
        
        if self.cross_attention:
            xtc, _ = self.cross_atten1(xt, xs, xs)
            xsc, _ = self.cross_atten2(xs, xt, xt)
            x = x + self.fuse(xtc, xsc)
        
        else:
            x = xt if self.temporal_attention else xs
        
        x = self.dropout(x)
        x = self.norm(x)

        x = x + self.ffn(x)
        return x, spat_attn_score, temp_attn_score

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=True)
        return self._forward(x)


class TokenMerging(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.reduce = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(2 * d_model)
    
    def forward(self, x):
        _, l, _ = x.size()

        if l % 2 != 0:
            padding = x[:, -1:, :]  # Take the last token
            x = torch.cat([x, padding], dim=1)  # Pad it to the end

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)
        x = self.norm(x)
        x = self.reduce(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        """
        Sinusoidal positional encoding for sequences.

        Args:
            embed_dim (int): The dimensionality of the embedding space.
            max_len (int): The maximum sequence length to support.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Create a matrix to hold the positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))  # Shape: [embed_dim // 2]

        # Compute the sinusoidal encodings
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Register as a buffer so it is not updated during training
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, embed_dim]

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        seq_len = x.size(1)  # Get the sequence length from the input
        return x + self.pe[:, :seq_len, :]


class GraphSeq(nn.Module):
    def __init__(self, d_in, d_model=256, n_classes=2, max_len=1000, n_layers=6, n_head=8, d_x=32,
            d_inner=512, dropout=0.1, num_phenotype=10, spatial_attention=True, temporal_attention=True):
        super(GraphSeq, self).__init__()

        # self.d_model = d_in if d_in % n_head == 0 else d_model
        self.d_model = d_model
        self.embedding = LineaEmbedding(d_in, self.d_model, dropout=dropout)

        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model, max_len)

        self.classifier = nn.Linear(self.d_model, n_classes)

        self.norm_out = nn.LayerNorm(self.d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.drop_out = nn.Dropout(dropout)

        layers = [GridEncoderLayer(self.d_model, d_inner, n_head=n_head, d_x=d_x, dropout=dropout, downsample=None,
                                   spatial_attention=spatial_attention, temporal_attention=temporal_attention)]
        if n_layers > 1:
            for _ in range(n_layers - 1):
                layers.append(GridEncoderLayer(self.d_model, d_inner, n_head=n_head, d_x=d_x, 
                dropout=dropout, downsample=TokenMerging(self.d_model),
                spatial_attention=spatial_attention, temporal_attention=temporal_attention))

        self.layers = nn.ModuleList(layers)
        
        self.num_phenotype = num_phenotype

        self.contrast_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 128)
        )
        
        self._init_params()

    
    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)


    def forward(self, x):
        # x: [B, T, V, N]
        B, T, V, N = x.shape
        x = rearrange(x, 'b t v n -> (b v) t n', b=B, v=V)
        
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for i, enc_layer in enumerate(self.layers):
            x, spat_attn, temp_attn = enc_layer(x)

        features = self.norm_out(self.pool(x.transpose(1, 2)).squeeze(-1))

        logits = self.classifier(features) # [B V, n_cls]
        logits = rearrange(logits, '(b v) c -> b v c', b=B, v=V)
        cp_fea = self.contrast_head(features)
        cp_fea = rearrange(cp_fea, '(b v) c -> b v c', b=B, v=V)
    
        return ModelOutputs(features=features, logits=logits, cp_features=cp_fea,
                            spatial_attention=spat_attn, temporal_attention=temp_attn)

import math
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv
from .utils import ModelOutputs


class LineaEmbedding(nn.Module):
    def __init__(self, n_regions, embed_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(n_regions, embed_dim)

    def forward(self, x):
        x = self.fc(x)
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
        x = F.leaky_relu(x)
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
        
        # Use a single projection matrix for Q, K, V
        self.w_qkv = nn.Linear(d_model, n_head * d_x * 3, bias=bias)
        self.fc = nn.Linear(n_head * d_x, d_model, bias=bias)
    
    def ScaledDotProductChannelAttention(self, query, key, value, adj=None):
        dx = query.size()[-1]
        scores = query.transpose(-2, -1).matmul(key) / math.sqrt(dx)
        attention = F.softmax(scores, dim=-1)
        if adj is not None:
            # adj @ attention @ adj.T
            attention = adj.matmul(attention)
            attention = attention.matmul(adj.transpose(-1, -2))
        attention = self.dropout(attention)     
        return value.matmul(attention)

    def forward(self, x, adj=None):
        
        d_x, n_head = self.d_x, self.n_head
        sz_b, len_q, _ = x.size()

        residual = x

        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(sz_b, len_q, n_head, d_x)
        k = k.view(sz_b, len_q, n_head, d_x)
        v = v.view(sz_b, len_q, n_head, d_x)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # b x n x lq x dx
        q = self.ScaledDotProductChannelAttention(q, k, v, adj) # b x n x lq x dx

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dx)
        q= self.dropout(self.fc(q))

        q += residual
        q = self.norm(q)
        
        return q


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
        
        return attention.matmul(value)  # (batch, n_head, seq_len_q, d_x)

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
        q = self.ScaledDotProductTokenAttention(q, k, v, attn_mask)  # (batch, n_head, seq_len_q, d_x)

        # Concatenate heads and project back to d_model
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # (batch, seq_len_q, n_head * d_x)
        q = self.dropout(self.fc(q))  # Final linear projection

        # Add residual connection and apply LayerNorm
        q += query
        q = self.norm(q)
        
        return q


class CrossAttentionGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)  # Learnable gate for each feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, cs1, cs2):
        # Compute gate values based on cs1
        gate = self.sigmoid(self.gate(cs1))  # Shape: [batch_size, seq_len, 1]
        
        # Weighted combination of cs1 and cs2
        output = gate * cs1 + (1 - gate) * cs2
        return output


class GridEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner=1024, n_head=8, d_x=64, dropout=0.1):
        super().__init__()
        self.tkn_attn = MultiheadTokenAttention(d_model, n_head, d_x, dropout)
        self.ch_atten = MultiheadChannelAttention(d_model, n_head, d_x, dropout)
        self.cross_atten1 = MultiheadTokenAttention(d_model, n_head, d_x, dropout)
        self.cross_atten2 = MultiheadTokenAttention(d_model, n_head, d_x, dropout)
        
        # Add feedforward network
        self.ffn = PositionwiseFeedForward(d_model, d_inner, d_model, dropout=dropout)
        self.gating = CrossAttentionGating(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, adj=None):
        # Channel attention with residual
        ch = self.ch_atten(x, adj)
        
        # Token attention with residual
        tkn = self.tkn_attn(x)

        cs1 = self.cross_atten1(tkn, ch, ch)
        cs2 = self.cross_atten2(ch, tkn, tkn)
        
        x = x + self.gating(cs1, cs2)
        x = self.dropout(x)
        x = self.norm(x)

        x = x + self.ffn(x)
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
    def __init__(self, d_in, d_model=256, n_classes=2, max_len=1000, n_layers=6, n_head=8, d_x=64,
            d_inner=1024, dropout=0.1, num_phenotype=10, brain_graph="small-world", window_size=7, cls_token=True):
        super(GraphSeq, self).__init__()

        self.d_model = d_in if d_in % n_head == 0 else d_model
        self.embedding = LineaEmbedding(d_in, self.d_model, dropout=dropout)

        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model, max_len)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_classes)
        )

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.drop_out = nn.Dropout(dropout)

        layers = [GridEncoderLayer(self.d_model, d_inner, n_head=1, d_x=self.d_model, dropout=dropout)]
        if n_layers > 1:
            layers.extend([GridEncoderLayer(self.d_model, d_inner, n_head=n_head, d_x=64, dropout=dropout) for _ in range(n_layers - 1)])

        self.layers = nn.ModuleList(layers)
        
        self.num_phenotype = num_phenotype

        self.contrast_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, 16 * self.num_phenotype, bias=False)
        )

        self.relation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, 16, bias=False)
        )
        
        self.adj = self._generate_adj_matrix(self.d_model, brain_graph)

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



    @staticmethod
    def _generate_adj_matrix(num_nodes, graph_type, **kwargs):
        """Generates adjacency matrix based on the specified graph structure."""

        adj_matrix = 0.3 * np.ones((num_nodes, num_nodes), dtype=np.float32)

        if graph_type == 'random':
            edge_prob = kwargs.get('edge_prob', 0.3)  # Default probability
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and np.random.rand() < edge_prob:
                        adj_matrix[i, j] = 1.0
        
        elif graph_type == 'scale-free':
            G = nx.barabasi_albert_graph(num_nodes, 2)  # 2 edges to attach from a new node to existing nodes
            for edge in G.edges:
                adj_matrix[edge[0], edge[1]] = 1.0
                adj_matrix[edge[1], edge[0]] = 1.0  # Assuming undirected graph
        
        elif graph_type == 'small-world':
            k = kwargs.get('k', 4)  # Each node is connected to k nearest neighbors
            p = kwargs.get('p', 0.1)  # Probability of rewiring each edge
            G = nx.watts_strogatz_graph(num_nodes, k, p)
            for edge in G.edges:
                adj_matrix[edge[0], edge[1]] = 1.0
                adj_matrix[edge[1], edge[0]] = 1.0  # Assuming undirected graph
        
        elif graph_type == 'fully-connected':
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        adj_matrix[i, j] = 1.0
        
        else:
            raise ValueError("Unsupported graph type. Choose from 'random', 'scale-free', 'small-world', or 'fully-connected'.")
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        adj_matrix = nn.Parameter(adj_matrix, requires_grad=True)
        return adj_matrix
    
    @staticmethod
    def _prepare_adj(adj, device):
        """Prepares the adjacency matrix for use in the model."""
        adj = adj.to(device)
        # Ensure the adjacency matrix is symmetric
        adj = adj.matmul(adj.transpose(-1, -2))
        # Normalize the adjacency matrix
        adj = F.softmax(adj, dim=-1)
        return adj

    def forward(self, x):
        # x: [B, T, V, N]
        B, T, V, N = x.shape
        adj = self._prepare_adj(self.adj, x.device)
        x = rearrange(x, 'b t v n -> (b v) t n', b=B, v=V)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.norm_in(x)
        x = self.drop_out(x)

        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(x, adj) if i == 0 else enc_layer(x) # [B V, T, d_model]

        features = self.norm_out(self.pool(x.transpose(1, 2)).squeeze(-1))

        logits = self.classifier(features) # [B V, n_cls]
        logits = rearrange(logits, '(b v) c -> b v c', b=B, v=V)
        cp_fea = self.contrast_head(features)
        cp_fea = rearrange(cp_fea, '(b v) c -> b v c', b=B, v=V)

        cnp_fea = self.relation_head(features)
    
        return ModelOutputs(features=features, logits=logits, cp_features=cp_fea, cnp_features=cnp_fea)

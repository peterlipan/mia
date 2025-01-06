import math
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GCNConv


class LineaEmbedding(nn.Module):
    def __init__(self, n_regions, embed_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(n_regions, embed_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_out) # position-wise
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):


        x = self.w_2(F.gelu(self.w_1(x)))
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


class GridEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner=1024, n_head=8, d_x=64, dropout=0.1):
        super().__init__()
        self.tkn_attn = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        self.ch_atten = MultiheadChannelAttention(d_model, 1, d_x, dropout)
        self.cross_atten1 = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        self.cross_atten2 = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        
        # Add feedforward network
        self.ffn = PositionwiseFeedForward(d_model * 2, d_inner, d_model, dropout=dropout)
    
    def forward(self, x, adj=None):
        # Channel attention with residual
        ch = self.ch_atten(x, adj)
        
        # Token attention with residual
        tkn = self.tkn_attn(x, x, x)[0]

        cs1 = self.cross_atten1(tkn, ch, ch)[0]
        cs2 = self.cross_atten2(ch, tkn, tkn)[0]
        # concat the two cross attention
        x = torch.cat([cs1, cs2], dim=-1)

        x = self.ffn(x)
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
            d_inner=512, dropout=0.1, num_phenotype=10, brain_graph="small-world"):
        super(GraphSeq, self).__init__()

        self.embedding = LineaEmbedding(d_in, d_in)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_in), requires_grad=True)
        # self.positional_encoding = SinusoidalPositionalEncoding(d_in, max_len)

        self.classifier = nn.Linear(d_in, n_classes)

        self.norm_in = nn.LayerNorm(d_in)
        self.norm_out = nn.LayerNorm(d_in)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.drop_out = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GridEncoderLayer(d_in, d_inner, n_head, d_x=d_in, dropout=dropout)
            for _ in range(n_layers)])
        
        self.num_phenotype = num_phenotype

        self.contrast_head = nn.Sequential(
            nn.Linear(d_in, d_in, bias=False),
            nn.GELU(),
            nn.Linear(d_in, 64 * self.num_phenotype, bias=False)
        )
        
        self.adj = self._generate_adj_matrix(d_in, brain_graph)

        self._init_params()

    
    def _init_params(self):
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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
        x = self.embedding(x) # [B V, T, N]
        x = x + self.positional_encoding[:, :T]

        for enc_layer in self.layers:
            x = enc_layer(x, adj)

        x = self.norm_out(self.pool(x.transpose(1, 2)).squeeze(-1)) # [B V, C]

        logits = self.classifier(x) # [B V, n_cls]
        con_fea = self.contrast_head(x)
        con_fea = rearrange(con_fea, '(b v) c -> b v c', b=B, v=V)
    
        return logits, con_fea, x

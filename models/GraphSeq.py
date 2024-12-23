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


class ChannelAttention(nn.Module):
    def __init__(self, in_features, bias=True, activation=F.relu):
        super().__init__()
        self.in_features = in_features
        self.activation = activation
        self.bias = bias
        
        # Use a single projection matrix for Q, K, V
        self.qkv_proj = nn.Linear(in_features, in_features * 3, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
    
    def ScaledDotProductChannelAttention(self, query, key, value, adj=None):
        dk = query.size()[-1]
        scores = query.transpose(-2, -1).matmul(key) / math.sqrt(dk)
        attention = F.softmax(scores, dim=-1)
        if adj is not None:
            attention = attention * adj     
        return value.matmul(attention)

    def forward(self, x, adj=None):
        # Project all at once and split
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: self.activation(t) if self.activation else t, qkv)

        y = self.ScaledDotProductChannelAttention(q, k, v, adj)
        y = self.linear_o(y)
        
        # Add residual connection and normalization
        return y


class GridSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=8, drop_rate=0.1):
        super().__init__()
        self.tkn_attn = nn.MultiheadAttention(embed_dim, n_heads, drop_rate, batch_first=True)
        self.ch_atten = ChannelAttention(embed_dim)
        self.cross_atten = nn.MultiheadAttention(embed_dim, n_heads, drop_rate, batch_first=True)
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Add feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x, adj=None):
        # Channel attention with residual
        ch = x + self.drop(self.ch_atten(x, adj))
        
        # Token attention with residual
        tkn = x + self.tkn_attn(x, x, x)[0]

        x = self.cross_atten(tkn, ch, ch)[0] + x
        
        # FFN with residual
        x = x + self.drop(self.ffn(self.norm(x)))
        return x



class Transformer(nn.Module):
    def __init__(self, n_regions, embed_dim, n_classes, max_len=1000, n_heads=8, drop_rate=0.1, brain_graph='small-world'):
        super(Transformer, self).__init__()

        self.embedding = LineaEmbedding(n_regions, n_regions)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, n_regions))

        self.attention1 = GridSelfAttention(n_regions, n_heads, drop_rate)
        self.attention2 = GridSelfAttention(embed_dim, n_heads, drop_rate)
        self.classifier = nn.Linear(embed_dim, n_classes)

        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.shadow_layer = nn.Sequential(
            nn.Linear(n_regions, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.contrast_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, 64, bias=False)
        )
        
        self.adj = self._generate_adj_matrix(n_regions, brain_graph)
        self.adj = nn.Parameter(self.adj, requires_grad=False)

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

        return torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)


    def forward(self, x):
        # x: [B, T, V, N]
        B, T, V, N = x.shape
        self.adj = self.adj.to(x.device)
        x = rearrange(x, 'b t v n -> (b v) t n', b=B, v=V)
        x = self.embedding(x) # [B V, T, N]
        x = x + self.positional_encoding[:, :T]

        x = self.attention1(x, self.adj) # [B V, T, C]
        x = self.shadow_layer(x) # [B V, T, C]
        x = self.attention2(x)
        x = self.norm(self.pool(x.transpose(1, 2)).squeeze(-1)) # [B V, C]

        logits = self.classifier(x) # [B V, n_cls]
        con_fea = self.contrast_head(x)
        con_fea = rearrange(con_fea, '(b v) c -> b v c', b=B, v=V)
    
        return logits, con_fea

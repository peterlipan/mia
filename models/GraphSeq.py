import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from einops import rearrange
from torch_geometric.nn import GCNConv


class LineaEmbedding(nn.Module):
    def __init__(self, n_regions):
        super(LineaEmbedding, self).__init__()
        self.fc = nn.Linear(n_regions, n_regions)

    def forward(self, x):
        return self.fc(x)


class GraphEmbedding(nn.Module):
    def __init__(self, n_regions):
        super(GraphEmbedding, self).__init__()
        # keep the node dim
        self.conv1 = GCNConv(1, 1, node_dim=1)
        self.conv2 = GCNConv(1, 1, node_dim=1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


class Transformer(nn.Module):
    def __init__(self, n_regions, embed_dim, n_classes, max_len=1000, n_heads=8, drop_rate=0.1, brain_graph='fully-connected'):
        super(Transformer, self).__init__()

        self.embedding = LineaEmbedding(n_regions)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, n_regions))
        self.attention1 = nn.MultiheadAttention(n_regions, n_heads, drop_rate, batch_first=True)
        self.attention2 = nn.MultiheadAttention(n_regions, n_heads, drop_rate, batch_first=True)
        self.classifier = nn.Linear(n_regions, n_classes)
        self.norm = nn.LayerNorm(n_regions)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.contrast_head = nn.Sequential(
            nn.Linear(n_regions, n_regions, bias=False),
            nn.ReLU(),
            nn.Linear(n_regions, 64, bias=False)
        )

        # self.edge_index = self._generate_edge_index(n_regions, brain_graph)

        # self.num_attn = nn.Parameter(torch.ones(1, num_dim) / num_dim, requires_grad=True)
        # self.str_attn = nn.Parameter(torch.ones(1, str_num) / str_num, requires_grad=True)
        # self.str_oh_exp = torch.tensor(str_oh_exp, requires_grad=False)
        # self.imputation = nn.Parameter(torch.zeros(1, phe_dim), requires_grad=True)

    @staticmethod
    def _generate_edge_index(num_nodes, graph_type, **kwargs):
        """Generates edge_index based on the specified graph structure."""

        if graph_type == 'random':
            edge_prob = kwargs.get('edge_prob', 0.3)  # Default probability
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and np.random.rand() < edge_prob:
                        edges.append((i, j))
            edge_index = torch.tensor(edges, dtype=torch.long, requires_grad=False).t().contiguous()
        
        elif graph_type == 'scale-free':
            G = nx.barabasi_albert_graph(num_nodes, 2)  # 2 edges to attach from a new node to existing nodes
            edge_index = np.array(G.edges).T
            edge_index = torch.tensor(edge_index, dtype=torch.long, requires_grad=False).contiguous()
        
        elif graph_type == 'small-world':
            k = kwargs.get('k', 4)  # Each node is connected to k nearest neighbors
            p = kwargs.get('p', 0.1)  # Probability of rewiring each edge
            G = nx.watts_strogatz_graph(num_nodes, k, p)
            edge_index = np.array(G.edges).T
            edge_index = torch.tensor(edge_index, dtype=torch.long, requires_grad=False).contiguous()
        
        elif graph_type == 'fully-connected':
            edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
            edge_index = torch.tensor(edges, dtype=torch.long, requires_grad=False).t().contiguous()
        
        else:
            raise ValueError("Unsupported graph type. Choose from 'random', 'scale-free', 'small-world', or 'fully-connected'.")
        
        return edge_index
    

    def forward(self, x):
        # x: [B, T, V, N]
        B, T, V, N = x.shape

        # self.edge_index = self.edge_index.to(x.device)
        # x = rearrange(x, 'b t v n -> (b v t) n')
        # x = x.unsqueeze(-1) # [B V T, N, 1]
        # x = self.embedding(x, self.edge_index) # [B V T, N, 1]
        # x = x.squeeze(-1) # [B V T, N]
        # x = rearrange(x, '(b v t) n -> (b v) t n', b=B, v=V, t=T)

        x = rearrange(x, 'b t v n -> (b v) t n', b=B, v=V)
        x = self.embedding(x) # [B V, T, N]
        x = x + self.positional_encoding[:, :T]
        x = self.attention1(x, x, x)[0] # [B V, T, C]
        x = self.norm(x)
        x = self.attention2(x, x, x)[0]
        x = self.norm(self.pool(x.transpose(1, 2)).squeeze(-1)) # [B V, C]

        logits = self.classifier(x) # [B V, n_cls]
        con_fea = self.contrast_head(x)
        con_fea = rearrange(con_fea, '(b v) c -> b v c', b=B, v=V)
    
        return logits, con_fea

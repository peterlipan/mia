import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNTEmbeddings(nn.Module):
    def __init__(self, d_in, d_hid, max_len, kernel_size=3, embed_type='conv', 
                 dropout=0.0, use_mask_token=False, use_cls_token=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hid)) if use_cls_token else None
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_hid)) if use_mask_token else None
        self.window_embeddings = BNTWindowEmbeddings(d_in, d_hid, kernel_size, embed_type)
        self.position_embeddings = nn.Parameter(torch.randn(1, max_len, d_hid))
        self.addition = 1 if use_cls_token else 0
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, bool_masked_pos=None):
        # x: [B, L, C]
        batch_size, seq_len, _ = x.size()
        T = seq_len + self.addition
        embeddings = self.window_embeddings(x)

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - mask) + mask_tokens * mask
        
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        embeddings = embeddings + self.position_embeddings[:, :T]
        embeddings = self.dropout(embeddings)

        return embeddings


class BNTWindowEmbeddings(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size=3, embed_type='conv'):
        super().__init__()
        self.embed_type = embed_type
        if embed_type == 'conv':
            self.window = nn.Conv1d(d_in, d_hid, kernel_size, stride=1, padding='same')
        elif embed_type == 'linear':
            self.window = nn.Linear(d_in, d_hid)
        else:
            raise NotImplementedError(f"Embedding type {embed_type} not implemented")
    
    def forward(self, x):
        # x: [B, L, C]
        if self.embed_type == 'conv':
            x = x.transpose(1, 2)
            x = self.window(x)
            x = x.transpose(1, 2)
        elif self.embed_type == 'linear':
            x = self.window(x)
        return x
        

class BNTChannelAttention(nn.Module):
    def __init__(self, d_hid, n_heads, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_hid // n_heads
        self.all_head_dim = self.n_heads * self.d_head 

        self.qkv = nn.Linear(d_hid, self.all_head_dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.d_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        # x: [B, L, C]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self.transpose_for_scores, (q, k, v))

        # attention among channels
        attn = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        outputs = torch.matmul(v, attn)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_shape = outputs.size()[:-2] + (self.all_head_dim,)
        outputs = outputs.view(*new_shape)

        return outputs


class BNTTokenAttention(nn.Module):
    def __init__(self, d_hid, n_heads, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_hid // n_heads
        self.all_head_dim = self.n_heads * self.d_head 

        self.qkv = nn.Linear(d_hid, self.all_head_dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.d_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        # x: [B, L, C]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self.transpose_for_scores, (q, k, v))

        # attention among tokens
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        outputs = torch.matmul(attn, v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_shape = outputs.size()[:-2] + (self.all_head_dim,)
        outputs = outputs.view(*new_shape)

        return outputs


class BNTCrossAttention(nn.Module):
    def __init__(self, d_hid, n_heads, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_hid // n_heads
        self.all_head_dim = self.n_heads * self.d_head 

        self.linear_q = nn.Linear(d_hid, self.all_head_dim, bias=qkv_bias)
        self.linear_k = nn.Linear(d_hid, self.all_head_dim, bias=qkv_bias)
        self.linear_v = nn.Linear(d_hid, self.all_head_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.d_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v):
        # q, k, v: [B, L, C]
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q, k, v = map(self.transpose_for_scores, (q, k, v))

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        outputs = torch.matmul(attn, v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_shape = outputs.size()[:-2] + (self.all_head_dim,)
        outputs = outputs.view(*new_shape)

        return outputs


class BNTFeedForward(nn.Module):
    def __init__(self, d_hid, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_hid * 2, d_hid * 4)
        self.linear2 = nn.Linear(d_hid * 4, d_hid)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, L, C]
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class BNTLayer(nn.Module):
    def __init__(self, d_hid, n_heads, qkv_bias=True, 
                 attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0):
        super().__init__()
        self.channel_attention = BNTChannelAttention(d_hid, n_heads, attention_probs_dropout_prob, qkv_bias)
        self.token_attention = BNTTokenAttention(d_hid, n_heads, attention_probs_dropout_prob, qkv_bias)
        self.corss_attention1 = BNTCrossAttention(d_hid, n_heads, attention_probs_dropout_prob, qkv_bias)
        self.corss_attention2 = BNTCrossAttention(d_hid, n_heads, attention_probs_dropout_prob, qkv_bias)

        self.feed_forward = BNTFeedForward(d_hid, hidden_dropout_prob)

        self.layernorm_before = nn.LayerNorm(d_hid, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(d_hid * 2, eps=1e-12)
    
    def forward(self, x, mask=None):
        # x: [B, L, C]
        x = self.layernorm_before(x)
        ch_attn = x + self.channel_attention(x, mask)
        tk_attn = x + self.token_attention(x, mask)
        cs_attn1 = self.corss_attention1(ch_attn, tk_attn, tk_attn)
        cs_attn2 = self.corss_attention2(tk_attn, ch_attn, ch_attn)
        
        attn = torch.cat((cs_attn1, cs_attn2), dim=-1)
        attn = self.layernorm_after(attn)
        attn = self.feed_forward(attn)
        x = x + attn

        return x


class BNTEncoder(nn.Module):
    def __init__(self, d_hid, n_layers, n_heads, qkv_bias=True, 
                 attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0):
        super().__init__()

        self.layers = nn.ModuleList([
            BNTLayer(d_hid, n_heads, qkv_bias, attention_probs_dropout_prob, hidden_dropout_prob)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BNT(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, n_classes, n_heads=8, kernel_size=3,
                 embed_type='conv', qkv_bias=True, teacher=False, max_len=1000, 
                 use_cls_token=True, use_mask_token=True, add_pooloer=True,
                 attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.embeddings = BNTEmbeddings(d_in, d_hid, max_len, kernel_size, embed_type,
                                        use_cls_token=use_cls_token, use_mask_token=use_mask_token)
        self.encoder = BNTEncoder(d_hid, n_layers, n_heads, qkv_bias,
                                  attention_probs_dropout_prob, hidden_dropout_prob)
        self.classifier = nn.Linear(d_hid, n_classes)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooloer else None

        if teacher:
            for p in self.parameters():
                p.detach_()
    
        else:
            self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1)
        elif isinstance(module, BNTEmbeddings):
            module.position_embeddings.data.normal_(mean=0.0, std=0.02)         
        
    def forward(self, x, token_mask=None):
        x = self.embeddings(x, token_mask)
        hidden_states = self.encoder(x)
        if self.use_cls_token:
            if self.pooler is not None:
                features = self.pooler(hidden_states.transpose(1, 2)[:, :, 1:]).squeeze(-1)
            else:
                features = hidden_states[:, 0].contiguous()
        else:
            features = self.pooler(hidden_states.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(features)
        return features, logits

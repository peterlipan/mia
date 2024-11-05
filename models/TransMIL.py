import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, d_in):
        super(TransMIL, self).__init__()
        self.layer1 = TransLayer(dim=d_in)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, h):
        # h = kwargs['data'].float()  # [B, n, d_in]


        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, d_in]
        # ---->Translayer x2
        # h = self.layer2(h)  # [B, N, d_in]

        h = self.norm(self.pool(h.transpose(1, 2)).squeeze(-1))


        return h


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(d_in=256).cuda()
    print(model.eval())
    results_dict = model(data=data)
    print(results_dict)
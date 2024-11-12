# decouple the encoder and classifier for interative training
from torch import nn
from einops import rearrange
from timm.models.layers import trunc_normal_


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        return x


def get_classifier(n_features, n_classes):
    return nn.Linear(n_features, n_classes)


def get_encoder(args):
    if args.backbone == 'groupvit':
        from .GroupViT1D import GroupViT1D
        return GroupViT1D(n_regions=args.num_roi, expand=args.expand, embed_dim=args.embed_dim, drop_rate=args.dropout)
    elif args.backbone == 'swin':
        from .Swin3D import SwinTransformer3d
        return SwinTransformer3d(patch_size=args.patch_size, embed_dim=args.embed_dim, dropout=args.dropout) 
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

def get_aggregator(args):
    if args.aggregator == 'transmil':
        from .TransMIL import TransMIL
        return TransMIL(d_in=args.embed_dim)
    elif args.aggregator == 'pool':
        return Pool()


class WholeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        self.aggregator = get_aggregator(args)
        self.classifier = get_classifier(args.embed_dim, args.n_classes)


    def forward(self, roi):
        # roi [B, N, T]
        features = self.encoder(roi)
        # features: [B, N, C]
        features = self.aggregator(features)
        # features: [B, C]
        logits = self.classifier(features)

        return logits

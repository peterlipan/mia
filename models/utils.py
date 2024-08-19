# decouple the encoder and classifier for interative training
from torch import nn
from timm.models.layers import trunc_normal_


def get_classifier(n_features, n_classes):
    return nn.Linear(n_features, n_classes)


def get_encoder(args):
    if args.backbone == 'groupvit':
        from .GroupViT import GroupViT
        return GroupViT(img_size=args.image_size, patch_size=args.patch_size, embed_dim=args.embed_dim)
    elif args.backbone == 'swin':
        from .Swin3D import SwinTransformer3d
        return SwinTransformer3d(patch_size=args.patch_size, embed_dim=args.embed_dim)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

def get_aggregator(args):
    from .TransMIL import TransMIL
    return TransMIL(d_in=args.embed_dim)

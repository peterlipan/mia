# decouple the encoder and classifier for interative training
from torch import nn
from timm.models.layers import trunc_normal_


def get_classifier(n_features, n_classes):
    return nn.Linear(n_features, n_classes)


def get_encoder(args):
    if args.model == 'groupvit':
        from .GroupViT import GroupViT
        return GroupViT(img_size=args.image_size, patch_size=args.patch_size)
    elif args.model == 'swin':
        from models.Swin3D import SwinTransformer3d
        return SwinTransformer3d(patch_size=args.patch_size)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.fc = nn.Linear(n_features, n_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        return self.fc(self.norm(x))


# pack the encoder and classifier into one for easier training
class MyModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

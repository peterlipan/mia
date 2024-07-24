# decouple the encoder and classifier for interative training
from torch import nn


def get_classifier(n_features, n_classes):
    return nn.Linear(n_features, n_classes)


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

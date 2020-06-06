import torch.nn as nn
from timm import create_model

from src.models.classifiers import Classifier


class TimmModel(nn.Module):
    def __init__(self,
                 encoder="gluon_resnet50_v1d",
                 pretrained=True):
        super().__init__()
        self.model = create_model(encoder, pretrained=pretrained)

        if 'resnet' in encoder:
            self.model.fc = Classifier(self.model.fc.in_features)
        elif 'efficientnet' in encoder:
            self.model.classifier = Classifier(self.model.classifier.in_features)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

    def forward(self, x):
        return self.model(x)

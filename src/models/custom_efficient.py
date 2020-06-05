from functools import partial

import torch.nn as nn
from timm import create_model

from src.models.classifiers import Classifier


ENCODERS = {
    "efficientnet_b0": (partial(create_model, 'efficientnet_b0'), 1280),
    "tf_efficientnet_b1": (partial(create_model, 'tf_efficientnet_b1'), 1280),
    "tf_efficientnet_b0_ns": (partial(create_model, 'tf_efficientnet_b0_ns'), 1280),
    "tf_efficientnet_b2_ns": (partial(create_model, 'tf_efficientnet_b2_ns'), 1408),
    "tf_efficientnet_b3_ns": (partial(create_model, 'tf_efficientnet_b3_ns'), 1536),
    "tf_efficientnet_b4_ns": (partial(create_model, 'tf_efficientnet_b4_ns'), 1792),
    "tf_efficientnet_b5_ns": (partial(create_model, 'tf_efficientnet_b5_ns'), 2048),
}


class CustomEfficient(nn.Module):
    def __init__(self,
                 encoder="tf_efficientnet_b0_ns",
                 pretrained=True):
        super().__init__()

        efficient, num_bottleneck_filters = ENCODERS[encoder]
        self.efficient = efficient(pretrained=pretrained)
        self.efficient.classifier = Classifier(num_bottleneck_filters)

    def forward(self, x):
        return self.efficient(x)

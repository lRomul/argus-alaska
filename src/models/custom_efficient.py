from functools import partial

import torch.nn as nn
from timm import create_model
from timm.models.layers.conv2d_same import Conv2dSame

from src.models.classifiers import Classifier


ENCODERS = {
    "efficientnet_b0": (partial(create_model, 'efficientnet_b0'), 1280),
    "tf_efficientnet_b1": (partial(create_model, 'tf_efficientnet_b1'), 1280),
    "tf_efficientnet_b0_ns": (partial(create_model, 'tf_efficientnet_b0_ns'), 1280),
    "tf_efficientnet_b3_ns": (partial(create_model, 'tf_efficientnet_b3_ns'), 1536),
    "tf_efficientnet_b4_ns": (partial(create_model, 'tf_efficientnet_b4_ns'), 1792),
    "tf_efficientnet_b5_ns": (partial(create_model, 'tf_efficientnet_b5_ns'), 2048),
}


class CustomEfficient(nn.Module):
    def __init__(self,
                 encoder="tf_efficientnet_b0_ns",
                 in_channels=3,
                 pretrained=True):
        super().__init__()

        efficient, num_bottleneck_filters = ENCODERS[encoder]
        self.efficient = efficient(pretrained=pretrained)
        if in_channels != 3:
            new_conv = Conv2dSame(in_channels, 32, kernel_size=(3, 3),
                                  stride=(2, 2), bias=False)
            old_conv = self.efficient.conv_stem
            num_copy = min(in_channels, old_conv.in_channels)
            new_conv.weight.data[:, :num_copy] = old_conv.weight.data[:, :num_copy]
            self.efficient.conv_stem = new_conv
        self.efficient.classifier = Classifier(num_bottleneck_filters)

    def forward(self, x):
        return self.efficient(x)

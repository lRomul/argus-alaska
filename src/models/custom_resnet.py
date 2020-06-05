from functools import partial

import torch.nn as nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)
from timm import create_model

from src.models.classifiers import Classifier


ENCODERS = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 2048),
    "resnet101": (resnet101, 2048),
    "resnet152": (resnet152, 2048),
    "gluon_resnet34_v1b": (partial(create_model, 'gluon_resnet34_v1b'), 512),
    "gluon_resnet50_v1d": (partial(create_model, 'gluon_resnet50_v1d'), 2048),
    "gluon_seresnext50_32x4d": (partial(create_model, 'gluon_seresnext50_32x4d'), 2048),
    "seresnext26t_32x4d": (partial(create_model, 'seresnext26t_32x4d'), 2048),
    "resnet50_jsd": (partial(create_model, 'resnet50'), 2048),
    "skresnext50_32x4d": (partial(create_model, 'skresnext50_32x4d'), 2048),
}


class CustomResnet(nn.Module):
    def __init__(self,
                 encoder="resnet34",
                 pretrained=True):
        super().__init__()
        resnet, num_bottleneck_filters = ENCODERS[encoder]
        resnet = resnet(pretrained=pretrained)

        if hasattr(resnet, 'relu'):
            act = resnet.relu
        elif hasattr(resnet, 'act1'):
            act = resnet.act1
        else:
            raise Exception

        self.first_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, act, resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.global_pool = resnet.global_pool

        self.classifier = Classifier(num_bottleneck_filters)

    def forward(self, x):
        x = self.first_layers(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)

        return x

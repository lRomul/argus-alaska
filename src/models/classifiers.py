from torch import nn

from src import config


class Classifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.altered_fc = nn.Linear(in_features,
                                    config.num_unique_targets)
        self.quality_fc = nn.Linear(in_features,
                                    config.num_qualities)

    def forward(self, x):
        altered = self.altered_fc(x)
        quality = self.quality_fc(x)
        return altered, quality

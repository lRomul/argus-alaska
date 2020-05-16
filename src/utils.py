import torch

from src import config


def target2altered(probs):
    altered = probs[:, config.altered_targets]
    altered = torch.sum(altered, dim=1)
    return altered

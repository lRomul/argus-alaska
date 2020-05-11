import torch
import numpy as np

from sklearn import metrics

from argus.metrics.metric import Metric


def alaska_weighted_auc(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


class BinaryAccuracy(Metric):
    name = 'binary_accuracy'
    better = 'max'

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.correct = 0
        self.count = 0

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        pred = step_output['prediction']
        trg = step_output['target']
        pred = (pred > self.threshold).to(torch.float32)
        correct = torch.eq(pred, trg).all(dim=1).view(-1)
        self.correct += torch.sum(correct).item()
        self.count += correct.shape[0]

    def compute(self):
        if self.count == 0:
            raise Exception('Must be at least one example for computation')
        return self.correct / self.count


class WeightedAuc(Metric):
    name = 'weighted_auc'
    better = 'max'

    def __init__(self):
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, step_output: dict):
        pred = step_output['prediction']
        target = step_output['target']
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        score = alaska_weighted_auc(y_true, y_pred)
        return score

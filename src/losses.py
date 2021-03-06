import torch
from torch import nn
import torch.nn.functional as F

from src import config


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, softmax=True):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.softmax = softmax

    def forward(self, x, target):
        if self.softmax:
            logprobs = F.log_softmax(x, dim=-1)
        else:
            logprobs = torch.log(x)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class SmoothingOhemCrossEntropy(nn.Module):
    def __init__(self, smooth_factor=0.0, ohem_rate=1.0, softmax=True):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ohem_rate = ohem_rate
        self.ce = LabelSmoothingCrossEntropy(smoothing=self.smooth_factor,
                                             softmax=softmax)

    def forward(self, label_input, label_target, training=False):
        if isinstance(label_target, (tuple, list)):
            y1, y2, lam = label_target
            loss = self.ce(label_input, y1) * lam + self.ce(label_input, y2) * (1 - lam)
        else:
            loss = self.ce(label_input, label_target)

        if training and self.ohem_rate < 1.0:
            _, idx = torch.sort(loss, descending=True)
            keep_num = int(label_input.size(0) * self.ohem_rate)
            if keep_num < label_input.size(0):
                keep_idx = idx[:keep_num]
                loss = loss[keep_idx]
                return loss.sum() / keep_num

        return loss.mean()


class AlaskaCrossEntropy(nn.Module):
    def __init__(self,
                 stegano_weight=1.0,
                 altered_weight=1.0,
                 quality_weight=1.0,
                 smooth_factor=0,
                 ohem_rate=1.0):
        super().__init__()

        self.stegano_weight = stegano_weight
        self.altered_weight = altered_weight
        self.quality_weight = quality_weight
        self.smooth_factor = smooth_factor
        self.ohem_rate = ohem_rate

        self.stegano_softmax = not bool(self.altered_weight)

        self.stegano_ce = SmoothingOhemCrossEntropy(smooth_factor=smooth_factor,
                                                    ohem_rate=ohem_rate,
                                                    softmax=self.stegano_softmax)
        self.altered_ce = SmoothingOhemCrossEntropy(smooth_factor=smooth_factor,
                                                    ohem_rate=ohem_rate,
                                                    softmax=self.stegano_softmax)
        self.quality_ce = SmoothingOhemCrossEntropy(smooth_factor=smooth_factor,
                                                    ohem_rate=ohem_rate,
                                                    softmax=True)

    def __call__(self, pred, target, training=False):
        stegano_pred, quality_pred = pred
        stegano_target, quality_target = target

        if not self.stegano_softmax:
            stegano_pred = F.softmax(stegano_pred, dim=-1)

        loss = 0
        if self.stegano_weight:
            loss += (
                self.stegano_weight
                * self.stegano_ce(stegano_pred, stegano_target,
                                  training=training)
            )

        if self.altered_weight:
            altered_pred = torch.stack([
                stegano_pred[:, config.unaltered_target],
                torch.sum(stegano_pred[:, config.altered_targets], dim=1),
            ], dim=1)
            if isinstance(stegano_target, (tuple, list)):
                stegano_trg1, stegano_trg2, lam = stegano_target
                stegano_trg1 = stegano_trg1.to(torch.bool).to(torch.int64)
                stegano_trg2 = stegano_trg2.to(torch.bool).to(torch.int64)
                altered_target = stegano_trg1, stegano_trg2, lam
            else:
                altered_target = stegano_target.to(torch.bool).to(torch.int64)
            loss += (
                self.altered_weight
                * self.altered_ce(altered_pred,
                                  altered_target,
                                  training=training)
            )

        if self.quality_weight:
            loss += (
                self.quality_weight
                * self.quality_ce(quality_pred, quality_target,
                                  training=training)
            )

        return loss

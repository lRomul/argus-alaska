import torch
from adamp import AdamP

from argus import Model
from argus.utils import deep_to, deep_detach
from argus.optimizer import pytorch_optimizers

from src.models import CustomEfficient, CustomResnet, TimmModel
from src.losses import AlaskaCrossEntropy


def get_prediction_transform():
    return lambda logits: [torch.softmax(x, dim=1) for x in logits]


class AlaskaModel(Model):
    nn_module = {
        'CustomEfficient': CustomEfficient,
        'CustomResnet': CustomResnet,
        'TimmModel': TimmModel,
    }
    loss = {
        'AlaskaCrossEntropy': AlaskaCrossEntropy
    }
    optimizer = {
        'AdamP': AdamP,
        **pytorch_optimizers
    }
    prediction_transform = get_prediction_transform

    def __init__(self, params):
        super().__init__(params)
        self.amp = None
        self.model_ema = None

        if 'iter_size' not in self.params:
            self.params['iter_size'] = 1

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()

        input, target = batch
        stegano_target, quality_target = target

        inputs = torch.chunk(input, self.params['iter_size'], dim=0)
        stegano_targets = torch.chunk(stegano_target, self.params['iter_size'], dim=0)
        quality_targets = torch.chunk(quality_target, self.params['iter_size'], dim=0)
        n_chunks = len(inputs)

        for i, input, stegano_target, quality_target in zip(range(n_chunks), inputs,
                                                            stegano_targets, quality_targets):
            target = stegano_target, quality_target
            input = deep_to(input, self.device, non_blocking=True)
            target = deep_to(target, self.device, non_blocking=True)

            prediction = self.nn_module(input)
            loss = self.loss(prediction, target, training=True)
            if self.amp is not None:
                delay_unscale = i != (n_chunks - 1)
                with self.amp.scale_loss(loss, self.optimizer,
                                         delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        self.optimizer.step()

        torch.cuda.synchronize()
        if self.model_ema is not None:
            with torch.no_grad():
                self.model_ema.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch, state) -> dict:
        self.eval()
        with torch.no_grad():
            input, target = self.prepare_batch(batch, self.device)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }

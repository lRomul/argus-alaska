import torch

from argus import Model
from argus.utils import deep_detach

from src.models.custom_efficient import CustomEfficient


class AlaskaModel(Model):
    nn_module = {
        'CustomEfficient': CustomEfficient,
    }

    def __init__(self, params):
        super().__init__(params)
        self.amp = None
        self.model_ema = None

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        if self.amp is not None:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        torch.cuda.synchronize()
        if self.model_ema is not None:
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

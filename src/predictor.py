import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import AlaskaDataset


@torch.no_grad()
def predict_data(data, model, batch_size, transform, tta=False):

    dataset = AlaskaDataset(data,
                            target=False,
                            folds=None,
                            transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False)

    grapheme_preds_lst = []
    quality_preds_lst = []

    for batch in loader:
        altered_pred, quality_pred = model.predict(batch)

        if tta:
            hflip_batch = torch.flip(batch, [2])
            hflip_altered_pred, hflip_quality_pred = model.predict(hflip_batch)

            vflip_batch = torch.flip(batch, [3])
            vflip_altered_pred, vflip_quality_pred = model.predict(vflip_batch)

            altered_pred = (0.5 * altered_pred
                            + 0.25 * hflip_altered_pred
                            + 0.25 * vflip_altered_pred)
            quality_pred = (0.5 * quality_pred
                            + 0.25 * hflip_quality_pred
                            + 0.25 * vflip_quality_pred)

        grapheme_preds_lst.append(altered_pred)
        quality_preds_lst.append(quality_pred)

    altered_pred = torch.cat(grapheme_preds_lst, dim=0)
    altered_pred = altered_pred.cpu().numpy()

    quality_pred = torch.cat(quality_preds_lst, dim=0)
    quality_pred = quality_pred.cpu().numpy()

    return altered_pred, quality_pred


class Predictor:
    def __init__(self,
                 model_path,
                 batch_size,
                 transform,
                 device='cuda',
                 logits=False,
                 tta=False):
        self.model = load_model(model_path, device=device)
        if logits:
            self.model.prediction_transform = lambda x: x
        self.batch_size = batch_size
        self.transform = transform
        self.tta = tta

    def predict(self, data):
        pred = predict_data(data, self.model,
                            self.batch_size, self.transform, self.tta)
        return pred

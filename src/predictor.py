import torch
from torch.utils.data import DataLoader

from argus import load_model

from src.datasets import AlaskaDataset


@torch.no_grad()
def predict_data(data, model, batch_size, transform):

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
        pred_batch = model.predict(batch)
        altered_pred, quality_pred = pred_batch

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
                 logits=False):
        self.model = load_model(model_path, device=device)
        if logits:
            self.model.prediction_transform = lambda x: x
        self.batch_size = batch_size
        self.transform = transform

    def predict(self, data):
        pred = predict_data(data, self.model,
                            self.batch_size, self.transform)
        return pred

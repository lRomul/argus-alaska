import argparse
import numpy as np
import pandas as pd

from src.utils import get_best_model_path, target2altered, check_dir_not_exist
from src.datasets import get_test_data
from src.predictor import Predictor
from src.transforms import get_transforms
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

EXPERIMENT_DIR = config.experiments_dir / args.experiment
PREDICTION_DIR = config.predictions_dir / args.experiment
DEVICE = 'cuda'
BATCH_SIZE = 32
LOGITS = False
TTA = True

NAME = args.experiment
if TTA:
    NAME += '-tta'
if LOGITS:
    NAME += '-logits'


def predict_test_fold(test_data, predictor):
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    image_names = [s['name'] for s in test_data]

    preds = predictor.predict(test_data)
    altered_pred, quality_pred = preds

    np.savez(
        PREDICTION_DIR / f'preds-{NAME}.npz',
        altered_pred=altered_pred,
        quality_pred=quality_pred,
        name=image_names,
    )

    preds_df = pd.DataFrame(index=image_names, columns=config.classes)
    preds_df.index.name = 'Id'
    preds_df.values[:] = altered_pred
    preds_df.to_csv(PREDICTION_DIR / f'pred-{NAME}.csv')

    subm_df = pd.DataFrame({'Id': image_names,
                            'Label': target2altered(altered_pred)})
    subm_df.to_csv(PREDICTION_DIR / f'subm-{NAME}.csv', index=False)


if __name__ == "__main__":
    if check_dir_not_exist(PREDICTION_DIR):
        transforms = get_transforms(train=False)
        test_data = get_test_data()

        fold_dir = EXPERIMENT_DIR / f'fold_0'
        model_path = get_best_model_path(fold_dir)

        print("Model path", model_path)
        predictor = Predictor(model_path,
                              batch_size=BATCH_SIZE,
                              transform=transforms,
                              device=DEVICE,
                              logits=LOGITS,
                              tta=TTA)

        print("Test predict")
        predict_test_fold(test_data, predictor)

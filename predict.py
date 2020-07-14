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
IMAGE_SIZE = None
LOGITS = True


def predict_test_fold(test_data, predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    image_names = [s['name'] for s in test_data]

    preds = predictor.predict(test_data)
    altered_pred, quality_pred = preds

    np.savez(
        fold_prediction_dir / 'preds.npz',
        altered_pred=altered_pred,
        quality_pred=quality_pred,
        name=image_names,
    )

    preds_df = pd.DataFrame(index=image_names, columns=config.classes)
    preds_df.index.name = 'Id'
    preds_df.values[:] = altered_pred
    preds_df.to_csv(fold_prediction_dir / f'{PREDICTION_DIR.name}.csv')


def blend_folds_submission():
    image_names = None
    altered_pred_lst = []
    for fold in config.folds:
        fold_prediction_path = PREDICTION_DIR / f'fold_{fold}' / 'test' / 'preds.npz'
        if not fold_prediction_path.exists():
            continue
        preds = np.load(fold_prediction_path)
        altered_pred = preds['altered_pred']
        altered_pred = target2altered(altered_pred)
        altered_pred_lst.append(altered_pred)

        names = preds['name']
        if image_names is not None:
            assert np.all(image_names == names)
        image_names = names

    altered_pred = np.stack(altered_pred_lst, axis=0)
    altered_pred = np.mean(altered_pred, axis=0)

    pred_df = pd.DataFrame({'Id': image_names,
                            'Label': altered_pred})
    pred_df.to_csv(PREDICTION_DIR / 'submission.csv', index=False)


if __name__ == "__main__":
    if check_dir_not_exist(PREDICTION_DIR):
        transforms = get_transforms(train=False)
        test_data = get_test_data()

        for fold in config.folds:
            print("Predict fold", fold)
            fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
            model_path = get_best_model_path(fold_dir)

            if model_path is None:
                print("Skip fold", fold)
                continue

            print("Model path", model_path)
            predictor = Predictor(model_path,
                                  batch_size=BATCH_SIZE,
                                  transform=transforms,
                                  device=DEVICE,
                                  logits=LOGITS)

            print("Test predict")
            predict_test_fold(test_data, predictor, fold)

        print("Blend folds predictions")
        blend_folds_submission()

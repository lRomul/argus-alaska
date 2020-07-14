import argparse
import numpy as np
import pandas as pd

from src.utils import get_best_model_path, target2altered, check_dir_not_exist
from src.datasets import get_test_data, get_folds_data
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


def predict_test(predictor):
    test_prediction_dir = PREDICTION_DIR / 'test'
    test_prediction_dir.mkdir(parents=True, exist_ok=True)

    test_data = get_test_data()
    image_names = [s['name'] for s in test_data]

    preds = predictor.predict(test_data)
    altered_pred, quality_pred = preds

    np.savez(
        test_prediction_dir / f'preds-{NAME}.npz',
        altered_pred=altered_pred,
        quality_pred=quality_pred,
        name=image_names,
    )

    preds_df = pd.DataFrame(index=image_names, columns=config.classes)
    preds_df.index.name = 'Id'
    preds_df.values[:] = altered_pred
    preds_df.to_csv(test_prediction_dir / f'pred-{NAME}.csv')

    subm_df = pd.DataFrame({'Id': image_names,
                            'Label': target2altered(altered_pred)})
    subm_df.to_csv(test_prediction_dir / f'subm-{NAME}.csv', index=False)


def predict_validation(predictor):
    val_prediction_dir = PREDICTION_DIR / 'val'
    val_prediction_dir.mkdir(parents=True, exist_ok=True)

    folds_data = get_folds_data()
    val_data = [s for s in folds_data if s['fold'] == 0]

    pred_lst = []
    target_lst = []
    image_names = []
    for cls, trg in config.class2altered.items():
        cls_data = [{'image_path': s[cls]['image_path']} for s in val_data]
        cls_preds = predictor.predict(cls_data)[0]
        cls_trgs = np.array([trg] * len(cls_preds), dtype=np.float32)
        names = np.array([f"{cls}-{s['name']}" for s in val_data])

        pred_lst.append(cls_preds)
        target_lst.append(cls_trgs)
        image_names.append(names)

    pred = np.concatenate(pred_lst)
    target = np.concatenate(target_lst)
    names = np.concatenate(image_names)

    np.savez(
        val_prediction_dir / f'preds-{NAME}.npz',
        altered_pred=pred,
        altered_target=target,
        name=names,
    )


if __name__ == "__main__":
    if check_dir_not_exist(PREDICTION_DIR):
        fold_dir = EXPERIMENT_DIR / f'fold_0'
        model_path = get_best_model_path(fold_dir)

        print("Model path", model_path)
        predictor = Predictor(model_path,
                              batch_size=BATCH_SIZE,
                              transform=get_transforms(train=False),
                              device=DEVICE,
                              logits=LOGITS,
                              tta=TTA)

        print("Test predict")
        predict_test(predictor)

        print("Val predict")
        predict_validation(predictor)

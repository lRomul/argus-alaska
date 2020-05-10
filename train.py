import os
import json
import argparse
from subprocess import Popen

from argus.callbacks import (
    LoggingToFile,
    MonitorCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    LoggingToCSV
)

from torch.utils.data import DataLoader

from src.datasets import AlaskaDataset, get_folds_data
from src.argus_models import AlaskaModel
from src.transforms import get_transforms
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

BATCH_SIZE = 16
TRAIN_EPOCHS = 10
BASE_LR = 0.001
NUM_WORKERS = 16


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('CustomEfficient', {
        'encoder': 'tf_efficientnet_b0_ns',
        'pretrained': True,
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE)}),
    'device': 'cuda',
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()

    model = AlaskaModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = AlaskaDataset(folds_data, train_folds, transform=train_transform)
    val_dataset = AlaskaDataset(folds_data, val_folds, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS * 2)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_loss', max_saves=1),
        ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.6, min_lr=1e-8),
        EarlyStopping(monitor='val_loss', patience=18),
        LoggingToFile(save_dir / 'log.txt'),
        LoggingToCSV(save_dir / 'log.csv')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=TRAIN_EPOCHS,
              callbacks=callbacks)


if __name__ == "__main__":
    if args.fold is None:
        for fold in config.folds:
            command = [
                'python',
                os.path.abspath(__file__),
                '--experiment', args.experiment,
                '--fold', str(fold)
            ]
            pipe = Popen(command)
            pipe.wait()
    elif args.fold in config.folds:
        if not SAVE_DIR.exists():
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

        with open(SAVE_DIR / 'source.py', 'w') as outfile:
            outfile.write(open(__file__).read())

        print("Model params", PARAMS)
        with open(SAVE_DIR / 'params.json', 'w') as outfile:
            json.dump(PARAMS, outfile)

        val_folds = [args.fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{args.fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)

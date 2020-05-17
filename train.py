import os
import json
import argparse
from subprocess import Popen

from argus.callbacks import (
    LoggingToFile,
    MonitorCheckpoint,
    CosineAnnealingLR,
    LoggingToCSV
)

from torch.utils.data import DataLoader

from src.datasets import AlaskaDataset, AlaskaBatchSampler, get_folds_data
from src.argus_models import AlaskaModel
from src.transforms import get_transforms
from src.utils import initialize_amp
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
args = parser.parse_args()

BATCH_SIZE = 32
TRAIN_EPOCH_SIZE = 12000
VAL_EPOCH_SIZE = 3000
TRAIN_EPOCHS = 150
BASE_LR = 1e-4
NUM_WORKERS = 16
USE_AMP = True
DEVICES = ['cuda']


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('CustomEfficient', {
        'encoder': 'tf_efficientnet_b0_ns',
        'num_classes': config.num_unique_targets,
        'pretrained': True,
    }),
    'loss': 'CrossEntropyLoss',
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE)}),
    'prediction_transform': ('Softmax', {'dim': 1}),
    'device': DEVICES[0],
}


def train_fold(save_dir, train_folds, val_folds):
    folds_data = get_folds_data()

    model = AlaskaModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    if USE_AMP:
        initialize_amp(model)
    model.set_device(DEVICES)

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = AlaskaDataset(folds_data, train_folds, transform=train_transform)
    train_sampler = AlaskaBatchSampler(train_dataset, TRAIN_EPOCH_SIZE, BATCH_SIZE, train=True)
    val_dataset = AlaskaDataset(folds_data, val_folds, transform=test_transform)
    val_sampler = AlaskaBatchSampler(val_dataset, VAL_EPOCH_SIZE, BATCH_SIZE * 2, train=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=NUM_WORKERS * 2)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_weighted_auc', max_saves=1),
        CosineAnnealingLR(T_max=TRAIN_EPOCHS, eta_min=1e-6),
        LoggingToFile(save_dir / 'log.txt'),
        LoggingToCSV(save_dir / 'log.csv')
    ]
    metrics = ['weighted_auc', 'altered_accuracy', 'accuracy']

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=TRAIN_EPOCHS,
              callbacks=callbacks,
              metrics=metrics)


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

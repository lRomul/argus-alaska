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
from src.metrics import Accuracy
from src.transforms import get_transforms
from src.utils import initialize_amp, get_best_model_path, load_pretrain_weigths
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--fold', required=False, type=int)
parser.add_argument('--pretrain', default='', type=str)
args = parser.parse_args()

BATCH_SIZE = 44
TRAIN_EPOCHS = 20
BASE_LR = 3e-5
NUM_WORKERS = 2
USE_AMP = True
DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


PARAMS = {
    'nn_module': ('TimmModel', {
        'encoder': 'tf_efficientnet_b5_ns',
        'pretrained': True,
        'drop_rate': 0.4,
        'drop_path_rate': 0.2,
    }),
    'loss': ('AlaskaCrossEntropy', {
        'stegano_weight': 1.0,
        'quality_weight': 0.05,
        'smooth_factor': 0.05,
        'ohem_rate': 1.0
    }),
    'optimizer': ('AdamW', {'lr': get_lr(BASE_LR, BATCH_SIZE)}),
    'device': DEVICES[0],
}


def train_fold(save_dir, train_folds, val_folds, pretrain_dir=''):
    folds_data = get_folds_data()

    model = AlaskaModel(PARAMS)
    model.params['nn_module'][1]['pretrained'] = False

    if pretrain_dir:
        pretrain_path = get_best_model_path(pretrain_dir)
        if pretrain_path is not None:
            print(f'Pretrain model path {pretrain_path}')
            load_pretrain_weigths(model, pretrain_path)
        else:
            print(f"Pretrain model not found in '{pretrain_dir}'")

    if USE_AMP:
        initialize_amp(model)
    model.set_device(DEVICES)

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = AlaskaDataset(folds_data, train_folds, transform=train_transform)
    train_sampler = AlaskaBatchSampler(train_dataset, BATCH_SIZE, train=True)
    val_dataset = AlaskaDataset(folds_data, val_folds, transform=test_transform)
    val_sampler = AlaskaBatchSampler(val_dataset, BATCH_SIZE * 2,
                                     train=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=NUM_WORKERS * 2)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_weighted_auc', max_saves=1),
        CosineAnnealingLR(T_max=TRAIN_EPOCHS, eta_min=get_lr(1e-7, BATCH_SIZE)),
        LoggingToFile(save_dir / 'log.txt'),
        LoggingToCSV(save_dir / 'log.csv')
    ]
    metrics = ['weighted_auc', Accuracy('stegano'), Accuracy('quality')]

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
        save_dir = config.experiments_dir / args.experiment
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'source.py', 'w') as outfile:
            outfile.write(open(__file__).read())

        print("Model params", PARAMS)
        with open(save_dir / 'params.json', 'w') as outfile:
            json.dump(PARAMS, outfile)

        val_folds = [args.fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = save_dir / f'fold_{args.fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")

        pretrain_dir = ''
        if args.pretrain:
            pretrain_dir = config.experiments_dir / args.pretrain / f'fold_{args.fold}'

        train_fold(save_fold_dir, train_folds, val_folds, pretrain_dir)

import os
import json
import torch
import argparse

import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import argus
from argus.callbacks import (
    LoggingToFile,
    MonitorCheckpoint,
    CosineAnnealingLR,
    LambdaLR,
    LoggingToCSV
)

from src.datasets import (
    AlaskaDistributedSampler, AlaskaDataset,
    AlaskaSampler, get_folds_data
)
from src.argus_models import AlaskaModel
from src.metrics import Accuracy
from src.transforms import get_transforms
from src.utils import (
    initialize_amp, initialize_ema,
    get_best_model_path, load_pretrain_weigths
)
from src.ema import EmaMonitorCheckpoint
from src.mixers import EmptyMix
from src import config

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
parser.add_argument('--pretrain', default='', type=str)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

FOLD = 0
BATCH_SIZE = 21
VAL_BATCH_SIZE = 32
ITER_SIZE = 3
TRAIN_EPOCHS = [3, 60, 10]
STAGE = ['warmup', 'train', 'cooldown']
BASE_LR = 3e-4
NUM_WORKERS = 4
USE_AMP = False
USE_EMA = True
DEVICES = ['cuda']

if args.distributed:
    assert DEVICES == ['cuda']
    WORLD_BATCH_SIZE = BATCH_SIZE * dist.get_world_size()
else:
    WORLD_BATCH_SIZE = BATCH_SIZE
print("World batch size:", WORLD_BATCH_SIZE)


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


PARAMS = {
    'nn_module': ('TimmModel', {
        'encoder': 'tf_efficientnet_b3_ns',
        'pretrained': True,
        'drop_rate': 0.3,
        'drop_path_rate': 0.2,
    }),
    'loss': ('AlaskaCrossEntropy', {
        'stegano_weight': 1.0,
        'altered_weight': 0.0,
        'quality_weight': 0.05,
        'smooth_factor': 0.01,
        'ohem_rate': 1.0
    }),
    'optimizer': ('AdamW', {
        'lr': get_lr(BASE_LR, WORLD_BATCH_SIZE)
    }),
    'device': DEVICES[0],
    'iter_size': ITER_SIZE
}


def train_fold(save_dir, train_folds, val_folds,
               local_rank=0, distributed=False, pretrain_dir=''):
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

    if distributed:
        model.nn_module = SyncBatchNorm.convert_sync_batchnorm(model.nn_module)
        model.nn_module = DistributedDataParallel(model.nn_module,
                                                  device_ids=[local_rank])
        if local_rank:
            model.logger.disabled = True
    else:
        model.set_device(DEVICES)

    if USE_EMA:
        initialize_ema(model, decay=0.9999)
        checkpoint = EmaMonitorCheckpoint
    else:
        checkpoint = MonitorCheckpoint

    for epochs, stage in zip(TRAIN_EPOCHS, STAGE):
        test_transform = get_transforms(train=False)

        if stage == 'train':
            train_transform = get_transforms(train=True)
        else:
            train_transform = get_transforms(train=False)

        train_dataset = AlaskaDataset(folds_data, train_folds,
                                      transform=train_transform,
                                      mixer=EmptyMix())
        val_dataset = AlaskaDataset(folds_data, val_folds, transform=test_transform)
        val_sampler = AlaskaSampler(val_dataset, train=False)

        if distributed:
            train_sampler = AlaskaDistributedSampler(train_dataset)
        else:
            train_sampler = AlaskaSampler(train_dataset, train=True)

        train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                  num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
        val_loader = DataLoader(val_dataset, sampler=val_sampler,
                                num_workers=NUM_WORKERS, batch_size=VAL_BATCH_SIZE)

        callbacks = []
        if local_rank == 0:
            callbacks += [
                checkpoint(save_dir, monitor='val_weighted_auc', max_saves=10,
                           file_format=stage + '-model-{epoch:03d}-{monitor:.6f}.pth'),
                LoggingToFile(save_dir / 'log.txt'),
                LoggingToCSV(save_dir / 'log.csv', append=True)
            ]

        if stage == 'train':
            callbacks += [
                CosineAnnealingLR(T_max=epochs,
                                  eta_min=get_lr(3e-6, WORLD_BATCH_SIZE))
            ]
        elif stage == 'warmup':
            warmup_iterations = epochs * (len(train_sampler) / WORLD_BATCH_SIZE)
            callbacks += [
                LambdaLR(lambda x: x / warmup_iterations,
                         step_on_iteration=True)
            ]

        if distributed:
            @argus.callbacks.on_epoch_complete
            def schedule_sampler(state):
                train_sampler.set_epoch(state.epoch + 1)
            callbacks += [schedule_sampler]

        metrics = ['weighted_auc', Accuracy('stegano'), Accuracy('quality')]

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=epochs,
                  callbacks=callbacks,
                  metrics=metrics)


if __name__ == "__main__":
    save_dir = config.experiments_dir / args.experiment
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(save_dir / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    val_folds = [FOLD]
    train_folds = list(set(config.folds) - set(val_folds))
    save_fold_dir = save_dir / f'fold_{FOLD}'

    if args.local_rank == 0:
        print("Model params", PARAMS)
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")

    pretrain_dir = ''
    if args.pretrain:
        pretrain_dir = config.experiments_dir / args.pretrain / f'fold_{FOLD}'

    train_fold(save_fold_dir, train_folds, val_folds,
               args.local_rank, args.distributed, pretrain_dir)

import json

import torch
from torch.utils.data import DataLoader

import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

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
from src.utils import get_best_model_path, load_pretrain_weigths
from src.mixers import EmptyMix
from src import config

EXPERIMENT = 'test_tpu_001'
PRETRAIN = ''
FOLD = 0
BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
ITER_SIZE = 1
TRAIN_EPOCHS = [3, 60, 10]
STAGE = ['warmup', 'train', 'cooldown']
BASE_LR = 3e-4
NUM_WORKERS = 0
NPROCS = 8
WORLD_BATCH_SIZE = BATCH_SIZE * NPROCS
print("World batch size:", WORLD_BATCH_SIZE)


def get_lr(base_lr, batch_size):
    return base_lr * (batch_size / 16)


def model_to_tpu(model, device):
    model.device = device
    model.nn_module = model.nn_module.to(model.device)
    model.xm = xm


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
    'device': 'cpu',
    'iter_size': ITER_SIZE
}


def train_fold(rank, save_dir, train_folds, val_folds, pretrain_dir=''):
    torch.set_default_tensor_type('torch.FloatTensor')

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

    device = xm.xla_device()
    print(f"Rank: {rank}, device: {device}")
    model_to_tpu(model, device)

    local_rank = xm.get_ordinal()
    assert rank == local_rank
    if local_rank:
        model.logger.disabled = True

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

        train_sampler = AlaskaDistributedSampler(train_dataset,
                                                 num_replicas=xm.xrt_world_size(),
                                                 rank=local_rank)

        train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                  num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
        val_loader = DataLoader(val_dataset, sampler=val_sampler,
                                num_workers=NUM_WORKERS, batch_size=VAL_BATCH_SIZE)

        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

        callbacks = []
        if local_rank == 0:
            callbacks += [
                MonitorCheckpoint(save_dir, monitor='val_weighted_auc', max_saves=1,
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
    save_dir = config.experiments_dir / EXPERIMENT
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    val_folds = [FOLD]
    train_folds = list(set(config.folds) - set(val_folds))
    save_fold_dir = save_dir / f'fold_{FOLD}'

    print("Model params", PARAMS)
    print(f"Val folds: {val_folds}, Train folds: {train_folds}")
    print(f"Fold save dir {save_fold_dir}")

    pretrain_dir = ''
    if PRETRAIN:
        pretrain_dir = config.experiments_dir / PRETRAIN / f'fold_{FOLD}'

    xmp.spawn(train_fold,
              args=(save_fold_dir,
                    train_folds,
                    val_folds,
                    pretrain_dir),
              nprocs=NPROCS,
              start_method='fork')

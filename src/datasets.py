import math
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from src import config


class AlaskaDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def get_samples(self):
        samples = []
        for idx in range(len(self.dataset)):
            for cls in config.classes:
                samples.append((idx, cls))
        return samples

    def __iter__(self):
        samples = self.get_samples()
        indices = []
        g = torch.Generator()
        g.manual_seed(self.epoch)
        for idx in torch.randperm(len(samples), generator=g).tolist():
            indices.append(samples[idx])

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image


def get_folds_data(quality=True):
    train_folds_df = pd.read_csv(config.train_folds_path)
    if quality:
        with open(config.quality_json_path) as file:
            quality_dict = json.load(file)
    folds_data = []

    for name, fold in zip(train_folds_df.name, train_folds_df.fold):
        sample = {
            'name': name,
            'fold': fold,
        }
        if quality:
            sample['quality'] = quality_dict[name]

        for cls, trg in config.class2target.items():
            image_path = config.data_dir / cls / name
            sample[cls] = {
                'image_path': str(image_path),
                'target': trg
            }
        folds_data.append(sample)

    return folds_data


def get_test_data():
    test_data = []
    for image_path in sorted(config.test_dir.glob("*")):
        sample = {
            'name': image_path.name,
            'image_path': str(image_path),
        }
        test_data.append(sample)
    return test_data


class AlaskaSampler(Sampler):
    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.epoch_size = len(dataset) * len(config.classes)
        self.train = train

    def get_samples(self):
        samples = []
        for idx in range(len(self.dataset)):
            for cls in config.classes:
                samples.append((idx, cls))
        return samples

    def train_samples(self):
        samples = self.get_samples()
        random.shuffle(samples)
        return samples

    def val_samples(self):
        return self.get_samples()

    def __iter__(self):
        if self.train:
            samples = self.train_samples()
        else:
            samples = self.val_samples()

        return iter(samples)

    def __len__(self):
        return self.epoch_size


class AlaskaDataset(Dataset):
    def __init__(self,
                 data,
                 folds=None,
                 target=True,
                 transform=None,
                 mixer=None):
        self.folds = folds
        self.target = target
        self.transform = transform
        self.mixer = mixer

        self.data = data

        if folds is not None:
            self.data = [s for s in self.data if s['fold'] in folds]

    def __len__(self):
        return len(self.data) * len(config.classes)

    def get_sample(self, idx):
        if isinstance(idx, (tuple, list)):
            name_sample = self.data[idx[0]]
            sample = name_sample[idx[1]]
        else:
            sample = self.data[idx]
            name_sample = sample

        image = load_image(sample['image_path'])

        if not self.target:
            return image

        stegano_target = torch.tensor(sample['target'], dtype=torch.int64)
        quality_target = config.quality2target[name_sample['quality']]
        quality_target = torch.tensor(quality_target, dtype=torch.int64)
        target = stegano_target, quality_target
        return image, target

    def _set_random_seed(self, idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    @torch.no_grad()
    def __getitem__(self, idx):
        self._set_random_seed(idx)
        if not self.target:
            image = self.get_sample(idx)
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            image, target = self.get_sample(idx)
            if self.mixer is not None:
                stegano = idx[1] in config.altered_classes
                if stegano:
                    stegano_sample = image, target
                    cover_sample = self.get_sample((idx[0], 'Cover'))
                    _, stegano_sample = self.mixer(cover_sample, stegano_sample)
                    image, target = stegano_sample
                else:
                    cover_sample = image, target
                    altered_cls = np.random.choice(config.altered_classes)
                    stegano_sample = self.get_sample((idx[0], altered_cls))
                    cover_sample, _ = self.mixer(cover_sample, stegano_sample)
                    image, target = cover_sample

            if self.transform is not None:
                image = self.transform(image)
            return image, target

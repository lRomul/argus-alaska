import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from src import config


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler,
                                                      num_replicas=num_replicas,
                                                      rank=rank,
                                                      shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


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


class AlaskaSampler(BatchSampler):
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
        return len(self.data)

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

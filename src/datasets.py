import cv2
import json
import numpy as np
import pandas as pd
from src import config

import torch
from torch.utils.data import Dataset, BatchSampler


def read_raw_image(path):
    with open(str(path), 'rb') as file:
        buffer = file.read()

    raw_image = np.frombuffer(buffer, dtype='uint8')
    return raw_image


def decode_raw_image(raw_image):
    image = cv2.imdecode(raw_image, cv2.IMREAD_UNCHANGED)
    return image


def get_folds_data(raw_images=False, quality=True):
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
            if raw_images:
                sample[cls]['raw_image'] = read_raw_image(image_path)
        folds_data.append(sample)

    return folds_data


class AlaskaBatchSampler(BatchSampler):
    def __init__(self, dataset, epoch_size, batch_size, train=True, drop_last=True):
        self.dataset = dataset
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.train = train
        self.drop_last = drop_last

    def train_samples(self):
        indexes = np.random.randint(len(self.dataset), size=self.epoch_size)
        classes = np.random.choice(config.classes, size=self.epoch_size)
        return zip(indexes, classes)

    def val_samples(self):
        indexes = np.arange(self.epoch_size)
        repeat_count = (self.epoch_size // len(config.classes)) + 1
        classes = config.classes * repeat_count
        return zip(indexes, classes[:self.epoch_size])

    def __iter__(self):
        batch = []
        if self.train:
            samples = self.train_samples()
        else:
            samples = self.val_samples()

        for sample in samples:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.epoch_size) // self.batch_size
        else:
            return (len(self.epoch_size) + self.batch_size - 1) // self.batch_size


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
        name_sample = self.data[idx[0]]
        sample = name_sample[idx[1]]

        if 'raw_image' in sample:
            image = decode_raw_image(sample['raw_image'])
        else:
            image = cv2.imread(sample['image_path'])

        if not self.target:
            return image

        stegano_target = torch.tensor(sample['target'], dtype=torch.int64)
        quality_target = config.quality2target[name_sample['quality']]
        quality_target = torch.tensor(quality_target, dtype=torch.int64)
        target = stegano_target, quality_target
        return image, target

    @torch.no_grad()
    def __getitem__(self, idx):
        if not self.target:
            image = self.get_sample(idx)
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            image, target = self.get_sample(idx)
            if self.mixer is not None:
                image, target = self.mixer(self, image, target)
            if self.transform is not None:
                image = self.transform(image)
            return image, target

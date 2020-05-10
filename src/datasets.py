import cv2
import time
import random
import numpy as np
import pandas as pd
from src import config

import torch
from torch.utils.data import Dataset


def read_raw_image(path):
    with open(str(path), 'rb') as file:
        buffer = file.read()

    raw_image = np.frombuffer(buffer, dtype='uint8')
    return raw_image


def decode_raw_image(raw_image):
    image = cv2.imdecode(raw_image, cv2.IMREAD_UNCHANGED)
    return image


def get_folds_data(raw_images=False):
    train_folds_df = pd.read_csv(config.train_folds_path)
    folds_data = []

    for name, fold in zip(train_folds_df.name, train_folds_df.fold):
        sample = {
            'name': name,
            'fold': fold,

        }
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
        random_class = np.random.choice(config.classes)
        sample = self.data[idx][random_class]

        if 'raw_image' in sample:
            image = decode_raw_image(sample['raw_image'])
        else:
            image = cv2.imread(sample['image_path'])

        if not self.target:
            return image

        target = torch.tensor(sample['target'], dtype=torch.float32)
        target = target.unsqueeze(0)
        return image, target

    def _set_random_seed(self, idx):
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
                image, target = self.mixer(self, image, target)
            if self.transform is not None:
                image = self.transform(image)
            return image, target

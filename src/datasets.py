import cv2
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


def get_folds_data():
    train_folds_df = pd.read_csv(config.train_folds_path)
    folds_data = []

    for name, fold in zip(train_folds_df.name, train_folds_df.fold):
        sample = {
            'name': name,
            'fold': fold,

        }
        for cls, trg in config.class2target.items():
            image_path = config.data_dir / cls / name
            raw_image = read_raw_image(image_path)
            sample[cls] = {
                'raw_image': raw_image,
                'image_path': image_path,
                'target': trg
            }

        folds_data.append(sample)

    return folds_data

import torch
import random
import numpy as np

import albumentations as alb


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class ImageToTensor:
    def __call__(self, image):
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image


class Albumentations:
    def __init__(self, p=1.0):
        self.augmentation = alb.Compose([
                    alb.VerticalFlip(p=0.5),
                    alb.HorizontalFlip(p=0.5),
                    alb.RandomRotate90(p=0.5)
                ], p=p)

    def __call__(self, image):
        augmented = self.augmentation(image=image)
        image = augmented["image"]
        return image


def get_transforms(train):
    if train:
        transforms = Compose([
            Albumentations(),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            ImageToTensor()
        ])
    return transforms

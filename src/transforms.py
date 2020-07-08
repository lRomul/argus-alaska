import random
import numpy as np
from PIL import Image

from torchvision import transforms


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        image = transform(image)
        return image


class RandomRotate90:
    def __init__(self):
        self.angles = [None, Image.ROTATE_90,
                       Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, img):
        rot = np.random.choice(self.angles)
        if rot is not None:
            img = img.transpose(rot)
        return img


def get_transforms(train):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        trns = transforms.Compose([
            UseWithProb(OneOf([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0)
            ]), prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        trns = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return trns

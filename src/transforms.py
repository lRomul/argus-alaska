import numpy as np
from PIL import Image

from torchvision import transforms


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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        trns = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return trns

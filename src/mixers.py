import torch
import random
import numpy as np

from src import config


def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_cls = np.random.choice(config.classes)
    rnd_image, rnd_target = dataset.get_sample((rnd_idx, rnd_cls))
    return rnd_image, rnd_target


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix:
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, dataset, image, target):
        lam = np.random.beta(self.beta, self.beta)
        rnd_image, rnd_target = get_random_sample(dataset)

        bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)
        if len(image.shape) == 2:
            image[bbx1:bbx2, bby1:bby2] = rnd_image[bbx1:bbx2, bby1:bby2]
        else:
            image[:, bbx1:bbx2, bby1:bby2] = rnd_image[:, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)
                   / (image.shape[-1] * image.shape[-2]))

        target = target, rnd_target, torch.tensor(lam, dtype=torch.float32)
        return image, target


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target = mixer(dataset, image, target)
        return image, target


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target

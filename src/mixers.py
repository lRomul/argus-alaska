import torch
import numpy as np
from PIL import Image


def rand_bbox(size, gamma):
    w, h = size[:2]

    cut_rat = np.sqrt(gamma)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_target(target, other_target, lmb):
    new_target = []
    for trg, rnd_trg in zip(target, other_target):
        new_trg = trg, rnd_trg, torch.tensor(lmb, dtype=torch.float32)
        new_target.append(new_trg)
    return new_target


class BitMix:
    def __init__(self, gamma=0.25):
        self.gamma = gamma

    def __call__(self, cover_sample, stegano_sample):
        cover_img, cover_trg = cover_sample
        cover_img = np.array(cover_img).copy()
        stegano_img, stegano_trg = stegano_sample
        stegano_img = np.array(stegano_img).copy()

        diff = np.sum(cover_img != stegano_img)
        
        if not diff:
            cover_target = mix_target(cover_trg, cover_trg, 1)
            cover_sample = Image.fromarray(cover_img), cover_target
            return cover_sample, cover_sample

        gamma = np.random.uniform(0, self.gamma)

        bbx1, bby1, bbx2, bby2 = rand_bbox(cover_img.shape, gamma)

        cover_crop = cover_img[bbx1:bbx2, bby1:bby2].copy()
        stegano_crop = stegano_img[bbx1:bbx2, bby1:bby2].copy()

        lmb = 1 - np.sum(cover_crop != stegano_crop) / diff

        assert not np.isnan(lmb)

        cover_img[bbx1:bbx2, bby1:bby2] = stegano_crop
        stegano_img[bbx1:bbx2, bby1:bby2] = cover_crop

        cover_target = mix_target(cover_trg, stegano_trg, lmb)
        cover_sample = Image.fromarray(cover_img), cover_target

        stegano_target = mix_target(cover_trg, stegano_trg, 1 - lmb)
        stegano_sample = Image.fromarray(stegano_img), stegano_target
        return cover_sample, stegano_sample


class EmptyMix:
    def __call__(self, cover_sample, stegano_sample):
        cover_img, cover_trg = cover_sample
        cover_img = np.array(cover_img).copy()
        stegano_img, stegano_trg = stegano_sample
        stegano_img = np.array(stegano_img).copy()

        diff = np.sum(cover_img != stegano_img)

        cover_target = mix_target(cover_trg, cover_trg, 1)
        cover_sample = Image.fromarray(cover_img), cover_target

        if not diff:
            return cover_sample, cover_sample

        stegano_target = mix_target(cover_trg, stegano_trg, 0)
        stegano_sample = Image.fromarray(stegano_img), stegano_target

        return cover_sample, stegano_sample


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, cover_sample, stegano_sample):
        mixer = np.random.choice(self.mixers, p=self.p)
        return mixer(cover_sample, stegano_sample)

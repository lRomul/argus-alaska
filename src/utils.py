import re
import torch
import jpegio
import shutil
import numpy as np
from pathlib import Path

from argus import load_model

from src.ema import ModelEma
from src import config


def target2altered(probs):
    altered = probs[:, config.altered_targets]
    if isinstance(altered, torch.Tensor):
        altered = torch.sum(altered, dim=1)
    else:
        altered = np.sum(altered, axis=1)
    return altered


def initialize_amp(model,
                   opt_level='O1',
                   keep_batchnorm_fp32=None,
                   loss_scale='dynamic'):
    from apex import amp
    model.nn_module, model.optimizer = amp.initialize(
        model.nn_module, model.optimizer,
        opt_level=opt_level,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        loss_scale=loss_scale
    )
    model.amp = amp


def initialize_ema(model, decay=0.9999, device='', resume=''):
    model.model_ema = ModelEma(model.get_nn_module(),
                               decay=decay,
                               device=device,
                               resume=resume)


def get_image_quality(image_path):
    jpeg = jpegio.read(str(image_path))
    first_element = jpeg.quant_tables[0][0, 0]
    if first_element == 2:
        return 95
    elif first_element == 3:
        return 90
    elif first_element == 8:
        return 75
    else:
        raise Exception(f"Unknown image quality, quant tables: {jpeg.quant_tables}")


def get_best_model_path(dir_path, return_score=False):
    dir_path = Path(dir_path)
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        return None

    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    if return_score:
        best_score = model_score[-1][1]
        return best_model_path, best_score
    else:
        return best_model_path


def load_pretrain_weigths(model, pretrain_path):
    pretrain_model = load_model(pretrain_path, device=model.device)
    nn_state_dict = pretrain_model.get_nn_module().state_dict()
    model.get_nn_module().load_state_dict(nn_state_dict)
    return model


def check_dir_not_exist(dir_path, remove=False):
    dir_path = Path(dir_path)
    if dir_path.exists():
        if remove:
            shutil.rmtree(dir_path)
            print(f"Folder '{dir_path}' removed")
        else:
            response = input(f"Remove '{dir_path}' (y/n)? ")
            if response.lower() == 'y':
                shutil.rmtree(dir_path)
                print(f"Folder '{dir_path}' removed")
            else:
                return False
    return True

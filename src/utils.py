import torch
import jpegio

from src import config


def target2altered(probs):
    altered = probs[:, config.altered_targets]
    altered = torch.sum(altered, dim=1)
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

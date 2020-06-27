import torch.nn as nn

from timm import create_model

from src.models.classifiers import Classifier


def get_timm_activations():
    import inspect
    import timm

    members = inspect.getmembers(timm.models.layers.activations, inspect.isclass)
    return tuple([m[1] for m in members])


def convert_layers(model, layer_type_old, layer_type_new,
                   convert_weights=False, p_name='', p_module=None):
    activations = get_timm_activations()
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new,
                                                  convert_weights, p_name, p_module)

        if type(module) == layer_type_old and isinstance(p_module, activations):
            model._modules[p_name] = nn.Identity()
            layer_old = module
            layer_new = (layer_type_new(module.num_features)
                         if hasattr(module, 'num_features') else layer_type_new())

            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new
        p_name, p_module = name, module

    return model


class TimmModel(nn.Module):
    def __init__(self,
                 encoder="gluon_resnet50_v1d",
                 inplace_abn=False,
                 **kwargs):
        super().__init__()
        self.model = create_model(encoder, **kwargs)

        if inplace_abn:
            from inplace_abn.abn import InPlaceABN
            convert_layers(self.model, nn.BatchNorm2d,
                           InPlaceABN, convert_weights=True)

        if 'resnet' in encoder:
            self.model.fc = Classifier(self.model.fc.in_features)
        elif 'efficientnet' in encoder:
            self.model.classifier = Classifier(self.model.classifier.in_features)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

    def forward(self, x):
        return self.model(x)

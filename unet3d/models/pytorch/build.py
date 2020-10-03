import os
import math
import torch
import torch.nn

from .classification import resnet
from .classification import custom
from .segmentation import unet
from . import graph


def fetch_model_by_name(model_name, *args, **kwargs):
    try:
        if "res" in model_name.lower():
            return getattr(resnet, model_name)(*args, **kwargs)
        elif "graph" in model_name.lower():
            return getattr(graph, model_name)(*args, **kwargs)
        elif "unet" in model_name.lower():
            return getattr(unet, model_name)(*args, **kwargs)
        else:
            return getattr(custom, model_name)(*args, **kwargs)
    except AttributeError:
        raise ValueError("model name {} not supported".format(model_name))


def build_or_load_model(model_name, model_filename, n_features, n_outputs, n_gpus=0, bias=None, freeze_bias=False,
                        strict=False, **kwargs):
    model = fetch_model_by_name(model_name, n_features=n_features, n_outputs=n_outputs, **kwargs)
    if bias is not None:
        model.fc.bias = torch.nn.Parameter(torch.from_numpy(bias))
    if freeze_bias:
        model.fc.bias.requires_grad_(False)
    if n_gpus > 1:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    elif n_gpus > 0:
        model = model.cuda()
    if os.path.exists(model_filename):
        state_dict = torch.load(model_filename)
        model = load_state_dict(model, state_dict, n_gpus=n_gpus, strict=strict)
    return model


def load_state_dict(model, state_dict, n_gpus, strict=False):
    try:
        if not strict:
            state_dict = match_state_dict_shapes(model.state_dict(), state_dict)
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        if n_gpus > 1:
            if not strict:
                state_dict = match_state_dict_shapes(model.module.state_dict(), state_dict)
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            raise error
    return model


def match_state_dict_shapes(fixed_state_dict, moving_state_dict):
    for key in fixed_state_dict:
        if key in moving_state_dict and fixed_state_dict[key].size() != moving_state_dict[key].size():
            moving_state_dict[key] = match_tensor_sizes(fixed_state_dict[key], moving_state_dict[key])
    return moving_state_dict


def match_tensor_sizes(fixed_tensor, moving_tensor):
    fixed_tensor_size = fixed_tensor.size()
    moving_tensor_size = moving_tensor.size()
    for dim in range(len(moving_tensor_size)):
        if fixed_tensor_size[dim] > moving_tensor_size[dim]:
            moving_tensor = torch.cat([moving_tensor] *
                                      int(math.ceil(fixed_tensor_size[dim] / moving_tensor_size[dim])),
                                      dim=dim)
        if fixed_tensor_size[dim] != moving_tensor_size[dim]:
            moving_tensor = moving_tensor.narrow(dim=dim, start=0, length=fixed_tensor_size[dim])
    return moving_tensor

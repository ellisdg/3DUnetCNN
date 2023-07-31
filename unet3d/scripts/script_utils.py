import monai.losses
import torch

from unet3d.utils.pytorch import losses
from unet3d.utils.utils import load_json
from unet3d.models.build import build_or_load_model


def add_machine_config_to_parser(parser):
    parser.add_argument("--machine_config_filename",
                        help="JSON configuration file containing the number of GPUs and threads that are available "
                             "for model training.",
                        required=False)
    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use during training (default = 1). Warning: using a high number of "
                             "threads can sometimes cause the computer to run out of memory. This setting is "
                             "ignored if machine_config_filename is set.")
    parser.add_argument("--ngpus", default=1, type=int,
                        help="Number of gpus to use for training. This setting is ignored if machine_config_filename is"
                             "set.")
    parser.add_argument("--pin_memory", action="store_true", default=False)


def get_machine_config(namespace):
    if namespace.machine_config_filename:
        print("MP Config: ", namespace.machine_config_filename)
        return load_json(namespace.machine_config_filename)
    else:
        return {"n_workers": namespace.nthreads,
                "n_gpus": namespace.ngpus,
                "pin_memory": namespace.pin_memory}


def build_or_load_model_from_config(config, model_filename, n_gpus, strict=False):
    return build_or_load_model(config["model"].pop("name"), model_filename, n_gpus=n_gpus, **config["model"],
                               strict=strict)


def load_criterion_from_config(config, n_gpus):
    return load_criterion(config['loss'].pop("name"), n_gpus=n_gpus, loss_kwargs=config["loss"])


def load_criterion(criterion_name, n_gpus=0, loss_kwargs=None):
    if loss_kwargs is None:
        loss_kwargs = dict()
    try:
        criterion = getattr(losses, criterion_name)(**loss_kwargs)
    except AttributeError:
        try:
            criterion = getattr(torch.nn, criterion_name)(**loss_kwargs)
        except AttributeError:
            criterion = getattr(monai.losses, criterion_name)(**loss_kwargs)
    if n_gpus > 0:
        criterion.cuda()
    return criterion


def build_optimizer(optimizer_name, model_parameters, **kwargs):
    return getattr(torch.optim, optimizer_name)(params=model_parameters, **kwargs)

import monai.losses
import torch
import nibabel as nib
from torch.utils.data import DataLoader
import numpy as np
import os
import warnings

from unet3d.utils.pytorch import losses
from unet3d.utils.utils import load_json
from unet3d.models.build import build_or_load_model
from unet3d.utils.filenames import load_dataset_class


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


def in_config(string, dictionary, if_not_in_config_return=None):
    return dictionary[string] if string in dictionary else if_not_in_config_return


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


def build_data_loaders_from_config(config, system_config, output_dir):

    dataset_class = load_dataset_class(config["dataset"])

    check_hierarchy(config)

    if in_config("add_contours", config["dataset"], False):
        config["n_outputs"] = config["n_outputs"] * 2

    return build_data_loaders(config, output_dir=output_dir,
                              dataset_class=dataset_class,
                              # system_config args
                              pin_memory=in_config("pin_memory", system_config, False),
                              n_workers=in_config("n_workers", system_config, 1),
                              # training args
                              test_input=in_config("test_input", config["training"], 1),
                              batch_size=in_config("batch_size", config["training"], 1),
                              validation_batch_size=in_config("validation_batch_size", config["training"], 1),
                              prefetch_factor=in_config("prefetch_factor", config["training"], 1))


def build_data_loaders(config, output_dir, dataset_class, metric_to_monitor="val_loss",
                       # from the system config
                       n_workers=1, pin_memory=False,
                       # from config["training"]
                       test_input=1, batch_size=1, validation_batch_size=1, prefetch_factor=1,
                       ):
    if "training" in config["dataset"]:
        training_kwargs = config["dataset"].pop("training")
    else:
        training_kwargs = dict()

    if "validation" in config["dataset"]:
        validation_kwargs = config["dataset"].pop("validation")
    else:
        validation_kwargs = dict()

    # 4. Create datasets
    training_dataset = dataset_class(filenames=config['training_filenames'],
                                     **training_kwargs,
                                     **config["dataset"])

    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 pin_memory=pin_memory,
                                 prefetch_factor=prefetch_factor)

    if test_input:
        test_dataset(test_input, training_dataset, output_dir=os.path.join(output_dir, "data_loader_testing"))

    if 'validation_filenames' not in config:
        warnings.warn(RuntimeWarning("No 'validation_filenames' key found in config. "
                                     "Validation will not be run!"))
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        validation_dataset = dataset_class(filenames=config['validation_filenames'],
                                           **validation_kwargs,
                                           **config["dataset"])
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=validation_batch_size,
                                       shuffle=False,
                                       num_workers=n_workers,
                                       pin_memory=pin_memory,
                                       prefetch_factor=prefetch_factor)
    return training_loader, validation_loader, metric_to_monitor


def build_scheduler_from_config(config, optimizer):
    if "scheduler" not in config:
        scheduler = None
    else:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["scheduler"].pop("name"))
        scheduler = scheduler_class(optimizer, **config["scheduler"])
    return scheduler


def test_dataset(n_test_cases, training_dataset, output_dir):
    """
    param n_test_cases: integer with the number of inputs from the generator to write to file. 0, False, or None will
    """
    os.makedirs(output_dir, exist_ok=True)
    for index in range(n_test_cases):
        x, y = training_dataset[index]
        affine = x.affine
        x = np.moveaxis(x.numpy(), 0, -1).squeeze()
        x_image = nib.Nifti1Image(x, affine=affine)
        x_image.to_filename(os.path.join(output_dir, "input_test_{}.nii.gz".format(index)))
        if len(y.shape) >= 3:
            y = np.moveaxis(y.numpy(), 0, -1)
            y_image = nib.Nifti1Image(y.squeeze(), affine=affine)
            y_image.to_filename(os.path.join(output_dir, "target_test_{}.nii.gz".format(index)))


def check_hierarchy(config):
    if in_config("labels", config["dataset"]) and in_config("use_label_hierarchy", config["dataset"]):
        config["dataset"].pop("use_label_hierarchy")
        labels = config["dataset"].pop("labels")
        new_labels = list()
        while len(labels):
            new_labels.append(labels)
            labels = labels[1:]
        config["dataset"]["labels"] = new_labels
    if "use_label_hierarchy" in config["dataset"]:
        # Remove this flag aas it has already been accounted for
        config["dataset"].pop("use_label_hierarchy")

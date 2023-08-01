import monai.losses
import torch
import nibabel as nib
from monai.data import DataLoader
import numpy as np
import os
import warnings
from copy import deepcopy

from unet3d.utils.pytorch import losses
from unet3d.utils.utils import load_json, dump_json
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


def build_data_loaders_from_config(config, system_config, output_dir, dataset_class):

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

    # Create datasets
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
        write_dataset_examples(test_input, training_dataset, output_dir=os.path.join(output_dir, "data_loader_testing"))

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


def build_inference_loaders_from_config(config, dataset_class, system_config):
    inference_dataloaders = list()
    inference_dataset_kwargs = in_config("inference",
                                         in_config("dataset", config["inference"], dict()),
                                         dict())
    for key in config:
        if "_filenames" in key and key.split("_filenames")[0] not in ("training",):
            name = key.split("_filenames")[0]
            print("Found inference filenames: {} (n={})".format(name, len(config[key])))
            inference_dataloaders.append([build_inference_loader(filenames=config[key],
                                                                 dataset_class=dataset_class,
                                                                 dataset_kwargs=config["dataset"],
                                                                 inference_kwargs=inference_dataset_kwargs,
                                                                 batch_size=in_config("batch_size",
                                                                                      config["inference"], 1),
                                                                 num_workers=in_config("n_workers", system_config, 1),
                                                                 pin_memory=in_config("pin_memory", system_config,
                                                                                      False),
                                                                 prefetch_factor=in_config("prefetch_factor",
                                                                                           config["inference"], 1)),
                                          name])
    return inference_dataloaders


def build_inference_loader(filenames, dataset_class, inference_kwargs, dataset_kwargs,
                           batch_size=1, num_workers=1, pin_memory=False, prefetch_factor=1):
    _dataset = dataset_class(filenames=filenames,
                             **inference_kwargs,
                             **dataset_kwargs)
    _loader = DataLoader(_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor)
    return _loader


def build_scheduler_from_config(config, optimizer):
    if "scheduler" not in config:
        scheduler = None
    else:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["scheduler"].pop("name"))
        scheduler = scheduler_class(optimizer, **config["scheduler"])
    return scheduler


def write_dataset_examples(n_test_cases, training_dataset, output_dir):
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
    label_hierarchy = False
    if in_config("labels", config["dataset"]) and in_config("setup_label_hierarchy", config["dataset"]):
        config["dataset"].pop("setup_label_hierarchy")
        labels = config["dataset"].pop("labels")
        new_labels = list()
        while len(labels):
            new_labels.append(labels)
            labels = labels[1:]
        config["dataset"]["labels"] = new_labels
        label_hierarchy = True
    if "setup_label_hierarchy" in config["dataset"]:
        # Remove this flag as it has already been accounted for
        config["dataset"].pop("setup_label_hierarchy")
    return label_hierarchy


def setup_cross_validation(config, work_dir, n_folds, random_seed=25):
    filenames = config["training_filenames"]
    np.random.seed(random_seed)
    np.random.shuffle(filenames)
    val_step = int(len(filenames) / n_folds)
    fold_configs = list()
    for fold_i in range(0, n_folds):
        val_start = val_step * fold_i
        if (fold_i + 1) == n_folds:
            training_filenames = filenames[:val_start] + filenames[(val_start + val_step):]
            validation_filenames = filenames[val_start:(val_start + val_step)]
        else:
            training_filenames = filenames[:val_start]
            validation_filenames = filenames[val_start:]
        assert not np.any(np.isin(validation_filenames, training_filenames))
        assert (len(validation_filenames) + len(training_filenames)) == len(filenames)
        fold = fold_i + 1
        config_filename = os.path.join(work_dir, "fold{}.json".format(fold))
        fold_config = deepcopy(config)
        fold_config["training_filenames"] = training_filenames
        fold_config["validation_filenames"] = validation_filenames
        dump_json(fold_config, config_filename)
        fold_configs.append([fold_config, config_filename])
    return fold_configs


def load_filenames_from_config(config):
    for key in config:
        if "_filenames" in key:
            config[key] = load_filenames(config[key])


def load_filenames(filenames):
    if type(filenames) == list:
        return filenames
    elif ".npy" in filenames:
        return np.load(filenames, allow_pickle=True).tolist()
    else:
        raise (RuntimeError("Could not load filenames: {}".format(filenames)))

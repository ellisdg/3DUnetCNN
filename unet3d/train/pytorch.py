import os
import shutil
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn

from ..models.pytorch.build import build_or_load_model
from ..utils.pytorch import WholeBrainCIFTI2DenseScalarDataset
from .pytorch_training_utils import epoch_training, epoch_validatation, collate_flatten, collate_5d_flatten
from ..utils.pytorch import functions
from ..utils.utils import in_config


def build_optimizer(optimizer_name, model_parameters, learning_rate=1e-4):
    return getattr(torch.optim, optimizer_name)(model_parameters, lr=learning_rate)


def run_pytorch_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                         n_workers=1, max_queue_size=5, model_name='resnet_34', n_gpus=1, regularized=False,
                         sequence_class=WholeBrainCIFTI2DenseScalarDataset, directory=None, test_input=1,
                         metric_to_monitor="loss", model_metrics=(), bias=None, pin_memory=False, **unused_args):
    """
    :param test_input: integer with the number of inputs from the generator to write to file. 0, False, or None will
    write no inputs to file.
    :param sequence_class: class to use for the generator sequence
    :param model_name:
    :param verbose:
    :param use_multiprocessing:
    :param n_workers:
    :param max_queue_size:
    :param config:
    :param model_filename:
    :param training_log_filename:
    :param metric_to_monitor:
    :param model_metrics:
    :return:

    Anything that directly affects the training results should go into the config file. Other specifications such as
    multiprocessing optimization should be arguments to this function, as these arguments affect the computation time,
    but the results should not vary based on whether multiprocessing is used or not.
    """
    window = np.asarray(config['window'])
    spacing = np.asarray(config['spacing']) if 'spacing' in config else None
    if 'model_name' in config:
        model_name = config['model_name']

    if "n_outputs" in config:
        n_outputs = config['n_outputs']
    else:
        n_outputs = len(np.concatenate(config['metric_names']))

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
        if "input_shape" not in config["model_kwargs"]:
            # assume that the model will take in the whole image window
            config["model_kwargs"]["input_shape"] = window
    else:
        model_kwargs = dict()

    model = build_or_load_model(model_name, model_filename, n_features=config["n_features"], n_outputs=n_outputs,
                                freeze_bias=in_config("freeze_bias", config, False),
                                bias=bias, n_gpus=n_gpus, **model_kwargs)
    model.train()

    criterion = load_criterion(config['loss'], n_gpus=n_gpus)

    if "weights" in config and config["weights"] is not None:
        criterion = functions.WeightedLoss(torch.tensor(config["weights"]), criterion)

    optimizer_kwargs = dict()
    if "initial_learning_rate" in config:
        optimizer_kwargs["learning_rate"] = config["initial_learning_rate"]

    optimizer = build_optimizer(optimizer_name=config["optimizer"],
                                model_parameters=model.parameters(),
                                **optimizer_kwargs)

    sequence_kwargs = in_config("sequence_kwargs", config, dict())

    if "flatten_y" in config and config["flatten_y"]:
        collate_fn = collate_flatten
    elif "collate_fn" in config and config["collate_fn"] == "collate_5d_flatten":
        collate_fn = collate_5d_flatten
    else:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # 4. Create datasets
    training_dataset = sequence_class(filenames=config['training_filenames'],
                                      flip=in_config('flip', config, False),
                                      reorder=config['reorder'],
                                      window=window,
                                      spacing=spacing,
                                      points_per_subject=in_config('points_per_subject', config, 1),
                                      surface_names=in_config('surface_names', config, None),
                                      metric_names=in_config('metric_names', config, None),
                                      base_directory=directory,
                                      subject_ids=config["training"],
                                      iterations_per_epoch=in_config("iterations_per_epoch", config, 1),
                                      **in_config("additional_training_args", config, dict()),
                                      **sequence_kwargs)

    training_loader = DataLoader(training_dataset,
                                 batch_size=config["batch_size"] // in_config('points_per_subject', config, 1),
                                 shuffle=True,
                                 num_workers=n_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)

    if test_input:
        for index in range(test_input):
            x, y = training_dataset[index]
            if not isinstance(x, np.ndarray):
                x = x.numpy()
                y = y.numpy()
            x = np.moveaxis(x, 0, -1)
            x_image = nib.Nifti1Image(x.squeeze(), affine=np.diag(np.ones(4)))
            x_image.to_filename(model_filename.replace(".h5",
                                                       "_input_test_{}.nii.gz".format(index)))
            if len(y.shape) >= 3:
                y = np.moveaxis(y, 0, -1)
                y_image = nib.Nifti1Image(y.squeeze(), affine=np.diag(np.ones(4)))
                y_image.to_filename(model_filename.replace(".h5",
                                                           "_target_test_{}.nii.gz".format(index)))

    if 'skip_validation' in config and config['skip_validation']:
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        validation_dataset = sequence_class(filenames=config['validation_filenames'],
                                            flip=False,
                                            reorder=config['reorder'],
                                            window=window,
                                            spacing=spacing,
                                            points_per_subject=in_config('validation_points_per_subject', config, 1),
                                            surface_names=in_config('surface_names', config, None),
                                            metric_names=in_config('metric_names', config, None),
                                            **sequence_kwargs,
                                            **in_config("additional_validation_args", config, dict()))
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=config["validation_batch_size"] // in_config("points_per_subject",
                                                                                               config, 1),
                                       shuffle=False,
                                       num_workers=n_workers,
                                       collate_fn=collate_fn,
                                       pin_memory=pin_memory)

    train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=config["n_epochs"], verbose=bool(verbose),
          training_loader=training_loader, validation_loader=validation_loader, model_filename=model_filename,
          training_log_filename=training_log_filename,
          metric_to_monitor=metric_to_monitor,
          early_stopping_patience=in_config("early_stopping_patience", config),
          save_best=in_config("save_best", config, False),
          learning_rate_decay_patience=in_config("decay_patience", config),
          regularized=in_config("regularized", config, regularized),
          n_gpus=n_gpus,
          vae=in_config("vae", config, False),
          decay_factor=in_config("decay_factor", config),
          min_lr=in_config("min_learning_rate", config),
          learning_rate_decay_step_size=in_config("decay_step_size", config),
          save_every_n_epochs=in_config("save_every_n_epochs", config),
          save_last_n_models=in_config("save_last_n_models", config, 1))


def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, metric_to_monitor="val_loss", early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best=False, n_gpus=1, verbose=True, regularized=False,
          vae=False, decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, save_every_n_epochs=None,
          save_last_n_models=1):
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
        start_epoch = int(training_log[-1][0]) + 1
    else:
        start_epoch = 0
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    if learning_rate_decay_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=learning_rate_decay_patience,
                                                               verbose=verbose, factor=decay_factor, min_lr=min_lr)
    elif learning_rate_decay_step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=learning_rate_decay_step_size,
                                                    gamma=decay_factor, last_epoch=-1)
        # Setting the last epoch to anything other than -1 requires the optimizer that was previously used.
        # Since I don't save the optimizer, I have to manually step the scheduler the number of epochs that have already
        # been completed. Stepping the scheduler before the optimizer raises a warning, so I have added the below
        # code to step the scheduler and catch the UserWarning that would normally be thrown.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(start_epoch):
                scheduler.step()
    else:
        scheduler = None

    for epoch in range(start_epoch, n_epochs):

        # early stopping
        if (training_log and early_stopping_patience
            and np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
                <= len(training_log) - early_stopping_patience):
            print("Early stopping patience {} has been reached.".format(early_stopping_patience))
            break

        # train the model
        loss = epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch, n_gpus=n_gpus,
                              regularized=regularized, vae=vae)
        try:
            training_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(training_loader.dataset)))
        # predict validation data
        if validation_loader:
            val_loss = epoch_validatation(validation_loader, model, criterion, n_gpus=n_gpus, regularized=regularized,
                                          vae=vae)
        else:
            val_loss = None

        # update the training log
        training_log.append([epoch, loss, get_lr(optimizer), val_loss])
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)
        min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

        # check loss and decay
        if scheduler:
            if validation_loader and scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            elif scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(loss)
            else:
                scheduler.step()

        # save model
        torch.save(model.state_dict(), model_filename)
        if save_best and min_epoch == len(training_log) - 1:
            best_filename = model_filename.replace(".h5", "_best.h5")
            forced_copy(model_filename, best_filename)

        if save_every_n_epochs and (epoch % save_every_n_epochs) == 0:
            epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
            forced_copy(model_filename, epoch_filename)

        if save_last_n_models > 1:
            if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
                to_delete = model_filename.replace(".h5", "_{}.h5".format(epoch - save_last_n_models))
                remove_file(to_delete)
            epoch_filename = model_filename.replace(".h5", "_{}.h5".format(epoch))
            forced_copy(model_filename, epoch_filename)


def forced_copy(source, target):
    remove_file(target)
    shutil.copy(source, target)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


def load_criterion(criterion_name, n_gpus=0):
    try:
        criterion = getattr(functions, criterion_name)
    except AttributeError:
        criterion = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            criterion.cuda()
    return criterion

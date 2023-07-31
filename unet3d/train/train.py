import os
import shutil
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn

from ..utils.pytorch.dataset import WholeVolumeSegmentationDataset
from .training_utils import epoch_training, epoch_validatation


def start_training(config, model, training_log_filename, batch_size, validation_batch_size, model_filename, criterion,
                   optimizer,
                   n_workers=1, n_gpus=1,
                   sequence_class=WholeVolumeSegmentationDataset, test_input=1,
                   metric_to_monitor="loss", pin_memory=False, amp=False, n_epochs=1000,
                   prefetch_factor=1, scheduler_name=None, scheduler_kwargs=None, samples_per_epoch=None,
                   save_best=False, early_stopping_patience=None, save_every_n_epochs=None, save_last_n_models=None,
                   skip_validation=False):
    """
    This function instantiates the training and validation datasets and then runs the training.
    Ultimately, I would like to simplify the functions such that each one has a unique job, such as: read the config,
    instantiate and then return the datasets, build or load the model, and finally run the training.

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

    model.train()

    if "training" in config["dataset"]:
        training_kwargs = config["dataset"].pop("training")
    else:
        training_kwargs = dict()

    if "validation" in config["dataset"]:
        validation_kwargs = config["dataset"].pop("validation")
    else:
        validation_kwargs = dict()

    # 4. Create datasets
    training_dataset = sequence_class(filenames=config['training_filenames'],
                                      **training_kwargs,
                                      **config["dataset"])

    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_workers,
                                 pin_memory=pin_memory,
                                 prefetch_factor=prefetch_factor)

    if test_input:
        for index in range(test_input):
            x, y = training_dataset[index]
            if not isinstance(x, np.ndarray):
                x = x.numpy()
                y = y.numpy()
            x = np.moveaxis(x, 0, -1)
            x_image = nib.Nifti1Image(x.squeeze(), affine=np.diag(np.ones(4)))
            x_image.to_filename(model_filename.split(".")[0] + "_input_test_{}.nii.gz".format(index))
            if len(y.shape) >= 3:
                y = np.moveaxis(y, 0, -1)
                y_image = nib.Nifti1Image(y.squeeze(), affine=np.diag(np.ones(4)))
                y_image.to_filename(model_filename.split(".")[0] + "_target_test_{}.nii.gz".format(index))

    if skip_validation:
        validation_loader = None
        metric_to_monitor = "loss"
    else:
        validation_dataset = sequence_class(filenames=config['validation_filenames'],
                                            **validation_kwargs,
                                            **config["dataset"])
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=validation_batch_size,
                                       shuffle=False,
                                       num_workers=n_workers,
                                       pin_memory=pin_memory,
                                       prefetch_factor=prefetch_factor)

    train(model=model, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs,
          training_loader=training_loader, validation_loader=validation_loader, model_filename=model_filename,
          training_log_filename=training_log_filename,
          metric_to_monitor=metric_to_monitor,
          early_stopping_patience=early_stopping_patience,
          save_best=save_best,
          n_gpus=n_gpus,
          save_every_n_epochs=save_every_n_epochs,
          save_last_n_models=save_last_n_models,
          amp=amp,
          scheduler_name=scheduler_name,
          scheduler_kwargs=scheduler_kwargs,
          samples_per_epoch=samples_per_epoch)


def train(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
          model_filename, metric_to_monitor="val_loss", early_stopping_patience=None,
          save_best=False, n_gpus=1, save_every_n_epochs=None,
          save_last_n_models=None, amp=False, scheduler_name=None, scheduler_kwargs=None,
          samples_per_epoch=None):
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
        start_epoch = int(training_log[-1][0]) + 1
    else:
        start_epoch = 1
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    if scheduler_name is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        if start_epoch > 1:
            # step the scheduler and optimizer to account for previous epochs
            for i in range(start_epoch, 1):
                optimizer.step()
                if scheduler_class == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    metric = np.asarray(training_log)[i - 1, training_log_header.index(metric_to_monitor)]
                    scheduler.step(metric)
                else:
                    scheduler.step()
    else:
        scheduler = None

    if amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch, n_epochs):
        # early stopping
        if training_log:
            metric = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)]
        if (training_log and early_stopping_patience
                and metric.argmin() <= len(training_log) - early_stopping_patience):
            print("Early stopping patience {} has been reached.".format(early_stopping_patience))
            break

        if training_log and np.isnan(metric[-1]):
            print("Stopping as invalid results were returned.")
            break

        # train the model
        loss = epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch, n_gpus=n_gpus,
                              scaler=scaler, samples_per_epoch=samples_per_epoch)
        try:
            training_loader.dataset.on_epoch_end()
        except AttributeError:
            warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                type(training_loader.dataset)))

        # Clear the cache from the GPUs
        if n_gpus:
            torch.cuda.empty_cache()

        # predict validation data
        if validation_loader:
            val_loss = epoch_validatation(validation_loader, model, criterion, n_gpus=n_gpus,
                                          use_amp=scaler is not None)
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
        if n_gpus > 1:
            torch.save(model.module.state_dict(), model_filename)
        else:
            torch.save(model.state_dict(), model_filename)
        if save_best and min_epoch == len(training_log) - 1:
            best_filename = append_to_filename(model_filename, "best")
            forced_copy(model_filename, best_filename)

        if save_every_n_epochs and (epoch % save_every_n_epochs) == 0:
            epoch_filename = append_to_filename(model_filename, epoch)
            forced_copy(model_filename, epoch_filename)

        if save_last_n_models is not None and save_last_n_models > 1:
            if not save_every_n_epochs or not ((epoch - save_last_n_models) % save_every_n_epochs) == 0:
                to_delete = append_to_filename(model_filename, epoch - save_last_n_models)
                remove_file(to_delete)
            epoch_filename = append_to_filename(model_filename, epoch)
            forced_copy(model_filename, epoch_filename)


def forced_copy(source, target):
    remove_file(target)
    shutil.copy(source, target)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def append_to_filename(filename, what_to_append):
    dirname, basename = os.path.split(filename)
    basename_no_extension, extension = basename.split(".")
    return os.path.join(dirname, "{}_{}.{}".format(basename_no_extension, what_to_append, extension))


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))



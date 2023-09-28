import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn

from .training_utils import epoch_training, epoch_validation


def run_training(model, optimizer, criterion, n_epochs, training_loader, validation_loader, training_log_filename,
                 model_filename, metric_to_monitor="val_loss", early_stopping_patience=None, save_best=False, n_gpus=1,
                 save_every_n_epochs=None, save_last_n_models=None, amp=False, scheduler=None, samples_per_epoch=None,
                 inferer=None, training_iterations_per_epoch=1):
    training_log = list()
    if os.path.exists(training_log_filename):
        training_log.extend(pd.read_csv(training_log_filename).values)
        start_epoch = int(training_log[-1][0]) + 1
    else:
        start_epoch = 1
    training_log_header = ["epoch", "loss", "lr", "val_loss"]

    if scheduler is not None and start_epoch > 1:
        # step the scheduler and optimizer to account for previous epochs
        for i in range(1, start_epoch):
            optimizer.step()
            if scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                metric = np.asarray(training_log)[i - 1, training_log_header.index(metric_to_monitor)]
                scheduler.step(metric)
            else:
                scheduler.step()

    if amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch, n_epochs+1):
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
        losses = list()
        for i in range(training_iterations_per_epoch):
            losses.append(epoch_training(training_loader, model, criterion, optimizer=optimizer, epoch=epoch,
                                         n_gpus=n_gpus, scaler=scaler, samples_per_epoch=samples_per_epoch,
                                         iteration=i+1))
        loss = np.mean(losses)

        # Clear the cache from the GPUs
        if n_gpus:
            torch.cuda.empty_cache()

        # predict validation data
        if validation_loader:
            val_loss = epoch_validation(validation_loader, model, criterion, n_gpus=n_gpus, use_amp=scaler is not None,
                                        inferer=inferer)
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

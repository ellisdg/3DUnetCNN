#!/usr/bin/env python
import argparse
import os
import warnings

from unet3d.train import run_training
from unet3d.utils.utils import load_json
from unet3d.predict import volumetric_predictions
from unet3d.utils.filenames import load_dataset_class
from unet3d.scripts.script_utils import (get_machine_config, add_machine_config_to_parser, build_optimizer,
                                         build_or_load_model_from_config, load_criterion_from_config, in_config,
                                         build_data_loaders_from_config, build_scheduler_from_config,
                                         setup_cross_validation, load_filenames_from_config,
                                         build_inference_loaders_from_config, check_hierarchy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True,
                        help="JSON configuration file specifying the parameters for model training.")
    parser.add_argument("--output_dir", required=False,
                        help="Output directory where all the outputs will be saved. "
                             "Defaults to the directory of the configuration file.")
    parser.add_argument("--setup_crossval_only", action="store_true", default=False,
                        help="Only write the cross-validation configuration files. "
                             "If selected, training will not be run. Instead the filenames will be split into "
                             "folds and modified configuration files will be written to the working directory. "
                             "This is useful if you want to submit training folds to an HPC scheduler system.")
    parser.add_argument("--pretrained_model_filename",
                        help="If this filename exists prior to training, the model will be loaded from the filename. "
                             "Default is '{output_dir}/{config_basename}/model.pth'.",
                        required=False)
    parser.add_argument("--training_log_filename",
                        help="CSV filename to save the to save the training and validation results for each epoch. "
                             "Default is '{output_dir}/{config_basename}/training_log.csv'",
                        required=False)
    parser.add_argument("--batch_size", help="Override the batch size from the config file.", type=int)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Raises an error if a training file is not found. The default is to silently skip"
                             "any training files that cannot be found. Use this flag to debug the config for finding"
                             "the data.")
    add_machine_config_to_parser(parser)
    parser.add_argument("--n_examples", type=int, default=0,
                        help="Number of example input/output pairs to write to file for debugging purposes. "
                             "(default = 1)")
    args = parser.parse_args()

    return args


def run(config_filename, output_dir, namespace):
    print("Config: ", config_filename)
    config = load_json(config_filename)
    load_filenames_from_config(config)

    work_dir = os.path.join(output_dir, os.path.basename(config_filename).split(".")[0])
    print("Work Dir:", work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if "cross_validation" in config:
        # call parent function through each fold of the training set
        cross_validation_config = config.pop("cross_validation")
        for _config, _config_filename in setup_cross_validation(config,
                                                                work_dir=work_dir,
                                                                n_folds=in_config("n_folds",
                                                                                  cross_validation_config,
                                                                                  5),
                                                                random_seed=in_config("random_seed",
                                                                                      cross_validation_config,
                                                                                      25)):
            if not namespace.setup_crossval_only:
                print("Running cross validation fold:", _config_filename)
                run(_config_filename, work_dir, namespace)
            else:
                print("Setup cross validation fold:", _config_filename)
    else:
        # run the training
        system_config = get_machine_config(namespace)

        # set verbosity
        if namespace.debug:
            if "dataset" not in config:
                config["dataset"] = dict()
            config["dataset"]["verbose"] = namespace.debug
            warnings.filterwarnings('error')

        # Override the batch size from the config file
        if namespace.batch_size:
            warnings.warn(RuntimeWarning('Overwriting the batch size from the configuration file (batch_size={}) to '
                                         'batch_size={}'.format(config["training"]["batch_size"], namespace.batch_size)))
            config["training"]["batch_size"] = namespace.batch_size

        model_filename = os.path.join(work_dir, "model.pth")
        print("Model: ", model_filename)

        if namespace.training_log_filename:
            training_log_filename = namespace.model_filename
        else:
            training_log_filename = os.path.join(work_dir, "training_log.csv")
        print("Log: ", training_log_filename)

        label_hierarchy = check_hierarchy(config)
        dataset_class = load_dataset_class(config["dataset"], cache_dir=os.path.join(work_dir, "cache"))
        training_loader, validation_loader, metric_to_monitor = build_data_loaders_from_config(config,
                                                                                               system_config,
                                                                                               work_dir,
                                                                                               dataset_class)
        pretrained = namespace.pretrained_model_filename
        if pretrained:
            pretrained = os.path.abspath(pretrained)
        model = build_or_load_model_from_config(config,
                                                pretrained,
                                                system_config["n_gpus"])
        criterion = load_criterion_from_config(config, n_gpus=system_config["n_gpus"])
        optimizer = build_optimizer(optimizer_name=config["optimizer"].pop("name"),
                                    model_parameters=model.parameters(),
                                    **config["optimizer"])
        scheduler = build_scheduler_from_config(config, optimizer)

        run_training(model=model.train(), optimizer=optimizer, criterion=criterion,
                     n_epochs=in_config("n_epochs", config["training"], 1000),
                     training_loader=training_loader, validation_loader=validation_loader,
                     model_filename=model_filename,
                     training_log_filename=training_log_filename,
                     metric_to_monitor=metric_to_monitor,
                     early_stopping_patience=in_config("early_stopping_patience", config["training"], None),
                     save_best=in_config("save_best", config["training"], True),
                     n_gpus=system_config["n_gpus"],
                     save_every_n_epochs=in_config("save_every_n_epochs", config["training"], None),
                     save_last_n_models=in_config("save_last_n_models", config["training"], None),
                     amp=in_config("amp", config["training"], None),
                     scheduler=scheduler,
                     samples_per_epoch=in_config("samples_per_epoch", config["training"], None))

        for _dataloader, _name in build_inference_loaders_from_config(config,
                                                                      dataset_class=dataset_class,
                                                                      system_config=system_config):
            prediction_dir = os.path.join(work_dir, _name)
            os.makedirs(prediction_dir, exist_ok=True)
            volumetric_predictions(model=model,
                                   dataloader=_dataloader,
                                   prediction_dir=prediction_dir,
                                   interpolation="linear",  # TODO
                                   segmentation=False,  # TODO
                                   segmentation_labels=None,  # TODO: segment the labels
                                   threshold=0.5,
                                   sum_then_threshold=False,
                                   label_hierarchy=label_hierarchy)


def main():
    namespace = parse_args()
    if namespace.output_dir:
        output_dir = os.path.abspath(namespace.output_dir)
    else:
        output_dir = os.path.dirname(os.path.abspath(namespace.config_filename))
    run(namespace.config_filename, output_dir, namespace)


if __name__ == '__main__':
    main()

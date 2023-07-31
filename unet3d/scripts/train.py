#!/usr/bin/env python
import argparse
import os
from unet3d.train import run_training
from unet3d.utils.utils import load_json
from unet3d.scripts.predict import format_parser as format_prediction_args
from unet3d.scripts.predict import run_inference
from unet3d.scripts.script_utils import (get_machine_config, add_machine_config_to_parser, build_optimizer,
                                         build_or_load_model_from_config, load_criterion_from_config, in_config,
                                         build_data_loaders_from_config, build_scheduler_from_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True,
                        help="JSON configuration file specifying the parameters for model training.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory where all the outputs will be saved.")
    parser.add_argument("--model_filename",
                        help="Location to save the model during and after training. If this filename exists "
                             "prior to training, the model will be loaded from the filename. "
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
    parser.add_argument("--n_examples", type=int, default=1,
                        help="Number of example input/output pairs to write to file for debugging purposes. "
                             "(default = 1)")
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    args = parser.parse_args()

    return args


def main():
    namespace = parse_args()

    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)

    print("Model: ", namespace.model_filename)
    print("Log: ", namespace.training_log_filename)
    system_config = get_machine_config(namespace)

    # set verbosity
    if namespace.debug:
        if "dataset" not in config:
            config["dataset"] = dict()
        config["dataset"]["verbose"] = namespace.debug

        import warnings
        warnings.filterwarnings('error')

    # Override the batch size from the config file
    if namespace.batch_size:
        config["batch_size"] = namespace.batch_size

    work_dir = os.path.join(namespace.output_dir, os.path.basename(namespace.config_filename).split(".")[0])

    if namespace.model_filename:
        model_filename = namespace.model_filename
    else:
        model_filename = os.path.join(work_dir, "model.pth")

    if namespace.training_log_filename:
        training_log_filename = namespace.model_filename
    else:
        training_log_filename = os.path.join(work_dir, "training_log.csv")

    training_loader, validation_loader, metric_to_monitor = build_data_loaders_from_config(config,
                                                                                           system_config,
                                                                                           work_dir)
    model = build_or_load_model_from_config(config, os.path.abspath(namespace.model_filename), system_config["n_gpus"])
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

    if namespace.sub_command == "predict":
        run_inference(namespace)


if __name__ == '__main__':
    main()

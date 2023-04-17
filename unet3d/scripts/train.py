#!/usr/bin/env python
import argparse
import os
import numpy as np
from unet3d.train import start_training
from unet3d.utils.filenames import generate_filenames, load_dataset_class
from unet3d.utils.utils import load_json, in_config, dump_json
from unet3d.scripts.predict import format_parser as format_prediction_args
from unet3d.scripts.predict import run_inference
from unet3d.scripts.script_utils import (get_machine_config, add_machine_config_to_parser,
                                         build_or_load_model_from_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True,
                        help="JSON configuration file specifying the parameters for model training.")
    parser.add_argument("--model_filename",
                        help="Location to save the model during and after training. If this filename exists "
                             "prior to training, the model will be loaded from the filename.",
                        required=True)
    parser.add_argument("--training_log_filename",
                        help="CSV filename to save the to save the training and validation results for each epoch.",
                        required=True)
    parser.add_argument("--batch_size", help="Override the batch size from the config file.", type=int)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Raises an error if a training file is not found. The default is to silently skip"
                             "any training files that cannot be found. Use this flag to debug the config for finding"
                             "the data.")
    add_machine_config_to_parser(parser)
    parser.add_argument("--n_examples", type=int, default=0,
                        help="Number of example input/output pairs to write to file for debugging purposes. "
                             "(default = 0)")
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    args = parser.parse_args()

    return args


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


def compute_unet_number_of_voxels(window, channels, n_layers):
    n_voxels = 0
    for i in range(n_layers):
        n_voxels = n_voxels + ((1/(2**(3*i))) * window[0] * window[1] * window[2] * channels * 2**i * 2)
    return n_voxels


def compute_window_size(step, step_size, ratios):
    step_ratios = np.asarray(ratios) * step * step_size
    mod = np.mod(step_ratios, step_size)
    return np.asarray(step_ratios - mod + np.round(mod / step_size) * step_size, dtype=int)


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

    if "skip_validation" in config["training"] and config["training"]['skip_validation']:
        # if skipping the validation, the loss function will be montiored.
        metric_to_monitor = "loss"

    else:
        # when the validation is not skipped, the validation loss will be monitored.
        metric_to_monitor = "val_loss"

    dataset_class = load_dataset_class(config["dataset"])

    check_hierarchy(config)

    if in_config("add_contours", config["dataset"], False):
        config["n_outputs"] = config["n_outputs"] * 2

    model = build_or_load_model_from_config(config, os.path.abspath(namespace.model_filename), system_config["n_gpus"])

    start_training(config,
                   model,
                   os.path.abspath(namespace.training_log_filename),
                   model_filename=os.path.abspath(namespace.model_filename),
                   sequence_class=dataset_class,
                   metric_to_monitor=metric_to_monitor,
                   test_input=namespace.n_examples,
                   **in_config("training", config, dict()),
                   **system_config)

    if namespace.sub_command == "predict":
        run_inference(namespace)


if __name__ == '__main__':
    main()

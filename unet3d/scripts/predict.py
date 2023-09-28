#!/usr/bin/env python
import os
import logging
import argparse
from unet3d.utils.utils import load_json
from unet3d.predict.volumetric import volumetric_predictions
from unet3d.utils.filenames import load_dataset_class
from unet3d.scripts.segment import format_parser as format_segmentation_parser
from unet3d.scripts.script_utils import (get_machine_config, in_config,
                                         add_machine_config_to_parser,
                                         fetch_inference_dataset_kwargs_from_config,
                                         build_or_load_model_from_config,
                                         check_hierarchy, build_inference_loader, get_kwargs)


def format_parser(parser=argparse.ArgumentParser(), sub_command=False):
    parser.add_argument("--output_directory", required=True)
    if not sub_command:
        parser.add_argument("--config_filename", required=True)
        parser.add_argument("--model_filename", required=True)
        add_machine_config_to_parser(parser)

    parser.add_argument("--group", default="test",
                        help="Name of the group of filenames to make predictions on. The default is 'test'. "
                             "The script will look for a key in the configuration file that lists the filenames "
                             "to read in and make predictions on.")
    parser.add_argument("--activation",
                        help="Specify whether to apply an activation function to the outputs of the model before writing to "
                             "file.")

    format_segmentation_parser(parser, sub_command=True)
    return parser


def parse_args():
    return format_parser().parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    namespace = parse_args()
    logging.info("Config filename: %s", namespace.config_filename)
    config = load_json(namespace.config_filename)
    run_inference(config=config,
                  output_directory=namespace.output_directory,
                  model_filename=namespace.model_filename,
                  group=namespace.group,
                  activation=namespace.activation,
                  system_config=get_machine_config(namespace))


def run_inference(config, output_directory, model_filename, group, activation, system_config):
    logging.info("Output directory: %s", output_directory)
    work_dir = os.path.abspath(output_directory)
    label_hierarchy = check_hierarchy(config)
    cache_dir = os.path.join(work_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    logging.info("Cache dir: %s", cache_dir)
    dataset_class = load_dataset_class(config["dataset"],
                                       cache_dir=cache_dir)
    key = f"{group}_filenames"
    logging.info("Filenames key: %s", key)
    if key not in config:
        raise ValueError("Could not find key {} in the configuration file. Use the change the group "
                         "('--group' on commandline) to the name of the group of filenames "
                         "(e.g., 'validation' to use 'validation_filenames') "
                         "that you want to predict.")

    inference_dataset_kwargs, batch_size, prefetch_factor = fetch_inference_dataset_kwargs_from_config(config)

    dataloader = build_inference_loader(filenames=config[key],
                                        dataset_class=dataset_class,
                                        dataset_kwargs=get_kwargs(config["dataset"],
                                                                  ["name", "training", "validation"]),
                                        inference_kwargs=inference_dataset_kwargs,
                                        batch_size=batch_size,
                                        num_workers=in_config("n_workers", system_config, 1),
                                        pin_memory=in_config("pin_memory", system_config,
                                                             False),
                                        prefetch_factor=prefetch_factor)

    logging.info("Model filename: %s", model_filename)
    model = build_or_load_model_from_config(config,
                                            model_filename,
                                            system_config["n_gpus"],
                                            strict=True)
    model.eval()

    prediction_dir = os.path.join(work_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return volumetric_predictions(model=model,
                                  dataloader=dataloader,
                                  prediction_dir=prediction_dir,
                                  activation=activation,
                                  interpolation="trilinear",
                                  resample=in_config("resample", config["dataset"], False))


if __name__ == '__main__':
    main()

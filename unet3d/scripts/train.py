import argparse
import numpy as np
from unet3d.train import run_training
from unet3d.utils.filenames import generate_filenames, load_bias, load_sequence
from unet3d.utils.utils import load_json, in_config, dump_json
from unet3d.scripts.predict import format_parser as format_prediction_args
from unet3d.scripts.predict import run_inference
from unet3d.scripts.script_utils import get_machine_config, add_machine_config_to_parser


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
    parser.add_argument("--fit_gpu_mem", type=float,
                        help="Specify the amount of gpu memory available on a single gpu and change the image size to "
                             "fit into gpu memory automatically. Will try to find the largest image size that will fit "
                             "onto a single gpu. The batch size is overwritten and set to the number of gpus available."
                             " The new image size will be written to a new config file ending named "
                             "'<original_config>_auto.json'. This option is experimental and only works with the UNet "
                             "model. It has only been tested with gpus that have 12GB and 32GB of memory.")
    parser.add_argument("--batch_size", help="Override the batch size from the config file.", type=int)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Raises an error if a training file is not found. The default is to silently skip"
                             "any training files that cannot be found. Use this flag to debug the config for finding"
                             "the data.")
    add_machine_config_to_parser(parser)
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    args = parser.parse_args()

    return args


def check_hierarchy(config):
    if in_config("labels", config["sequence_kwargs"]) and in_config("use_label_hierarchy", config["sequence_kwargs"]):
        config["sequence_kwargs"].pop("use_label_hierarchy")
        labels = config["sequence_kwargs"].pop("labels")
        new_labels = list()
        while len(labels):
            new_labels.append(labels)
            labels = labels[1:]
        config["sequence_kwargs"]["labels"] = new_labels


def compute_unet_number_of_voxels(window, channels, n_layers):
    n_voxels = 0
    for i in range(n_layers):
        n_voxels = n_voxels + ((1/(2**(3*i))) * window[0] * window[1] * window[2] * channels * 2**i * 2)
    return n_voxels


def compute_window_size(step, step_size, ratios):
    step_ratios = np.asarray(ratios) * step * step_size
    mod = np.mod(step_ratios, step_size)
    return np.asarray(step_ratios - mod + np.round(mod / step_size) * step_size, dtype=int)


def update_config_to_fit_gpu_memory(config, n_gpus, gpu_memory, output_filename, voxels_per_gb=17000000.0,
                                    ratios=(1.22, 1.56, 1.0)):
    max_voxels = voxels_per_gb * gpu_memory
    n_layers = len(config["model_kwargs"]["encoder_blocks"])
    step_size = 2**(n_layers - 1)
    step = 1
    window = compute_window_size(step, step_size, ratios)
    n_voxels = compute_unet_number_of_voxels(window, config["model_kwargs"]["base_width"], n_layers)
    while n_voxels <= max_voxels:
        step = step + 1
        window = compute_window_size(step, step_size, ratios)
        n_voxels = compute_unet_number_of_voxels(window, config["model_kwargs"]["base_width"], n_layers)
    window = compute_window_size(step - 1, step_size, ratios).tolist()
    print("Setting window size to {} x {} x {}".format(*window))
    print("Setting batch size to", n_gpus)
    config["window"] = window
    config["model_kwargs"]["input_shape"] = window
    config["batch_size"] = n_gpus
    config["validation_batch_size"] = n_gpus
    print("Writing new configuration file:", output_filename)
    dump_json(config, output_filename)


def main():
    namespace = parse_args()

    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)

    print("Model: ", namespace.model_filename)
    print("Log: ", namespace.training_log_filename)
    system_config = get_machine_config(namespace)
    training_function_kwargs = in_config("training_function_kwargs", config, dict())

    # set verbosity
    if namespace.debug:
        if "sequence_kwargs" not in config:
            config["sequence_kwargs"] = dict()
        config["sequence_kwargs"]["verbose"] = namespace.debug

    # Override the batch size from the config file
    if namespace.batch_size:
        config["batch_size"] = namespace.batch_size

    if namespace.fit_gpu_mem and namespace.fit_gpu_mem > 0:
        update_config_to_fit_gpu_memory(config=config, n_gpus=system_config["n_gpus"], gpu_memory=namespace.fit_gpu_mem,
                                        output_filename=namespace.config_filename.replace(".json", "_auto.json"))

    model_metrics = []
    if config['skip_validation']:
        # if skipping the validation, the loss function will be montiored.
        metric_to_monitor = "loss"
        groups = ("training",)

    else:
        # when the validation is not skipped, the validation loss will be monitored.
        metric_to_monitor = "val_loss"
        groups = ("training", "validation")

    if "directory" in system_config:
        directory = system_config.pop("directory")
    elif "directory" in config:
        directory = config["directory"]
    else:
        directory = ""

    for name in groups:
        key = name + "_filenames"
        if key not in config:
            # If training or validation filenames are not listed in the config
            # try to generate the filenames
            config[key] = generate_filenames(config, name, directory,
                                             raise_if_not_exists=namespace.debug)
    sequence_class = load_sequence(config["sequence"])

    check_hierarchy(config)

    if in_config("add_contours", config["sequence_kwargs"], False):
        config["n_outputs"] = config["n_outputs"] * 2

    run_training(config, namespace.model_filename, namespace.training_log_filename,
                 sequence_class=sequence_class,
                 model_metrics=model_metrics,
                 metric_to_monitor=metric_to_monitor,
                 **training_function_kwargs,
                 **system_config)

    if namespace.sub_command == "predict":
        run_inference(namespace)


if __name__ == '__main__':
    main()

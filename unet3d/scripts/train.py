import os
import argparse
import pandas as pd
import numpy as np
from unet3d.train import run_training
from unet3d.utils.filenames import wrapped_partial, generate_filenames, load_bias, load_sequence
from unet3d.utils.sequences import (WholeVolumeToSurfaceSequence, HCPRegressionSequence, ParcelBasedSequence,
                                    WindowedAutoEncoderSequence)
from unet3d.utils.pytorch.dataset import (WholeBrainCIFTI2DenseScalarDataset, HCPRegressionDataset, AEDataset,
                                          WholeVolumeSegmentationDataset, WindowedAEDataset)
from unet3d.utils.utils import load_json, in_config, dump_json
from unet3d.utils.custom import get_metric_data_from_config
from unet3d.models.keras.resnet.resnet import compare_scores
from unet3d.scripts.predict import format_parser as format_prediction_args, check_hierarchy
from unet3d.scripts.predict import run_inference


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
    parser.add_argument("--directory", default="",
                        help="Directory within which to find the training data. This setting is ignored if "
                             "machine_config_filename is set.")
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--fit_gpu_mem", type=float,
                        help="Specify the amount of gpu memory available on a single gpu and change the image size to "
                             "fit into gpu memory automatically. Tries to find the largest image size that will fit "
                             "onto a single gpu. The batch size is overwritten and set to the number of gpus available."
                             " The new image size will be written to a new config file ending named "
                             "'<original_config>_auto.json'.")
    parser.add_argument("--group_average_filenames")
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    args = parser.parse_args()

    return args


def get_system_config(namespace):
    if namespace.machine_config_filename:
        print("MP Config: ", namespace.machine_config_filename)
        return load_json(namespace.machine_config_filename)
    else:
        return {"n_workers": namespace.nthreads,
                "n_gpus": namespace.ngpus,
                "use_multiprocessing": namespace.nthreads > 1,
                "pin_memory": namespace.pin_memory,
                "directory": namespace.directory}


def compute_unet_number_of_voxels(window, channels, n_layers):
    n_voxels = 0
    for i in range(n_layers):
        n_voxels = n_voxels + ((1/(2**(3*i))) * window[0] * window[1] * window[2] * channels * 2**i * 2)
    return n_voxels


def compute_window_size(step, step_size, ratios):
    step_ratios = np.asarray(ratios) * step * step_size
    mod = np.mod(step_ratios, step_size)
    return np.asarray(step_ratios - mod, dtype=int)


def update_config_to_fit_gpu_memory(config, n_gpus, gpu_memory, output_filename, voxels_per_gb=13200000.0,
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
    import nibabel as nib
    nib.imageglobals.logger.level = 40

    namespace = parse_args()

    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)

    if "package" in config:
        package = config["package"]
    else:
        package = "keras"

    if "metric_names" in config and not config["n_outputs"] == len(config["metric_names"]):
        raise ValueError("n_outputs set to {}, but number of metrics is {}.".format(config["n_outputs"],
                                                                                    len(config["metric_names"])))

    print("Model: ", namespace.model_filename)
    print("Log: ", namespace.training_log_filename)
    system_config = get_system_config(namespace)

    if namespace.fit_gpu_mem and namespace.fit_gpu_mem > 0:
        update_config_to_fit_gpu_memory(config=config, n_gpus=system_config["n_gpus"], gpu_memory=namespace.fit_gpu_mem,
                                        output_filename=namespace.config_filename.replace(".json", "_auto.json"))

    if namespace.group_average_filenames is not None:
        group_average = get_metric_data_from_config(namespace.group_average_filenames, namespace.config_filename)
        model_metrics = [wrapped_partial(compare_scores, comparison=group_average)]
        metric_to_monitor = "compare_scores"
    else:
        model_metrics = []
        if config['skip_validation']:
            metric_to_monitor = "loss"
        else:
            metric_to_monitor = "val_loss"

    if config["skip_validation"]:
        groups = ("training",)
    else:
        groups = ("training", "validation")

    for name in groups:
        key = name + "_filenames"
        if key not in config:
            config[key] = generate_filenames(config, name, system_config)
    if "directory" in system_config:
        directory = system_config.pop("directory")
    else:
        directory = "."

    if "sequence" in config:
        sequence_class = load_sequence(config["sequence"])
    elif "_wb_" in os.path.basename(namespace.config_filename):
        if "package" in config and config["package"] == "pytorch":
            if config["sequence"] == "AEDataset":
                sequence_class = AEDataset
            elif config["sequence"] == "WholeVolumeSegmentationDataset":
                sequence_class = WholeVolumeSegmentationDataset
            else:
                sequence_class = WholeBrainCIFTI2DenseScalarDataset
        else:
            sequence_class = WholeVolumeToSurfaceSequence
    elif config["sequence"] == "WindowedAutoEncoderSequence":
        sequence_class = WindowedAutoEncoderSequence
    elif config["sequence"] == "WindowedAEDataset":
        sequence_class = WindowedAEDataset
    elif "_pb_" in os.path.basename(namespace.config_filename):
        sequence_class = ParcelBasedSequence
        config["sequence_kwargs"]["parcellation_template"] = os.path.join(
            directory, config["sequence_kwargs"]["parcellation_template"])
    else:
        if config["package"] == "pytorch":
            sequence_class = HCPRegressionDataset
        else:
            sequence_class = HCPRegressionSequence

    if "bias_filename" in config and config["bias_filename"] is not None:
        bias = load_bias(config["bias_filename"])
    else:
        bias = None

    check_hierarchy(config)

    if in_config("add_contours", config["sequence_kwargs"], False):
        config["n_outputs"] = config["n_outputs"] * 2

    if sequence_class == ParcelBasedSequence:
        target_parcels = config["sequence_kwargs"].pop("target_parcels")
        for target_parcel in target_parcels:
            config["sequence_kwargs"]["target_parcel"] = target_parcel
            print("Training on parcel: {}".format(target_parcel))
            if type(target_parcel) == list:
                parcel_id = "-".join([str(i) for i in target_parcel])
            else:
                parcel_id = str(target_parcel)
            _training_log_filename = namespace.training_log_filename.replace(".csv", "_{}.csv".format(parcel_id))
            if os.path.exists(_training_log_filename):
                _training_log = pd.read_csv(_training_log_filename)
                if (_training_log[metric_to_monitor].values.argmin()
                        <= len(_training_log) - int(config["early_stopping_patience"])):
                    print("Already trained")
                    continue
            run_training(package,
                         config,
                         namespace.model_filename.replace(".h5", "_{}.h5".format(parcel_id)),
                         _training_log_filename,
                         sequence_class=sequence_class,
                         model_metrics=model_metrics,
                         metric_to_monitor=metric_to_monitor,
                         **system_config)

    else:
        run_training(package, config, namespace.model_filename, namespace.training_log_filename,
                     sequence_class=sequence_class,
                     model_metrics=model_metrics, metric_to_monitor=metric_to_monitor, bias=bias, **system_config)

    if namespace.sub_command == "predict":
        run_inference(namespace)


if __name__ == '__main__':
    main()

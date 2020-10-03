import os
import argparse
import pandas as pd
from fcnn.train import run_training
from fcnn.utils.filenames import wrapped_partial, generate_filenames, load_bias, load_sequence
from fcnn.utils.sequences import (WholeVolumeToSurfaceSequence, HCPRegressionSequence, ParcelBasedSequence,
                                  WindowedAutoEncoderSequence)
from fcnn.utils.pytorch.dataset import (WholeBrainCIFTI2DenseScalarDataset, HCPRegressionDataset, AEDataset,
                                        WholeVolumeSegmentationDataset, WindowedAEDataset)
from fcnn.utils.utils import load_json, in_config
from fcnn.utils.custom import get_metric_data_from_config
from fcnn.models.keras.resnet.resnet import compare_scores
from fcnn.scripts.run_unet_inference import format_parser as format_prediction_args, check_hierarchy
from fcnn.scripts.run_unet_inference import run_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True, help="Trail config JSON file.")
    parser.add_argument("--model_filename", required=True)
    parser.add_argument("--training_log_filename", required=True)
    parser.add_argument("--machine_config_filename", required=True)
    parser.add_argument("--group_average_filenames")
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    return parser.parse_args()


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
    print("MP Config: ", namespace.machine_config_filename)
    system_config = load_json(namespace.machine_config_filename)

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

#!/usr/bin/env python
import os
import argparse
from unet3d.utils.utils import load_json, in_config
from unet3d.predict.volumetric import volumetric_predictions
from unet3d.utils.filenames import load_dataset_class
from unet3d.scripts.segment import format_parser as format_segmentation_parser
from unet3d.scripts.script_utils import (get_machine_config, add_machine_config_to_parser,
                                         build_or_load_model_from_config)


def format_parser(parser=argparse.ArgumentParser(), sub_command=False):
    parser.add_argument("--output_directory", required=True)
    if not sub_command:
        parser.add_argument("--config_filename", required=True)
        parser.add_argument("--model_filename", required=True)
        add_machine_config_to_parser(parser)
    parser.add_argument("--directory_template", help="Set this if directory template for running the predictions is "
                                                     "different from the directory used for training.")
    parser.add_argument("--group", default="test")
    parser.add_argument("--eval", default=False, action="store_true",
                        help="Scores the predictions according to the validation criteria and saves the results to a "
                             "csv file in the prediction directory.")
    parser.add_argument("--no_resample", default=False, action="store_true",
                        help="Skips resampling the predicted images into the non-cropped image space. This can help "
                             "save on the storage space as the images can always be resampled back into the original "
                             "space when needed.")
    parser.add_argument("--interpolation", default="bilinear",
                        help="Interpolation method to use when resampling the predicted images into the non-cropped "
                             "image space. Options are 'bilinear', 'nearest', or spline interpolation "
                             "number 0-5.")
    parser.add_argument("--output_template")
    parser.add_argument("--replace", nargs="*")
    parser.add_argument("--filenames", nargs="*")
    parser.add_argument("--sub_volumes", nargs="*", type=int)
    parser.add_argument("--alternate_prediction_func", help="Manually set which function will be called to make the "
                                                            "volumetric predictions.")
    parser.add_argument("--activation", default=None)
    parser.add_argument("--write_input_images", default=False, action="store_true")
    format_segmentation_parser(parser, sub_command=True)
    return parser


def parse_args():
    return format_parser().parse_args()


def main():
    namespace = parse_args()
    run_inference(namespace)


def run_inference(namespace):
    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)
    key = namespace.group + "_filenames"

    system_config = get_machine_config(namespace)

    if namespace.filenames:
        filenames = list()
        for filename in namespace.filenames:
            filenames.append([filename, namespace.sub_volumes, None, None, os.path.basename(filename).split(".")[0]])
    else:
        filenames = config[key]

    print("Model: ", namespace.model_filename)

    print("Output Directory:", namespace.output_directory)

    if not os.path.exists(namespace.output_directory):
        os.makedirs(namespace.output_directory)

    dataset_class = load_dataset_class(config["dataset"])
    if "training" in config["dataset"]:
        config["dataset"].pop("training")

    if "validation" in config["dataset"]:
        validation_kwargs = config["dataset"].pop("validation")
    else:
        validation_kwargs = dict()

    labels = config["dataset"]["labels"] if namespace.segment else None
    if "use_label_hierarchy" in config["dataset"]:
        label_hierarchy = config["dataset"].pop("use_label_hierarchy")
    else:
        label_hierarchy = False

    if label_hierarchy and (namespace.threshold != 0.5 or namespace.sum):
        # TODO: put a warning here instead of a print statement
        print("Using label hierarchy. Resetting threshold to 0.5 and turning the summation off.")
        namespace.threshold = 0.5
        namespace.sum = False
    if in_config("add_contours", validation_kwargs, False):
        config["n_outputs"] = config["n_outputs"] * 2
        if namespace.use_contours:
            # this sets the labels for the contours
            if label_hierarchy:
                raise RuntimeError("Cannot use contours for segmentation while a label hierarchy is specified.")
            labels = list(labels) + list(labels)

    model = build_or_load_model_from_config(config, namespace.model_filename, system_config["n_gpus"], strict=True)
    model.eval()

    return volumetric_predictions(model=model,
                                  filenames=filenames,
                                  prediction_dir=namespace.output_directory,
                                  prefix=os.path.basename(namespace.model_filename).split(".")[0],
                                  batch_size=config['training']['validation_batch_size'],
                                  dataset_kwargs={**validation_kwargs, **config["dataset"]},
                                  sequence=dataset_class,
                                  resample_predictions=(not namespace.no_resample),
                                  interpolation=namespace.interpolation,
                                  output_template=namespace.output_template,
                                  segmentation=namespace.segment,
                                  segmentation_labels=labels,
                                  threshold=namespace.threshold,
                                  sum_then_threshold=namespace.sum,
                                  label_hierarchy=label_hierarchy,
                                  write_input_images=namespace.write_input_images,
                                  **system_config)


if __name__ == '__main__':
    main()

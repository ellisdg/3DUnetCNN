import os
import argparse
from unet3d.utils.utils import load_json, in_config
from unet3d.predict import volumetric_predictions
from unet3d.utils.filenames import generate_filenames, load_subject_ids, load_sequence
from unet3d.scripts.segment import format_parser as format_segmentation_parser
from unet3d.scripts.script_utils import get_machine_config, add_machine_config_to_parser


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
    parser.add_argument("--interpolation", default="linear")
    parser.add_argument("--output_template")
    parser.add_argument("--replace", nargs="*")
    parser.add_argument("--subjects_config_filename",
                        help="Allows for specification of the config that contains the subject ids. If not set and the "
                             "subject ids are not listed in the main config, then the filename for the subjects config "
                             "will be read from the main config.")
    parser.add_argument("--source",
                        help="If using multisource templates set this to predict only filenames from a single source.")
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

    machine_config = get_machine_config(namespace)

    if namespace.filenames:
        filenames = list()
        for filename in namespace.filenames:
            filenames.append([filename, namespace.sub_volumes, None, None, os.path.basename(filename).split(".")[0]])
    elif key not in config:
        if namespace.replace is not None:
            for _key in ("directory", "feature_templates", "target_templates"):
                if _key in config["generate_filenames_kwargs"]:
                    if type(config["generate_filenames_kwargs"][_key]) == str:

                        for i in range(0, len(namespace.replace), 2):
                            config["generate_filenames_kwargs"][_key] = config["generate_filenames_kwargs"][_key].replace(
                                namespace.replace[i], namespace.replace[i + 1])
                    else:
                        config["generate_filenames_kwargs"][_key] = [template.replace(namespace.replace[0],
                                                                                      namespace.replace[1]) for template
                                                                     in
                                                                     config["generate_filenames_kwargs"][_key]]
        if namespace.directory_template is not None:
            machine_config["directory"] = namespace.directory_template
        if namespace.subjects_config_filename:
            config[namespace.group] = load_json(namespace.subjects_config_filename)[namespace.group]
        filenames = generate_filenames(config, namespace.group, machine_config,
                                       skip_targets=(not namespace.eval))

    else:
        filenames = config[key]

    print("Model: ", namespace.model_filename)

    print("Output Directory:", namespace.output_directory)

    if not os.path.exists(namespace.output_directory):
        os.makedirs(namespace.output_directory)

    load_subject_ids(config)

    if "evaluation_metric" in config and config["evaluation_metric"] is not None:
        criterion_name = config['evaluation_metric']
    else:
        criterion_name = config['loss']

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
    else:
        model_kwargs = dict()

    if namespace.activation:
        model_kwargs["activation"] = namespace.activation

    if "sequence_kwargs" in config:
        sequence_kwargs = config["sequence_kwargs"]
        # make sure any augmentations are set to None
        for key in ["augment_scale_std", "additive_noise_std"]:
            if key in sequence_kwargs:
                sequence_kwargs[key] = None
    else:
        sequence_kwargs = dict()

    if "reorder" not in sequence_kwargs:
        sequence_kwargs["reorder"] = in_config("reorder", config, False)

    if "generate_filenames" in config and config["generate_filenames"] == "multisource_templates":
        if namespace.filenames is not None:
            sequence_kwargs["inputs_per_epoch"] = None
        else:
            # set which source(s) to use for prediction filenames
            if "inputs_per_epoch" not in sequence_kwargs:
                sequence_kwargs["inputs_per_epoch"] = dict()
            if namespace.source is not None:
                # just use the named source
                for dataset in filenames:
                    sequence_kwargs["inputs_per_epoch"][dataset] = 0
                sequence_kwargs["inputs_per_epoch"][namespace.source] = "all"
            else:
                # use all sources
                for dataset in filenames:
                    sequence_kwargs["inputs_per_epoch"][dataset] = "all"
    if namespace.sub_volumes is not None:
        sequence_kwargs["extract_sub_volumes"] = True

    if "sequence" in config:
        sequence = load_sequence(config["sequence"])
    else:
        sequence = None

    labels = sequence_kwargs["labels"] if namespace.segment else None
    if "use_label_hierarchy" in sequence_kwargs:
        label_hierarchy = sequence_kwargs.pop("use_label_hierarchy")
    else:
        label_hierarchy = False

    if label_hierarchy and (namespace.threshold != 0.5 or namespace.sum):
        # TODO: put a warning here instead of a print statement
        print("Using label hierarchy. Resetting threshold to 0.5 and turning the summation off.")
        namespace.threshold = 0.5
        namespace.sum = False
    if in_config("add_contours", sequence_kwargs, False):
        config["n_outputs"] = config["n_outputs"] * 2
        if namespace.use_contours:
            # this sets the labels for the contours
            if label_hierarchy:
                raise RuntimeError("Cannot use contours for segmentation while a label hierarchy is specified.")
            labels = list(labels) + list(labels)

    if namespace.alternate_prediction_func:
        from unet3d import predict
        func = getattr(predict, namespace.alternate_prediction_func)
    else:
        func = volumetric_predictions

    return func(model_filename=namespace.model_filename,
                filenames=filenames,
                prediction_dir=namespace.output_directory,
                model_name=config["model_name"],
                n_features=config["n_features"],
                window=config["window"],
                criterion_name=criterion_name,
                package=config['package'],
                n_gpus=machine_config['n_gpus'],
                batch_size=config['validation_batch_size'],
                n_workers=machine_config["n_workers"],
                model_kwargs=model_kwargs,
                sequence_kwargs=sequence_kwargs,
                sequence=sequence,
                n_outputs=config["n_outputs"],
                metric_names=in_config("metric_names", config, None),
                evaluate_predictions=namespace.eval,
                resample_predictions=(not namespace.no_resample),
                interpolation=namespace.interpolation,
                output_template=namespace.output_template,
                segmentation=namespace.segment,
                segmentation_labels=labels,
                threshold=namespace.threshold,
                sum_then_threshold=namespace.sum,
                label_hierarchy=label_hierarchy,
                write_input_images=namespace.write_input_images)


if __name__ == '__main__':
    main()

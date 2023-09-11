#!/usr/bin/env python
import argparse
import os
import warnings
import logging
from unet3d.train import run_training
from unet3d.utils.utils import load_json
from unet3d.predict import volumetric_predictions
from unet3d.utils.filenames import load_dataset_class
from unet3d.scripts.script_utils import (get_machine_config, add_machine_config_to_parser, build_optimizer,
                                         build_or_load_model_from_config, load_criterion_from_config, in_config,
                                         build_data_loaders_from_config, build_scheduler_from_config,
                                         setup_cross_validation, load_filenames_from_config,
                                         build_inference_loaders_from_config, check_hierarchy,
                                         build_inferer_from_config, get_activation_from_config)


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
                             "Default is '{output_dir}/{config_basename}/model.pth'. "
                             "The default behavior is to use flexible loading of the model where not all the "
                             "model layers/weights have to match. "
                             "If training is interrupted and resumed, if a pretrained model is defined "
                             "the pretrained model will be used instead of loading "
                             "the model that was being trained initially. Therefore, if you are resuming training "
                             "it is best to not set the pretrained_model_filename.",
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
    args = parser.parse_args()

    return args


def run(config_filename, output_dir, namespace):
    logging.info("Config: %s", config_filename)
    config = load_json(config_filename)
    load_filenames_from_config(config)

    work_dir = os.path.join(output_dir, os.path.basename(config_filename).split(".")[0])
    logging.info("Work Dir: %s", work_dir)
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
                logging.info("Running cross validation fold: %s", _config_filename)
                run(_config_filename, work_dir, namespace)
            else:
                logging.info("Setup cross validation fold: %s", _config_filename)
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
        logging.info("Model: %s", model_filename)

        training_log_filename = os.path.join(work_dir, "training_log.csv")
        logging.info("Log: %s", training_log_filename)

        label_hierarchy = check_hierarchy(config)
        dataset_class = load_dataset_class(config["dataset"], cache_dir=os.path.join(work_dir, "cache"))
        if namespace.n_examples:
            logging.info('Setting config["training"]["test_input"]=%d', namespace.n_examples)
            config["training"]["test_input"] = namespace.n_examples
        training_loader, validation_loader, metric_to_monitor = build_data_loaders_from_config(config,
                                                                                               system_config,
                                                                                               work_dir,
                                                                                               dataset_class)
        pretrained = namespace.pretrained_model_filename
        if pretrained:
            pretrained = os.path.abspath(pretrained)
        else:
            pretrained = model_filename
        model = build_or_load_model_from_config(config,
                                                pretrained,
                                                system_config["n_gpus"])
        criterion = load_criterion_from_config(config, n_gpus=system_config["n_gpus"])
        optimizer = build_optimizer(optimizer_name=config["optimizer"].pop("name"),
                                    model_parameters=model.parameters(),
                                    **config["optimizer"])
        scheduler = build_scheduler_from_config(config, optimizer)

        # build the inferer (e.g. sliding window inferer) that allows for inference to be distinct from training
        if "inference" in config:
            inferer = build_inferer_from_config(config)
        else:
            inferer = None

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
                     samples_per_epoch=in_config("samples_per_epoch", config["training"], None),
                     inferer=inferer,
                     training_iterations_per_epoch=in_config("training_iterations_per_epoch",
                                                             config["training"], 1))

        for _dataloader, _name in build_inference_loaders_from_config(config,
                                                                      dataset_class=dataset_class,
                                                                      system_config=system_config):
            prediction_dir = os.path.join(work_dir, _name)
            os.makedirs(prediction_dir, exist_ok=True)
            volumetric_predictions(model=model,
                                   dataloader=_dataloader,
                                   prediction_dir=prediction_dir,
                                   interpolation="trilinear",
                                   resample=in_config("resample", config["dataset"], False),
                                   inferer=inferer,
                                   activation=get_activation_from_config(config))


def main():
    # TODO: move this to only be set at debugging level when the debug flag is set
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    namespace = parse_args()
    config_filename = os.path.abspath(namespace.config_filename)
    if namespace.output_dir:
        output_dir = os.path.abspath(namespace.output_dir)
    else:
        output_dir = os.path.dirname(config_filename)
    run(config_filename, output_dir, namespace)


if __name__ == '__main__':
    main()

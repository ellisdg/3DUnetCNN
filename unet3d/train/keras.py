import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import keras
import nibabel as nib
from fcnn.models.keras.load import load_model
from fcnn.models.keras.build import build_model
import numpy as np
from fcnn.utils.sequences import HCPRegressionSequence


def run_keras_training(config, model_filename, training_log_filename, verbose=1, use_multiprocessing=False,
                       n_workers=1, max_queue_size=5, model_name='resnet_34', sequence_class=HCPRegressionSequence,
                       test_input=1, metric_to_monitor="loss", model_metrics=(), n_gpus=1):
    """
    :param test_input: integer with the number of inputs from the generator to write to file. 0, False, or None will
    write no inputs to file.
    :param sequence_class: class to use for the generator sequence
    :param model_name:
    :param verbose:
    :param use_multiprocessing:
    :param n_workers:
    :param max_queue_size:
    :param config:
    :param model_filename:
    :param training_log_filename:
    :param metric_to_monitor:
    :param model_metrics:
    :return:

    Anything that directly affects the training results should go into the config file. Other specifications such as
    multiprocessing optimization should be arguments to this function, as these arguments affect the computation time,
    but the results should not vary based on whether multiprocessing is used or not.
    """
    window = np.asarray(config['window'])
    spacing = np.asarray(config['spacing'])

    if 'model_name' in config:
        model_name = config['model_name']

    # 2. Create model_filename
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        input_shape = tuple(window.tolist() + [config['n_features']])
        if "n_outputs" in config:
            num_outputs = config['n_outputs']
        else:
            num_outputs = len(np.concatenate(config['metric_names']))
        model = build_model(model_name, input_shape=input_shape, num_outputs=num_outputs,
                            activation=config['activation'])
    if "freeze_bias" in config and config["freeze_bias"]:
        dense = model.layers[-1]
        bias = dense.trainable_weights.pop(dense.trainable_weights.index(dense.bias))
        dense.non_trainable_weights.append(bias)
        model.optimizer = None
    if n_gpus > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, n_gpus)
        model.optimizer = None

    if not hasattr(model, 'optimizer') or model.optimizer is None:
        model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=model_metrics)

    if "initial_learning_rate" in config:
        keras.backend.set_value(model.optimizer.lr, config['initial_learning_rate'])
    if "iterations_per_epoch" in config:
        iterations_per_epoch = config["iterations_per_epoch"]
    else:
        iterations_per_epoch = 1
    if "additional_training_args" in config:
        train_kwargs = config["additional_training_args"]
    else:
        train_kwargs = dict()

    if "sequence_kwargs" in config:
        sequence_kwargs = config["sequence_kwargs"]
    else:
        sequence_kwargs = dict()

    # 4. Create Generators
    training_generator = sequence_class(filenames=config['training_filenames'],
                                        batch_size=config['batch_size'],
                                        flip=config['flip'],
                                        reorder=config['reorder'],
                                        window=window,
                                        spacing=spacing,
                                        points_per_subject=config['points_per_subject'],
                                        surface_names=config['surface_names'],
                                        metric_names=config['metric_names'],
                                        iterations_per_epoch=iterations_per_epoch,
                                        **train_kwargs,
                                        **sequence_kwargs)

    if test_input:
        n_test_batches = int(np.ceil(test_input/float(config['batch_size'])))
        for batch_index in range(n_test_batches):
            x, y = training_generator[batch_index]
            for within_batch_index in range(min([config['batch_size'],
                                                 test_input - (batch_index * config['batch_size'])])):
                x_image = nib.Nifti1Image(x[within_batch_index], affine=np.diag(np.ones(4)))
                x_image.to_filename(model_filename.replace(".h5",
                                                           "_input_test_{}.nii.gz".format(
                                                               within_batch_index
                                                               + config['batch_size']
                                                               * batch_index)))

    if 'skip_validation' in config and config['skip_validation']:
        validation_generator = None
    else:
        validation_generator = sequence_class(filenames=config['validation_filenames'],
                                              batch_size=config['validation_batch_size'],
                                              flip=False,
                                              reorder=config['reorder'],
                                              window=window,
                                              spacing=spacing,
                                              points_per_subject=config['validation_points_per_subject'],
                                              surface_names=config['surface_names'],
                                              metric_names=config['metric_names'],
                                              **sequence_kwargs)

    # 5. Run Training

    callbacks = []
    checkpointer = ModelCheckpoint(filepath=model_filename,
                                   verbose=verbose,
                                   monitor=metric_to_monitor,
                                   mode="min",
                                   save_best_only=config['save_best_only'])
    callbacks.append(checkpointer)
    if config["decay_patience"]:
        reduce_lr = ReduceLROnPlateau(monitor=metric_to_monitor,
                                      factor=config['decay_factor'],
                                      patience=config['decay_patience'],
                                      min_lr=config['min_learning_rate'])
        print("Will reduce LR by a factor of {} after {} epochs.".format(config["decay_factor"],
                                                                        config["decay_patience"]))
        callbacks.append(reduce_lr)
    csv_logger = CSVLogger(training_log_filename, append=True)
    callbacks.append(csv_logger)
    if "early_stopping_patience" in config and config["early_stopping_patience"]:
        from keras.callbacks import EarlyStopping
        print("Will stop training after {} epochs without {} decrease.".format(config["early_stopping_patience"],
                                                                               metric_to_monitor))
        early_stopping = EarlyStopping(monitor=metric_to_monitor,
                                       patience=config["early_stopping_patience"],
                                       verbose=verbose)
        callbacks.append(early_stopping)
    history = model.fit_generator(generator=training_generator,
                                  epochs=config['n_epochs'],
                                  use_multiprocessing=use_multiprocessing,
                                  workers=n_workers,
                                  max_queue_size=max_queue_size,
                                  callbacks=callbacks,
                                  validation_data=validation_generator)

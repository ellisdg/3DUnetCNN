import os

import numpy as np
from nilearn.image import new_img_like, resample_to_img
from unet3d.predict.utils import pytorch_predict_batch_array, get_feature_filename_and_subject_id
from unet3d.utils.utils import one_hot_image_to_label_map, get_nibabel_data


def load_volumetric_model(model_name, model_filename, n_outputs, n_features, n_gpus, strict, **kwargs):
    from unet3d.models.pytorch.build import build_or_load_model
    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_outputs=n_outputs,
                                n_features=n_features, n_gpus=n_gpus, strict=strict, **kwargs)
    model.eval()
    return model


def load_volumetric_sequence(sequence, sequence_kwargs, filenames, window, spacing, metric_names, batch_size=1):
    from ..utils.pytorch.dataset import AEDataset
    if sequence is None:
        sequence = AEDataset
    if sequence_kwargs is None:
        sequence_kwargs = dict()
    dataset = sequence(filenames=filenames, window=window, spacing=spacing, batch_size=batch_size,
                       metric_names=metric_names,
                       **sequence_kwargs)
    return dataset


def load_volumetric_model_and_dataset(model_name, model_filename, model_kwargs, n_outputs, n_features,
                                      strict_model_loading, n_gpus, sequence, sequence_kwargs, filenames, window,
                                      spacing, metric_names):
    if model_kwargs is None:
        model_kwargs = dict()

    model = load_volumetric_model(model_name=model_name, model_filename=model_filename, n_outputs=n_outputs,
                                  n_features=n_features, strict=strict_model_loading, n_gpus=n_gpus, **model_kwargs)
    dataset = load_volumetric_sequence(sequence, sequence_kwargs, filenames, window, spacing, metric_names,
                                       batch_size=1)
    basename = os.path.basename(model_filename).split(".")[0]
    return model, dataset, basename


def load_images_from_dataset(dataset, idx, resample_predictions):
    if resample_predictions:
        x_image, ref_image = dataset.get_feature_image(idx, return_unmodified=True)
    else:
        x_image = dataset.get_feature_image(idx)
        ref_image = None
    return x_image, ref_image


def prediction_to_image(data, input_image, reference_image=None, interpolation="linear", segmentation=False,
                        segmentation_labels=None, threshold=0.5, sum_then_threshold=False, label_hierarchy=False):
    if data.dtype == np.float16:
        data = np.asarray(data, dtype=np.float32)
    pred_image = new_img_like(input_image, data=data)
    if reference_image is not None:
        pred_image = resample_to_img(pred_image, reference_image,
                                     interpolation=interpolation)
    if segmentation:
        pred_image = one_hot_image_to_label_map(pred_image,
                                                labels=segmentation_labels,
                                                threshold=threshold,
                                                sum_then_threshold=sum_then_threshold,
                                                label_hierarchy=label_hierarchy)
    return pred_image


def write_prediction_image_to_file(pred_image, output_template, subject_id, x_filename, prediction_dir, basename,
                                   verbose=False):
    if output_template is None:
        while type(x_filename) == list:
            x_filename = x_filename[0]
        pred_filename = os.path.join(prediction_dir,
                                     "_".join([subject_id,
                                               basename,
                                               os.path.basename(x_filename)]))
    else:
        pred_filename = os.path.join(prediction_dir,
                                     output_template.format(subject=subject_id))
    if verbose:
        print("Writing:", pred_filename)
    pred_image.to_filename(pred_filename)


def predict_volumetric_batch(model, batch, batch_references, batch_subjects, batch_filenames,
                             basename, prediction_dir,
                             segmentation, output_template, n_gpus, verbose, threshold, interpolation,
                             segmentation_labels, sum_then_threshold, label_hierarchy, write_input_image=False):
    pred_x = pytorch_predict_batch_array(model, batch, n_gpus=n_gpus)
    for batch_idx in range(len(batch)):
        pred_image = prediction_to_image(pred_x[batch_idx].squeeze(), input_image=batch_references[batch_idx][0],
                                         reference_image=batch_references[batch_idx][1], interpolation=interpolation,
                                         segmentation=segmentation, segmentation_labels=segmentation_labels,
                                         threshold=threshold, sum_then_threshold=sum_then_threshold,
                                         label_hierarchy=label_hierarchy)
        write_prediction_image_to_file(pred_image, output_template,
                                       subject_id=batch_subjects[batch_idx],
                                       x_filename=batch_filenames[batch_idx],
                                       prediction_dir=prediction_dir,
                                       basename=basename,
                                       verbose=verbose)
        if write_input_image:
            write_prediction_image_to_file(batch_references[batch_idx][0], output_template=output_template,
                                           subject_id=batch_subjects[batch_idx] + "_input",
                                           x_filename=batch_filenames[batch_idx],
                                           prediction_dir=prediction_dir,
                                           basename=basename,
                                           verbose=verbose)


def pytorch_volumetric_predictions(model_filename, model_name, n_features, filenames, window,
                                   criterion_name, prediction_dir=None, output_csv=None, reference=None,
                                   n_gpus=1, n_workers=1, batch_size=1, model_kwargs=None, n_outputs=None,
                                   sequence_kwargs=None, spacing=None, sequence=None,
                                   strict_model_loading=True, metric_names=None,
                                   print_prediction_time=True, verbose=True,
                                   evaluate_predictions=False, resample_predictions=False, interpolation="linear",
                                   output_template=None, segmentation=False, segmentation_labels=None,
                                   sum_then_threshold=True, threshold=0.7, label_hierarchy=None,
                                   write_input_images=False):
    import torch
    # from .train.pytorch import load_criterion

    model, dataset, basename = load_volumetric_model_and_dataset(model_name, model_filename, model_kwargs, n_outputs,
                                                                 n_features, strict_model_loading, n_gpus, sequence,
                                                                 sequence_kwargs, filenames, window, spacing,
                                                                 metric_names)

    # criterion = load_criterion(criterion_name, n_gpus=n_gpus)
    results = list()
    print("Dataset: ", len(dataset))
    with torch.no_grad():
        batch = list()
        batch_references = list()
        batch_subjects = list()
        batch_filenames = list()
        for idx in range(len(dataset)):
            x_filename, subject_id = get_feature_filename_and_subject_id(dataset, idx, verbose=verbose)
            x_image, ref_image = load_images_from_dataset(dataset, idx, resample_predictions)

            batch.append(get_nibabel_data(x_image))
            batch_references.append((x_image, ref_image))
            batch_subjects.append(subject_id)
            batch_filenames.append(x_filename)
            if len(batch) >= batch_size or idx == (len(dataset) - 1):
                predict_volumetric_batch(model=model, batch=batch, batch_references=batch_references,
                                         batch_subjects=batch_subjects, batch_filenames=batch_filenames,
                                         basename=basename, prediction_dir=prediction_dir,
                                         segmentation=segmentation, output_template=output_template, n_gpus=n_gpus,
                                         verbose=verbose, threshold=threshold, interpolation=interpolation,
                                         segmentation_labels=segmentation_labels,
                                         sum_then_threshold=sum_then_threshold, label_hierarchy=label_hierarchy,
                                         write_input_image=write_input_images)
                batch = list()
                batch_references = list()
                batch_subjects = list()
                batch_filenames = list()


def volumetric_predictions(model_filename, filenames, prediction_dir, model_name, n_features, window,
                           criterion_name, package="keras", n_gpus=1, n_workers=1, batch_size=1,
                           model_kwargs=None, n_outputs=None, sequence_kwargs=None, sequence=None,
                           metric_names=None, evaluate_predictions=False, interpolation="linear",
                           resample_predictions=True, output_template=None, segmentation=False,
                           segmentation_labels=None, threshold=0.5, sum_then_threshold=True, label_hierarchy=None,
                           write_input_images=False):
    if package == "pytorch":
        pytorch_volumetric_predictions(model_filename=model_filename,
                                       model_name=model_name,
                                       n_outputs=n_outputs,
                                       n_features=n_features,
                                       filenames=filenames,
                                       prediction_dir=prediction_dir,
                                       window=window,
                                       criterion_name=criterion_name,
                                       n_gpus=n_gpus,
                                       n_workers=n_workers,
                                       batch_size=batch_size,
                                       model_kwargs=model_kwargs,
                                       sequence_kwargs=sequence_kwargs,
                                       sequence=sequence,
                                       metric_names=metric_names,
                                       evaluate_predictions=evaluate_predictions,
                                       interpolation=interpolation,
                                       resample_predictions=resample_predictions,
                                       output_template=output_template,
                                       segmentation=segmentation,
                                       segmentation_labels=segmentation_labels,
                                       threshold=threshold,
                                       sum_then_threshold=sum_then_threshold,
                                       label_hierarchy=label_hierarchy,
                                       write_input_images=write_input_images)
    else:
        raise ValueError("Predictions not yet implemented for {}".format(package))
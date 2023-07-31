import os
import torch

from unet3d.utils.resample import resample_to_img
from unet3d.predict.utils import pytorch_predict_batch_array, get_feature_filename_and_subject_id
from unet3d.utils.utils import one_hot_image_to_label_map, load_image


def load_volumetric_model(model_name, model_filename, n_gpus, strict, **kwargs):
    from unet3d.models.build import build_or_load_model
    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_gpus=n_gpus, strict=strict,
                                **kwargs)
    model.eval()
    return model


def load_images_from_dataset(dataset, idx, resample_predictions):
    if resample_predictions:
        x_image, ref_image = dataset.get_feature_image(idx, return_unmodified=True)
    else:
        x_image = dataset.get_feature_image(idx)
        ref_image = None
    return x_image, ref_image


def prediction_to_image(data, input_image, reference_image=None, interpolation="linear", segmentation=False,
                        segmentation_labels=None, threshold=0.5, sum_then_threshold=False, label_hierarchy=False):
    if data.dtype == torch.float16:
        data = torch.as_tensor(data, dtype=torch.float32)
    pred_image = input_image.make_similar(data=data)
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
        pred_image = prediction_to_image(pred_x[batch_idx], input_image=batch_references[batch_idx][0],
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


def volumetric_predictions(model, dataloader, prediction_dir, interpolation="linear",
                           segmentation=False, segmentation_labels=None,
                           sum_then_threshold=True, threshold=0.5, label_hierarchy=None):

    print("Dataset: ", len(dataloader))
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            predictions = model(x)
            for prediction, _x, _y in zip(predictions, x, y):
                prediction_image = prediction_to_image(prediction, _x,
                                                       reference_image=load_image(x.meta["source_filename"]),
                                                       interpolation=interpolation, segmentation=segmentation,
                                                       segmentation_labels=segmentation_labels,
                                                       sum_then_threshold=sum_then_threshold,
                                                       label_hierarchy=label_hierarchy, threshold=threshold)
                out_filename = os.path.join(prediction_dir, os.path.basename(_y.meta["source_filename"]))
                prediction_image.to_filename(out_filename)

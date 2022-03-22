import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.image import new_img_like

from unet3d.predict.volumetric import load_volumetric_model_and_dataset, load_images_from_dataset, \
    prediction_to_image, write_prediction_image_to_file
from unet3d.predict.utils import pytorch_predict_batch_array, get_feature_filename_and_subject_id, pytorch_predict_batch
from unet3d.utils.utils import (load_json, get_nibabel_data, break_down_volume_into_half_size_volumes, combine_half_size_volumes)
from unet3d.utils.sequences import SubjectPredictionSequence
from unet3d.utils.pytorch.dataset import HCPSubjectDataset
from unet3d.utils.hcp import new_cifti_scalar_like, get_metric_data
from unet3d.utils.filenames import generate_hcp_filenames, load_subject_ids
from unet3d.utils.augment import generate_permutation_keys, permute_data, reverse_permute_data


def predict_data_loader(model, data_loader):
    import torch
    predictions = list()
    with torch.no_grad():
        for batch_x in data_loader:
            predictions.extend(model(batch_x).cpu().numpy())
    return np.asarray(predictions)


def predict_generator(model, generator, use_multiprocessing=False, n_workers=1, max_queue_size=8, verbose=1,
                      package="keras", batch_size=1):
    if package == "pytorch":
        from torch.utils.data import DataLoader

        loader = DataLoader(generator, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        if verbose:
            print("Loader: ", len(loader), "  Batch_size: ", batch_size, "  Dataset: ", len(generator))
        return predict_data_loader(model, loader)

    else:
        return model.predict_generator(generator, use_multiprocessing=use_multiprocessing, workers=n_workers,
                                       max_queue_size=max_queue_size, verbose=verbose)


def predict_subject(model, feature_filename, surface_filenames, surface_names, metric_names, output_filename,
                    reference_filename, batch_size=50, window=(64, 64, 64), flip=False, spacing=(1, 1, 1),
                    use_multiprocessing=False, workers=1, max_queue_size=10, overwrite=True,
                    generator=SubjectPredictionSequence, package="keras"):
    if overwrite or not os.path.exists(output_filename):
        generator = generator(feature_filename=feature_filename, surface_filenames=surface_filenames,
                              surface_names=surface_names, batch_size=batch_size, window=window, flip=flip,
                              spacing=spacing, reference_metric_filename=reference_filename)
        prediction = predict_generator(model, generator, use_multiprocessing=use_multiprocessing, n_workers=workers,
                                       max_queue_size=max_queue_size, verbose=1, batch_size=batch_size,
                                       package=package)
        output_image = new_cifti_scalar_like(np.moveaxis(prediction, 1, 0), scalar_names=metric_names,
                                             structure_names=surface_names,
                                             reference_cifti=nib.load(reference_filename), almost_equals_decimals=0)
        output_image.to_filename(output_filename)


def make_predictions(config_filename, model_filename, output_directory='./', n_subjects=None, shuffle=False,
                     key='validation_filenames', use_multiprocessing=False, n_workers=1, max_queue_size=5,
                     batch_size=50, overwrite=True, single_subject=None, output_task_name=None, package="keras",
                     directory="./", n_gpus=1):
    output_directory = os.path.abspath(output_directory)
    config = load_json(config_filename)

    if key not in config:
        name = key.split("_")[0]
        if name not in config:
            load_subject_ids(config)
        config[key] = generate_hcp_filenames(directory,
                                             config['surface_basename_template'],
                                             config['target_basenames'],
                                             config['feature_basenames'],
                                             config[name],
                                             config['hemispheres'])

    filenames = config[key]

    model_basename = os.path.basename(model_filename).replace(".h5", "")

    if "package" in config and config["package"] == "pytorch":
        generator = HCPSubjectDataset
        package = "pytorch"
    else:
        generator = SubjectPredictionSequence

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
    else:
        model_kwargs = dict()

    if "batch_size" in config:
        batch_size = config["batch_size"]

    if single_subject is None:
        if package == "pytorch":
            from unet3d.models.pytorch.build import build_or_load_model

            model = build_or_load_model(model_filename=model_filename, model_name=config["model_name"],
                                        n_features=config["n_features"], n_outputs=config["n_outputs"],
                                        n_gpus=n_gpus, **model_kwargs)
        else:
            from keras.models import load_model
            model = load_model(model_filename)
    else:
        model = None

    if n_subjects is not None:
        if shuffle:
            np.random.shuffle(filenames)
        filenames = filenames[:n_subjects]

    for feature_filename, surface_filenames, metric_filenames, subject_id in filenames:
        if single_subject is None or subject_id == single_subject:
            if model is None:
                if package == "pytorch":
                    from unet3d.models.pytorch.build import build_or_load_model

                    model = build_or_load_model(model_filename=model_filename, model_name=config["model_name"],
                                                n_features=config["n_features"], n_outputs=config["n_outputs"],
                                                n_gpus=n_gpus, **model_kwargs)
                else:
                    model = load_model(model_filename)
            if output_task_name is None:
                _output_task_name = os.path.basename(metric_filenames[0]).split(".")[0]
                if len(metric_filenames) > 1:
                    _output_task_name = "_".join(
                        _output_task_name.split("_")[:2] + ["ALL47"] + _output_task_name.split("_")[3:])
            else:
                _output_task_name = output_task_name

            output_basename = "{task}-{model}_prediction.dscalar.nii".format(model=model_basename,
                                                                             task=_output_task_name)
            output_filename = os.path.join(output_directory, output_basename)
            subject_metric_names = list()
            for metric_list in config["metric_names"]:
                for metric_name in metric_list:
                    subject_metric_names.append(metric_name.format(subject_id))
            predict_subject(model,
                            feature_filename,
                            surface_filenames,
                            config['surface_names'],
                            subject_metric_names,
                            output_filename=output_filename,
                            batch_size=batch_size,
                            window=np.asarray(config['window']),
                            spacing=np.asarray(config['spacing']),
                            flip=False,
                            overwrite=overwrite,
                            use_multiprocessing=use_multiprocessing,
                            workers=n_workers,
                            max_queue_size=max_queue_size,
                            reference_filename=metric_filenames[0],
                            package=package,
                            generator=generator)


def predict_local_subject(model, feature_filename, surface_filename, batch_size=50, window=(64, 64, 64),
                          spacing=(1, 1, 1), flip=False, use_multiprocessing=False, workers=1, max_queue_size=10, ):
    generator = SubjectPredictionSequence(feature_filename=feature_filename, surface_filename=surface_filename,
                                          surface_name=None, batch_size=batch_size, window=window,
                                          flip=flip, spacing=spacing)
    return model.predict_generator(generator, use_multiprocessing=use_multiprocessing, workers=workers,
                                   max_queue_size=max_queue_size, verbose=1)


def whole_brain_scalar_predictions(model_filename, subject_ids, hcp_dir, output_dir, hemispheres, feature_basenames,
                                   surface_basename_template, target_basenames, model_name, n_outputs, n_features,
                                   window, criterion_name, metric_names, surface_names, reference, package="keras",
                                   n_gpus=1, n_workers=1, batch_size=1, model_kwargs=None):
    from .scripts.train import generate_hcp_filenames
    filenames = generate_hcp_filenames(directory=hcp_dir, surface_basename_template=surface_basename_template,
                                       target_basenames=target_basenames, feature_basenames=feature_basenames,
                                       subject_ids=subject_ids, hemispheres=hemispheres)
    if package == "pytorch":
        pytorch_whole_brain_scalar_predictions(model_filename=model_filename,
                                               model_name=model_name,
                                               n_outputs=n_outputs,
                                               n_features=n_features,
                                               filenames=filenames,
                                               prediction_dir=output_dir,
                                               window=window,
                                               criterion_name=criterion_name,
                                               metric_names=metric_names,
                                               surface_names=surface_names,
                                               reference=reference,
                                               n_gpus=n_gpus,
                                               n_workers=n_workers,
                                               batch_size=batch_size,
                                               model_kwargs=model_kwargs)
    else:
        raise ValueError("Predictions not yet implemented for {}".format(package))


def pytorch_whole_brain_scalar_predictions(model_filename, model_name, n_outputs, n_features, filenames, window,
                                           criterion_name, metric_names, surface_names, prediction_dir=None,
                                           output_csv=None, reference=None, n_gpus=1, n_workers=1, batch_size=1,
                                           model_kwargs=None):
    from .train.pytorch import load_criterion
    from unet3d.models.pytorch.build import build_or_load_model
    from .utils.pytorch.dataset import WholeBrainCIFTI2DenseScalarDataset
    import torch
    from torch.utils.data import DataLoader

    if model_kwargs is None:
        model_kwargs = dict()

    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_outputs=n_outputs,
                                n_features=n_features, n_gpus=n_gpus, **model_kwargs)
    model.eval()
    basename = os.path.basename(model_filename).split(".")[0]
    if prediction_dir and not output_csv:
        output_csv = os.path.join(prediction_dir, str(basename) + "_prediction_scores.csv")
    dataset = WholeBrainCIFTI2DenseScalarDataset(filenames=filenames,
                                                 window=window,
                                                 metric_names=metric_names,
                                                 surface_names=surface_names,
                                                 spacing=None,
                                                 batch_size=1)
    criterion = load_criterion(criterion_name, n_gpus=n_gpus)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    results = list()
    print("Loader: ", len(loader), "  Batch_size: ", batch_size, "  Dataset: ", len(dataset))
    with torch.no_grad():
        if reference is not None:
            reference = torch.from_numpy(reference).unsqueeze(0)
            if n_gpus > 0:
                reference = reference.cuda()
        for batch_idx, (x, y) in enumerate(loader):
            print("Batch: ", batch_idx)
            if n_gpus > 0:
                x = x.cuda()
                y = y.cuda()
            pred_y = model(x)
            if type(pred_y) == tuple:
                pred_y = pred_y[0]  # This is a hack to ignore other outputs that are used only for training
            for i in range(batch_size):
                row = list()
                idx = (batch_idx * batch_size) + i
                print("i: ", i, "  idx: ", idx)
                if idx >= len(dataset):
                    break
                args = dataset.filenames[idx]
                subject_id = args[-1]
                row.append(subject_id)
                idx_score = criterion(pred_y[i].unsqueeze(0), y[i].unsqueeze(0)).item()
                row.append(idx_score)
                if reference is not None:
                    idx_ref_score = criterion(reference.reshape(y[i].unsqueeze(0).shape),
                                              y[i].unsqueeze(0)).item()
                    row.append(idx_ref_score)
                results.append(row)
                save_predictions(prediction=pred_y[i].cpu().numpy(), args=args, basename=basename,
                                 metric_names=metric_names, surface_names=surface_names, prediction_dir=prediction_dir)

    if output_csv is not None:
        columns = ["subject_id", criterion_name]
        if reference is not None:
            columns.append("reference_" + criterion_name)
        pd.DataFrame(results, columns=columns).to_csv(output_csv)


def save_predictions(prediction, args, basename, metric_names, surface_names, prediction_dir):
    ref_filename = args[2][0]
    subject_id = args[-1]
    ref_basename = os.path.basename(ref_filename)
    prediction_name = "_".join((subject_id, basename, "prediction"))
    _metric_names = [_metric_name.format(prediction_name) for _metric_name in np.asarray(metric_names).ravel()]
    output_filename = os.path.join(prediction_dir, ref_basename.replace(subject_id, prediction_name))
    if prediction_dir is not None and not os.path.exists(output_filename):
        ref_cifti = nib.load(ref_filename)
        prediction_array = prediction.reshape(len(_metric_names),
                                              np.sum(ref_cifti.header.get_axis(1).surface_mask))
        cifti_file = new_cifti_scalar_like(prediction_array, _metric_names, surface_names, ref_cifti)
        cifti_file.to_filename(output_filename)


def pytorch_subject_predictions(idx, model, dataset, criterion, basename, prediction_dir, surface_names, metric_names,
                                n_gpus, reference):
    import torch
    with torch.no_grad():
        args = dataset.filenames[idx]
        ref_filename = args[2][0]
        subject_id = args[-1]
        ref_basename = os.path.basename(ref_filename)
        prediction_name = "_".join((subject_id, basename, "prediction"))
        _metric_names = [_metric_name.format(prediction_name) for _metric_name in np.asarray(metric_names).ravel()]
        output_filename = os.path.join(prediction_dir, ref_basename.replace(subject_id, prediction_name))
        x, y = dataset[idx]
        if os.path.exists(output_filename):
            prediction = torch.from_numpy(get_metric_data([nib.load(output_filename)],
                                                          [_metric_names],
                                                          surface_names,
                                                          subject_id)).float().cpu()
        else:
            prediction = model(x.unsqueeze(0))
        if n_gpus > 0:
            prediction = prediction.cpu()
        y = y.unsqueeze(0)
        score = criterion(prediction.reshape(y.shape), y).item()
        row = [subject_id, score]
        if reference is not None:
            reference_score = criterion(reference.reshape(y.shape), y).item()
            row.append(reference_score)

        if prediction_dir is not None and not os.path.exists(output_filename):
            ref_cifti = nib.load(ref_filename)
            prediction_array = prediction.numpy().reshape(len(_metric_names),
                                                          np.sum(ref_cifti.header.get_axis(1).surface_mask))
            cifti_file = new_cifti_scalar_like(prediction_array, _metric_names, surface_names, ref_cifti)
            cifti_file.to_filename(output_filename)
    return row


def single_volume_zstat_denoising(model_filename, model_name, n_features, filenames, window, prediction_dir,
                                  n_gpus=1, batch_size=1, model_kwargs=None, n_outputs=None,
                                  sequence_kwargs=None, spacing=None, sequence=None,
                                  strict_model_loading=True, metric_names=None,
                                  verbose=True, resample_predictions=False, **unused_kwargs):
    import torch
    model, dataset, basename = load_volumetric_model_and_dataset(model_name, model_filename, model_kwargs, n_outputs,
                                                                 n_features, strict_model_loading, n_gpus, sequence,
                                                                 sequence_kwargs, filenames, window, spacing,
                                                                 metric_names)
    dataset.extract_sub_volumes = False
    print("Dataset: ", len(dataset))
    with torch.no_grad():
        completed = set()
        batch = list()
        for idx in range(len(dataset)):
            x_filename, subject_id = get_feature_filename_and_subject_id(dataset, idx, verbose=False)
            while type(x_filename) == list:
                x_filename = x_filename[0]
            if x_filename in completed:
                continue
            if verbose:
                print("Reading:", x_filename)
            x_image, ref_image = load_images_from_dataset(dataset, idx, resample_predictions)
            if len(x_image.shape) == 4:
                volumes_per_image = x_image.shape[3]
                prediction_data = np.zeros(x_image.shape)
            else:
                volumes_per_image = 1
                prediction_data = np.zeros(x_image.shape + (volumes_per_image,))
            data = get_nibabel_data(x_image)
            for image_idx in range(volumes_per_image):
                batch.append(data[..., image_idx][..., None])
                if len(batch) >= batch_size or image_idx == volumes_per_image - 1:
                    prediction = pytorch_predict_batch_array(model, batch, n_gpus)
                    prediction = np.moveaxis(prediction, 0, -1).squeeze()
                    prediction_data[..., (image_idx - prediction.shape[-1] + 1):(image_idx + 1)] = prediction
                    batch = list()
            pred_image = new_img_like(ref_niimg=x_image, data=prediction_data)
            output_filename = os.path.join(prediction_dir, "_".join((subject_id,
                                                                     basename,
                                                                     os.path.basename(x_filename))))
            if verbose:
                print("Writing:", output_filename)
            pred_image.to_filename(output_filename)
            completed.add(x_filename)


def predictions_with_permutations(model_filename, model_name, n_features, filenames, window, prediction_dir=None,
                                  n_gpus=1, batch_size=1, model_kwargs=None, n_outputs=None, sequence_kwargs=None,
                                  spacing=None, sequence=None, strict_model_loading=True, metric_names=None,
                                  verbose=True, resample_predictions=False, interpolation="linear",
                                  output_template=None, segmentation=False, segmentation_labels=None,
                                  sum_then_threshold=True, threshold=0.5, label_hierarchy=None, permutation_weight=None,
                                  **unused_args):
    import torch
    model, dataset, basename = load_volumetric_model_and_dataset(model_name, model_filename, model_kwargs, n_outputs,
                                                                 n_features, strict_model_loading, n_gpus, sequence,
                                                                 sequence_kwargs, filenames, window, spacing,
                                                                 metric_names)
    permutation_keys = list(generate_permutation_keys())
    permutation_weights = np.ones((len(permutation_keys), 1, 1, 1, 1))
    # TODO: make this work with models that only output one prediction map
    if permutation_weight is not None:
        non_perm_index = permutation_keys.index(((0, 0), 0, 0, 0, 0))
        permutation_weights = permutation_weights * permutation_weight
        permutation_weights[non_perm_index] = len(permutation_keys) * (1 - permutation_weight)
    dataset.extract_sub_volumes = False
    print("Dataset: ", len(dataset))
    with torch.no_grad():
        for idx in range(len(dataset)):
            x_filename, subject_id = get_feature_filename_and_subject_id(dataset, idx, verbose=verbose)
            x_image, ref_image = load_images_from_dataset(dataset, idx, resample_predictions)
            data = get_nibabel_data(x_image)
            prediction_data = predict_with_permutations(model, data, n_outputs, batch_size, n_gpus, permutation_keys,
                                                        permutation_weights)
            pred_image = prediction_to_image(prediction_data.squeeze(),
                                             input_image=x_image,
                                             reference_image=ref_image,
                                             interpolation=interpolation,
                                             segmentation=segmentation,
                                             segmentation_labels=segmentation_labels,
                                             threshold=threshold,
                                             sum_then_threshold=sum_then_threshold,
                                             label_hierarchy=label_hierarchy)
            write_prediction_image_to_file(pred_image,
                                           output_template,
                                           subject_id=subject_id,
                                           x_filename=x_filename,
                                           prediction_dir=prediction_dir,
                                           basename=basename,
                                           verbose=verbose)


def predict_with_permutations(model, data, n_outputs, batch_size, n_gpus, permutation_keys, permutation_weights):
    import torch
    prediction_data = np.zeros((len(permutation_keys),) + data.shape[:3] + (n_outputs,))
    batch = list()
    permutation_indices = list()
    data = np.moveaxis(data, 3, 0)
    for permutation_idx, permutation_key in enumerate(permutation_keys):
        batch.append(permute_data(data, permutation_key))
        permutation_indices.append(permutation_idx)
        if len(batch) >= batch_size or permutation_key == permutation_keys[-1]:
            batch_prediction = pytorch_predict_batch(torch.tensor(batch).float(), model, n_gpus).numpy()
            for batch_idx, perm_idx in enumerate(permutation_indices):
                prediction_data[perm_idx] = np.moveaxis(reverse_permute_data(batch_prediction[batch_idx],
                                                                             permutation_keys[perm_idx]).squeeze(),
                                                        0, 3)
            batch = list()
            permutation_indices = list()
    # average over all the permutations
    return np.mean(prediction_data * permutation_weights, axis=0)


def predict_super_resolution(model_filename, model_name, n_features, filenames, window, prediction_dir=None,
                             n_gpus=1, batch_size=1, model_kwargs=None, n_outputs=None, sequence_kwargs=None,
                             spacing=None, sequence=None, strict_model_loading=True, metric_names=None,
                             verbose=True, resample_predictions=False, interpolation="linear",
                             output_template=None, segmentation=False, segmentation_labels=None,
                             sum_then_threshold=True, threshold=0.5, label_hierarchy=None, **unused_args):
    import torch
    new_window = list(np.asarray(window) * 2)
    model, dataset, basename = load_volumetric_model_and_dataset(model_name, model_filename, model_kwargs, n_outputs,
                                                                 n_features, strict_model_loading, n_gpus, sequence,
                                                                 sequence_kwargs, filenames, new_window, spacing,
                                                                 metric_names)
    dataset.extract_sub_volumes = False
    print("Dataset: ", len(dataset))
    with torch.no_grad():
        for idx in range(len(dataset)):
            x_filename, subject_id = get_feature_filename_and_subject_id(dataset, idx, verbose=verbose)
            x_image, ref_image = load_images_from_dataset(dataset, idx, resample_predictions)
            data = get_nibabel_data(x_image)
            prediction_data = predict_super_resolution_data(model, data, batch_size, n_gpus)
            pred_image = prediction_to_image(prediction_data.squeeze(),
                                             input_image=x_image,
                                             reference_image=ref_image,
                                             interpolation=interpolation,
                                             segmentation=segmentation,
                                             segmentation_labels=segmentation_labels,
                                             threshold=threshold,
                                             sum_then_threshold=sum_then_threshold,
                                             label_hierarchy=label_hierarchy)
            write_prediction_image_to_file(pred_image,
                                           output_template,
                                           subject_id=subject_id,
                                           x_filename=x_filename,
                                           prediction_dir=prediction_dir,
                                           basename=basename,
                                           verbose=verbose)


def predict_super_resolution_data(model, data, batch_size, n_gpus):
    batch = list()
    input_data = break_down_volume_into_half_size_volumes(data)
    predicted_data = list()
    for i, volume in enumerate(input_data):
        batch.append(volume)
        if len(batch) >= batch_size or i == (len(input_data) - 1):
            batch_prediction = pytorch_predict_batch_array(model=model, batch=batch, n_gpus=n_gpus)
            for batch_idx in range(batch_prediction.shape[0]):
                predicted_data.append(batch_prediction[batch_idx])
            batch = list()
    return combine_half_size_volumes(predicted_data)

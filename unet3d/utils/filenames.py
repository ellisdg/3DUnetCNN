import os
from functools import partial, update_wrapper

from unet3d import datasets
from unet3d.utils.utils import load_image, load_json

unet3d_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def generate_hcp_filenames(directory, surface_basename_template, target_basenames, feature_basenames, subject_ids,
                           hemispheres):
    rows = list()
    for subject_id in subject_ids:
        subject_id = str(subject_id)
        subject_dir = os.path.join(directory, subject_id)
        if type(feature_basenames) == str:
            feature_filenames = os.path.join(subject_dir, feature_basenames)
            if not os.path.exists(feature_filenames):
                continue
        else:
            feature_filenames = [os.path.join(subject_dir, fbn) for fbn in feature_basenames]
        if surface_basename_template is not None:
            surface_filenames = [os.path.join(subject_dir,
                                              surface_basename_template.format(hemi=hemi, subject_id=subject_id))
                                 for hemi in hemispheres]
        else:
            surface_filenames = None
        if type(target_basenames) == str:
            metric_filenames = os.path.join(subject_dir, target_basenames)
            if "{}" in metric_filenames:
                metric_filenames = metric_filenames.format(subject_id)
            if not os.path.exists(metric_filenames):
                continue
        elif target_basenames is not None:
            metric_filenames = [os.path.join(subject_dir, mbn.format(subject_id)) for mbn in target_basenames]
        else:
            metric_filenames = None
        rows.append([feature_filenames, surface_filenames, metric_filenames, subject_id])
    return rows


def generate_paired_filenames(directory, subject_ids, group, keys, basename, additional_feature_basename=None,
                              raise_if_not_exists=False):
    rows = list()
    pair = keys["all"]
    pair_key = list(keys["all"].keys())[0]
    volume_numbers = dict()
    for subject_id in subject_ids:
        subject_id = str(subject_id)
        template = os.path.join(directory, subject_id, basename)
        if additional_feature_basename is not None:
            additional_feature_filename = os.path.join(directory, subject_id, additional_feature_basename)
            if not os.path.exists(additional_feature_filename):
                if raise_if_not_exists:
                    raise FileNotFoundError(additional_feature_filename)
                continue
        else:
            additional_feature_filename = None
        for key in keys[group]:
            for value in keys[group][key]:
                format_kwargs1 = {key: value, pair_key: pair[pair_key][0]}
                format_kwargs2 = {key: value, pair_key: pair[pair_key][1]}
                filename1 = template.format(**format_kwargs1)
                filename2 = template.format(**format_kwargs2)
                if os.path.exists(filename1) and os.path.exists(filename2):
                    if value not in volume_numbers:
                        volume_numbers[value] = range(load_image(filename1, force_4d=True).shape[-1])
                    for volume_number in volume_numbers[value]:
                        if additional_feature_filename is not None:
                            rows.append([[additional_feature_filename, filename1], [0, volume_number + 1],
                                         filename2, [volume_number], subject_id])
                            rows.append([[additional_feature_filename, filename2], [0, volume_number + 1],
                                         filename1, [volume_number], subject_id])
                        else:
                            rows.append([filename1, [volume_number], filename2, [volume_number], subject_id])
                            rows.append([filename2, [volume_number], filename1, [volume_number], subject_id])
                elif raise_if_not_exists:
                    for filename in (filename1, filename2):
                        raise FileNotFoundError(filename)
    return rows


def format_templates(templates, directory="", **kwargs):
    if type(templates) == str:
        return os.path.join(directory, templates).format(**kwargs)
    else:
        return [os.path.join(directory, template).format(**kwargs) for template in templates]


def exists(filenames):
    if type(filenames) == str:
        filenames = [filenames]
    return all([os.path.exists(filename) for filename in filenames])


def generate_filenames_from_templates(subject_ids, feature_templates, target_templates=None, feature_sub_volumes=None,
                                      target_sub_volumes=None, raise_if_not_exists=False, directory="",
                                      skip_targets=False):
    filenames = list()
    for subject_id in subject_ids:
        feature_filename = format_templates(feature_templates, directory=directory, subject=subject_id)
        if skip_targets:
            target_filename = None
        else:
            target_filename = format_templates(target_templates, directory=directory, subject=subject_id)
        if feature_sub_volumes is not None:
            _feature_sub_volumes = feature_sub_volumes
        else:
            _feature_sub_volumes = None
        if target_sub_volumes is not None:
            _target_sub_volumes = target_sub_volumes
        else:
            _target_sub_volumes = None
        if exists(feature_filename) and (skip_targets or exists(target_filename)):
            filenames.append([feature_filename, _feature_sub_volumes, target_filename, _target_sub_volumes, subject_id])
        elif raise_if_not_exists:
            for filename in (feature_filename, target_filename):
                if not exists(filename):
                    raise FileNotFoundError(filename)
    return filenames


def generate_filenames_from_multisource_templates(subject_ids, feature_templates, target_templates,
                                                  feature_sub_volumes=None, target_sub_volumes=None,
                                                  raise_if_not_exists=False, directory=""):
    filenames = dict()
    for dataset in subject_ids:
        filenames[dataset] = generate_filenames_from_templates(subject_ids[dataset],
                                                               feature_templates[dataset],
                                                               target_templates[dataset],
                                                               (feature_sub_volumes[dataset]
                                                                if feature_sub_volumes is not None else None),
                                                               (target_sub_volumes[dataset]
                                                                if target_sub_volumes is not None else None),
                                                               raise_if_not_exists=raise_if_not_exists,
                                                               directory=directory)
    return filenames


def generate_filenames(config, name, directory="", skip_targets=False, raise_if_not_exists=False):
    if name not in config:
        load_subject_ids(config, name)
    if "generate_filenames" not in config or config["generate_filenames"] == "classic":
        return generate_hcp_filenames(directory,
                                      config['surface_basename_template']
                                      if "surface_basename_template" in config else None,
                                      config['target_basenames'],
                                      config['feature_basenames'],
                                      config[name],
                                      config['hemispheres'] if 'hemispheres' in config else None)
    elif config["generate_filenames"] == "paired":
        return generate_paired_filenames(directory,
                                         config[name],
                                         name,
                                         raise_if_not_exists=raise_if_not_exists,
                                         **config["generate_filenames_kwargs"])
    elif config["generate_filenames"] == "multisource_templates":
        return generate_filenames_from_multisource_templates(config[name],
                                                             raise_if_not_exists=raise_if_not_exists,
                                                             **config["generate_filenames_kwargs"])
    elif config["generate_filenames"] == "templates":
        return generate_filenames_from_templates(config[name],
                                                 raise_if_not_exists=raise_if_not_exists,
                                                 **config["generate_filenames_kwargs"],
                                                 skip_targets=skip_targets)


def load_subject_ids(config, name):
    if "subjects_filename" in config:
        subjects = load_json(os.path.join(unet3d_path, config["subjects_filename"]))
        config[name] = subjects[name]


def load_bias(bias_filename):
    import numpy as np
    return np.fromfile(os.path.join(unet3d_path, bias_filename))


def load_dataset_class(dataset_kwargs, cache_dir="./cache"):
    if "Persistent" in dataset_kwargs["name"] and "cache_dir" not in dataset_kwargs:
        dataset_kwargs["cache_dir"] = os.path.abspath(cache_dir)
    return getattr(datasets, dataset_kwargs["name"])

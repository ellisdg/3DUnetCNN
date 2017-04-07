"""
Tools for converting, normalizing, and fixing the brats data.

Correcting the bias requires that N4BiasFieldCorrection be installed!
"""


import os
import glob

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from config import config


def append_basename(in_file, append):
    dirname, basename = os.path.split(in_file)
    base, ext = basename.split(".", 1)
    return os.path.join(dirname, base + append + "." + ext)


def get_background_mask(in_folder, out_file):
    background_image = None
    for name in config["modalities"] + [".OT"]:
        image = sitk.ReadImage(get_image(in_folder, name))
        if background_image:
            if name == ".OT" and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)


def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image, np.percentile(image_data, min_percent), np.percentile(image_data,
                                                                                                     max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)


def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def get_image(subject_folder, name):
    return glob.glob(os.path.join(subject_folder, "*" + name + ".*", "*" + name + ".*.mha"))[0]


def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
    sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, background_mask):
    converted = convert_image_format(in_file, append_basename(out_file, "_converted"))
    initial_rescale = rescale(converted, append_basename(out_file, "_initial_rescale"))
    zeroed = background_to_zero(initial_rescale, background_mask, append_basename(out_file, "_zeroed"))
    windowed = window_intensities(zeroed, append_basename(out_file, "_windowed"))
    corrected = correct_bias(windowed, append_basename(out_file, "_corrected"))
    rescaled = rescale(corrected, out_file, maximum=1)
    for f in [converted, initial_rescale, zeroed, windowed, corrected]:
        os.remove(f)
    return rescaled


def convert_brats_folder(in_folder, out_folder, background_mask):
    for name in config["modalities"] + [".OT"]:
        image_file = get_image(in_folder, name)
        if name == ".OT":
            out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
            converted = convert_image_format(image_file, out_file)
            check_origin(converted, background_mask)
        else:
            out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
            normalize_image(image_file, out_file, background_mask)


def convert_brats_data(brats_folder, out_folder):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder):
                os.makedirs(new_subject_folder)
            else:
                continue
            background_mask = get_background_mask(subject_folder,
                                                  os.path.join(new_subject_folder, "background.nii.gz"))
            convert_brats_folder(subject_folder, new_subject_folder, background_mask)

import monai
from monai.data import PersistentDataset, Dataset
from monai.transforms import (LoadImageD, ResizeWithPadOrCropD, Compose, NormalizeIntensityD, ResizeD, CropForegroundD,
                              RandSpatialCropD, OrientationD)
from unet3d.transforms import LabelMapToOneHotD
from unet3d.utils.threshold import percentile_threshold
from unet3d.utils.utils import get_class, get_kwargs

from functools import partial


def _build_segmentation_transform(
        filenames,
        labels=None,
        inference="auto",
        desired_shape=None,
        normalization="zero_mean",
        normalization_kwargs=None,
        crop_foreground=False,
        foreground_percentile=0.1,
        random_crop=False,
        resample=False,
        intensity_augmentations=None,
        spatial_augmentations=None,
        orientation=None,
        reader=None,
):
    """Build the monai Compose transform pipeline shared by both dataset variants.

    Parameters mirror those of the dataset __init__ functions. See `SegmentationDatasetPersistent` for docs.
    """
    transforms = []
    if inference == "auto":
        # Look at the first set and determine if label is present
        inference = "label" not in filenames[0].keys()

    if inference:
        keys = ("image",)
    else:
        keys = ("image", "label")

    if reader is not None:
        transforms.append(LoadImageD(keys=keys, image_only=True, ensure_channel_first=True, reader=reader))
    else:
        transforms.append(LoadImageD(keys=keys, image_only=True, ensure_channel_first=True))

    if orientation:
        transforms.append(OrientationD(keys=keys, axcodes=orientation, lazy=True))

    if not inference:
        if labels is None:
            raise ValueError("Must set 'labels' for segmentation dataset when not in inference mode.")
        transforms.append(LabelMapToOneHotD(keys=("label",), labels=labels))

    if crop_foreground:
        foreground_func = partial(percentile_threshold, percentile=foreground_percentile)
        transforms.append(CropForegroundD(keys=keys, source_key="image", select_fn=foreground_func,
                                          lazy=True, margin=1))

    if desired_shape:
        if random_crop:
            transforms.append(RandSpatialCropD(keys=keys, roi_size=desired_shape, random_size=False, lazy=True))
        elif resample:
            if inference:
                mode = ("trilinear",)
            else:
                mode = ("trilinear", "nearest")
            transforms.append(ResizeD(keys=keys, spatial_size=desired_shape, mode=mode, lazy=True))
        else:
            transforms.append(ResizeWithPadOrCropD(keys=keys, spatial_size=desired_shape, lazy=True))

    if spatial_augmentations is not None:
        for augmentation in spatial_augmentations:
            transforms.append(get_class(augmentation, monai.transforms)(keys=keys, lazy=True,
                                                                        **get_kwargs(augmentation)))

    if normalization_kwargs is None:
        normalization_kwargs = {}
    if normalization is not None:
        if normalization == "zero_mean":
            normalization_class = NormalizeIntensityD
        else:
            try:
                normalization_class = getattr(monai.transforms, normalization)
            except ValueError:
                raise ValueError(f"{normalization} normalization method not yet implemented")
        transforms.append(normalization_class(keys=("image",), **normalization_kwargs))

    if intensity_augmentations is not None:
        for augmentation in intensity_augmentations:
            transforms.append(get_class(augmentation, monai.transforms)(keys=("image",),
                                                                        **get_kwargs(augmentation)))

    return Compose(transforms, lazy=True)


class SegmentationDataset(Dataset):
    def __init__(self, filenames, labels=None, inference="auto", desired_shape=None,
                 normalization="zero_mean", normalization_kwargs=None, crop_foreground=False,
                 foreground_percentile=0.1, random_crop=False, resample=False, intensity_augmentations=None,
                 spatial_augmentations=None, orientation=None, reader=None):
        """Non-persistent version of the segmentation dataset.

        Parameters are identical to `SegmentationDatasetPersistent` except `cache_dir` is omitted.
        """
        transform = _build_segmentation_transform(
            filenames=filenames,
            labels=labels,
            inference=inference,
            desired_shape=desired_shape,
            normalization=normalization,
            normalization_kwargs=normalization_kwargs,
            crop_foreground=crop_foreground,
            foreground_percentile=foreground_percentile,
            random_crop=random_crop,
            resample=resample,
            intensity_augmentations=intensity_augmentations,
            spatial_augmentations=spatial_augmentations,
            orientation=orientation,
            reader=reader,
        )
        super().__init__(data=filenames, transform=transform)


class SegmentationDatasetPersistent(PersistentDataset):
    # TODO: make a base class that includes the features that would be standard even for non-segmentation cases
    def __init__(self, filenames, cache_dir, labels=None, inference="auto", desired_shape=None,
                 normalization="zero_mean", normalization_kwargs=None, crop_foreground=False,
                 foreground_percentile=0.1, random_crop=False, resample=False, intensity_augmentations=None,
                 spatial_augmentations=None, orientation=None, reader=None):
        transform = _build_segmentation_transform(
            filenames=filenames,
            labels=labels,
            inference=inference,
            desired_shape=desired_shape,
            normalization=normalization,
            normalization_kwargs=normalization_kwargs,
            crop_foreground=crop_foreground,
            foreground_percentile=foreground_percentile,
            random_crop=random_crop,
            resample=resample,
            intensity_augmentations=intensity_augmentations,
            spatial_augmentations=spatial_augmentations,
            orientation=orientation,
            reader=reader,
        )
        super().__init__(data=filenames, cache_dir=cache_dir, transform=transform)

import monai
from monai.data import PersistentDataset
from monai.transforms import LoadImageD, ResizeWithPadOrCropD, Compose, NormalizeIntensityD, ResizeD, CropForegroundD
from unet3d.transforms import LabelMapToOneHotD
from unet3d.utils.threshold import percentile_threshold

from functools import partial


class SegmentationDatasetPersistent(PersistentDataset):
    def __init__(self, filenames, cache_dir, labels=None, inference="auto", desired_shape=None,
                 normalization="zero_mean", normalization_kwargs=None, resample=False, crop_foreground=False,
                 foreground_percentile=0.1):
        transforms = list()
        if inference == "auto":
            # Look at the first set and determine if labels are present
            inference = "label" not in filenames[0].keys()

        if inference:
            keys = ("image",)
        else:
            keys = ("image", "label")
        transforms.append(LoadImageD(keys=keys, image_only=True, ensure_channel_first=True))

        if crop_foreground:
            foreground_func = partial(percentile_threshold, percentile=foreground_percentile)
            transforms.append(CropForegroundD(keys=keys, source_key="image", select_fn=foreground_func,
                                              lazy=True, margin=1))

        if desired_shape and not resample:
            transforms.append(ResizeWithPadOrCropD(keys=keys, spatial_size=desired_shape, lazy=True))
        elif desired_shape:
            mode = ("trilinear", "nearest")
            transforms.append(ResizeD(keys=keys, spatial_size=desired_shape, mode=mode[:len(keys)], lazy=True))

        if normalization_kwargs is None:
            normalization_kwargs = dict()
        if normalization is not None:
            if normalization == "zero_mean":
                normalization_class = NormalizeIntensityD
                transforms.append(NormalizeIntensityD(keys=("image",), **normalization_kwargs))
            else:
                try:
                    normalization_class = getattr(monai.transforms, normalization)
                except ValueError:
                    raise ValueError("{} normalization method not yet implemented".format(normalization))
            transforms.append(normalization_class(keys=("image",), **normalization_kwargs))

        if not inference:
            if labels is None:
                raise ValueError("Must set 'labels' for {}".format(self.__class__))
            transforms.append(LabelMapToOneHotD(keys=("label",), labels=labels))

        transform = Compose(transforms, lazy=True)
        super().__init__(data=filenames, cache_dir=cache_dir, transform=transform)

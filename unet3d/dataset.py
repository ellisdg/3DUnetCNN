import monai
from monai.data import PersistentDataset
from monai.transforms import LoadImageD, ResizeWithPadOrCropD, Compose, NormalizeIntensityD
from unet3d.transforms import LabelMapToOneHotD


class SegmentationDatasetPersistent(PersistentDataset):
    def __init__(self, filenames, cache_dir, labels=None, inference="auto", desired_shape=None,
                 normalization="zero_mean", normalization_kwargs=None):
        transforms = list()
        if inference == "auto":
            # Look at the first set and determine if labels are present
            inference = "label" not in filenames[0].keys()

        if inference:
            keys = ("image",)
        else:
            keys = ("image", "label")
        transforms.append(LoadImageD(keys=keys, image_only=True, ensure_channel_first=True))

        if desired_shape:
            transforms.append(ResizeWithPadOrCropD(keys=keys, spatial_size=desired_shape, lazy=True))

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

from monai.data import PersistentDataset
from monai.transforms import LoadImageD, ResizeWithPadOrCropD, Compose
from unet3d.transforms import LabelMapToOneHotD


class SegmentationDatasetPersistent(PersistentDataset):
    def __init__(self, filenames, cache_dir, labels=None, inference="auto", desired_shape=None):
        transforms = list()
        if inference == "auto":
            # Look at the first set and determine if labels are present
            inference = "labels" in filenames[0].keys()

        if inference:
            keys = ("image",)
        else:
            keys = ("image", "labels")
        transforms.append(LoadImageD(keys=keys, image_only=True, ensure_channel_first=True))

        if desired_shape:
            transforms.append(ResizeWithPadOrCropD(keys=keys, spatial_size=desired_shape, lazy=True))

        if not inference:
            if labels is None:
                raise ValueError("Must set 'labels' for {}".format(self.__class__))
            transforms.append(LabelMapToOneHotD(keys=("labels",), labels=labels))

        transform = Compose(transforms, lazy=True)
        super().__init__(data=filenames, cache_dir=cache_dir, transform=transform)

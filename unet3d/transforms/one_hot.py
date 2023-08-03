from monai.transforms import Transform, MapTransform
from monai.transforms.utils import TransformBackends
from unet3d.utils.one_hot import compile_one_hot_encoding
import torch


class LabelMapToOneHot(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(self, labels, dtype=torch.uint8):
        self.labels = labels
        self.dtype = dtype
        super().__init__()

    def __call__(self, image):
        return compile_one_hot_encoding(image, n_labels=len(self.labels), labels=self.labels, dtype=self.dtype)


class LabelMapToOneHotD(MapTransform):
    backend = LabelMapToOneHot.backend

    def __init__(self, keys, labels, allow_missing_keys: bool = False, **kwargs):
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelMapToOneHot(labels=labels, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

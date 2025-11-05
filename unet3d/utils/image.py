import torch
from monai.data.meta_tensor import MetaTensor


class Image(MetaTensor):

    def make_similar(self, data, affine=None, copy_meta=True):
        if affine is None:
            if hasattr(data, "affine") and data.affine is not None:
                affine = data.affine
            else:
                affine = self.affine
        if hasattr(data, "array"):
            data = data.array
        if copy_meta:
            # Copy instance meta, not the class property, and drop affine from meta as it is passed separately
            try:
                meta = dict(self.meta) if getattr(self, "meta", None) is not None else {}
            except Exception:
                meta = {}
            meta.pop("affine", None)
        else:
            meta = dict()
        return Image(x=data, affine=affine, meta=meta)

    def to_filename(self, filename):
        import nibabel as nib
        if len(self.shape) > 3:
            _data = torch.moveaxis(self, 0, -1)
        else:
            _data = self
        _image = nib.Nifti1Image(dataobj=_data.squeeze().numpy(), affine=self.affine.cpu().numpy())
        _image.to_filename(filename)


def get_image(data, affine, **kwargs):
    return Image(x=data, affine=affine, **kwargs)

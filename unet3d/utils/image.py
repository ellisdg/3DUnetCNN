import torch
from monai.data.meta_tensor import MetaTensor


class Image(MetaTensor):

    def make_similar(self, data, affine=None):
        if affine is None:
            if hasattr(data, "affine") and data.affine is not None:
                affine = data.affine
            else:
                affine = self.affine
        return Image(x=data, affine=affine, meta=self.meta)

    def to_filename(self, filename):
        import nibabel as nib
        if len(self.shape) > 3:
            _data = torch.moveaxis(self, 0, -1)
        else:
            _data = self
        _image = nib.Nifti1Image(dataobj=_data.squeeze().numpy(), affine=self.affine)
        _image.to_filename(filename)


def get_image(data, affine, **kwargs):
    return Image(x=data, affine=affine, **kwargs)

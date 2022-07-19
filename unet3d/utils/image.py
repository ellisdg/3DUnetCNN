import torch


class Image(object):
    def __init__(self, dataobj, affine):
        self.dataobj = dataobj
        self._affine = affine

    @property
    def affine(self):
        return self._affine

    @property
    def shape(self):
        return self.dataobj.shape

    def get_data(self):
        return self.dataobj

    def get_fdata(self):
        return self.dataobj.float()

    def set_dtype(self, dtype):
        self.dataobj = self.dataobj.to(dtype)

    def update(self, data, affine):
        self.set_data(data)
        self.set_affine(affine)

    def set_data(self, data):
        self.dataobj = data

    def set_affine(self, affine):
        self._affine = affine

    def make_similar(self, data, affine=None):
        if affine is None:
            affine = self.affine
        return Image(data, affine)

    def copy(self):
        return self.make_similar(self.get_data().detach().clone(),
                                 self._affine.detach().clone())

    def to_filename(self, filename):
        import nibabel as nib
        if len(self.shape) > 3:
            _data = torch.moveaxis(self.dataobj, 0, -1)
        else:
            _data = self.dataobj
        _image = nib.Nifti1Image(dataobj=_data.squeeze().numpy(), affine=self.affine)
        _image.to_filename(filename)


def get_image(data, affine):
    return Image(dataobj=data, affine=affine)

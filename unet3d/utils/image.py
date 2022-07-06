

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


def get_image(data, affine):
    return Image(dataobj=data, affine=affine)

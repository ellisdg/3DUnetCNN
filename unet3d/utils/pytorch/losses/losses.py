"""
Put custom loss classes here.
"""

from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, dim=(0, 2, 3, 4), smooth=1., generalized=False):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.dim = dim
        self.smooth = smooth
        self.generalized = generalized

    def forward(self, x, y):
        # input and target shapes must match
        assert x.size() == y.size(), "'input' ({}) and 'target' ({}) must have the same shape".format(
            x.size(), y.size())

        # compute per channel Dice Coefficient
        tp = (x * y).sum(dim=self.dim)
        if self.weight is not None:
            tp = self.weight * tp

        fn = ((1 - x) * y).sum(dim=self.dim)
        fp = (x * (1 - y)).sum(dim=self.dim)
        if self.generalized:
            vol = y.sum(dim=self.dim) + self.smooth
            tp = tp / vol
            fn = fn / vol
            fp = fp / vol
        denominator = (2 * tp + fn + fp)

        return 1 - ((2 * tp + self.smooth) / (denominator + self.smooth)).mean()


MyDiceLoss = DiceLoss  # this was the original name for this class

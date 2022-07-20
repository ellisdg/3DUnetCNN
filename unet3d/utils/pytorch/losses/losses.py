"""
Put custom loss classes here.
"""

from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, weight=None, dim=(0, 2, 3, 4), v_net=False,
                 smooth=1., generalized=True):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.dim = dim
        self.v_net = v_net
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

        if self.v_net:
            denominator = (x * x).sum(dim=self.dim) + (y * y).sum(dim=self.dim)
        else:
            fn = ((x == 0) * y).sum(dim=self.dim)
            fp = (x * (y == 0)).sum(dim=self.dim)
            if self.generalized:
                vol = y.sum(dim=self.dim)
                tp = tp / vol
                fn = fn / vol
                fp = fp / vol
            denominator = (2 * tp + fn + fp)

        return 1 - ((2 * tp + self.smooth) / (denominator + self.smooth)).mean()


MyDiceLoss = DiceLoss  # this was the original name for this class

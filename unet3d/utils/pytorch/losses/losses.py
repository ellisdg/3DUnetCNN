"""
Put custom loss classes here.
"""

from torch import nn


class MyDiceLoss(nn.Module):
    # modified from wolny's GitHub project: https://github.com/wolny/pytorch-3dunet
    def __init__(self, epsilon=1e-6, weight=None, dim=(0, 2, 3, 4)):
        super(MyDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.dim = dim

    def forward(self, x, y):
        # input and target shapes must match
        assert x.size() == y.size(), "'input' ({}) and 'target' ({}) must have the same shape".format(
            x.size(), y.size())

        # compute per channel Dice Coefficient
        tp = (x * y).sum(dim=self.dim)
        if self.weight is not None:
            tp = self.weight * tp
        fn = ((x == 0) * y).sum(dim=self.dim)
        fp = (x * (y == 0)).sum(dim=self.dim)

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (2 * tp + fn + fp)
        return 1 - (2 * (tp / denominator.clamp(min=self.epsilon))).mean()

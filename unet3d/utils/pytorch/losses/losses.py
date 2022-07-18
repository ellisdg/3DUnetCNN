"""
Put custom loss classes here.
"""

from torch import nn


class MyDiceLoss(nn.Module):
    # modified from wolny's GitHub project: https://github.com/wolny/pytorch-3dunet
    def __init__(self, epsilon=1e-6, weight=None):
        super(MyDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, x, y):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
             x (torch.Tensor): NxCxSpatial input tensor
             y (torch.Tensor): NxCxSpatial target tensor
             epsilon (float): prevents division by zero
             weight (torch.Tensor): Cx1 tensor of weight per channel/class
        """

        # input and target shapes must match
        assert x.size() == y.size(), "'input' ({}) and 'target' ({}) must have the same shape".format(
            x.size(), y.size())

        x = flatten(x)
        y = flatten(y)
        y = y.float()

        # compute per channel Dice Coefficient
        intersect = (x * y).sum(-1)
        if self.weight is not None:
            intersect = self.weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (x * x).sum(-1) + (y * y).sum(-1)
        return 2 * (intersect / denominator.clamp(min=self.epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

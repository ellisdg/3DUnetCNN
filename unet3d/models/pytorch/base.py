from torch import nn
import abc
import warnings


# class Module(nn.Module):
#     __metaclass__ = abc.ABCMeta
#
#     def __init__(self, autocast=False):
#         super(Module).__init__()
#         self.autocast = autocast
#
#     def forward(self, *input):
#         if self.autocast:
#             try:
#                 from torch.cuda.amp import autocast
#                 with autocast():
#                     output = self._forward(*input)
#                 return output
#             except ImportError:
#                 warnings.warn("Could not import autocast from PyTorch. Upgrade to PyTorch >= 1.6 or set autocast"
#                               " to False.")
#
#         return self._forward(*input)
#
#     @abc.abstractmethod
#     def _forward(self, *input):
#         raise NotImplementedError

"""
This file contains definitions of layers used to build the GraphCNN
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, adjacency_matrix_wrapper, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix_wrapper = adjacency_matrix_wrapper
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjacency_matrix_wrapper.adjacency_matrix, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjacency_matrix, support))
                output.append(spmm(self.adjacency_matrix_wrapper.adjacency_matrix, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphLinear(nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]


class GraphResBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, adjacency_matrix_wrapper):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, adjacency_matrix_wrapper)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))

    def forward(self, x):
        y = F.relu(self.pre_norm(x))
        y = self.lin1(y)

        y = F.relu(self.norm1(y))
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)

        y = F.relu(self.norm2(y))
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x+y


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)

"""
Modified from:
https://github.com/nkolot/GraphCMR/blob/master/models/graph_cnn.py
"""


from __future__ import division

import torch
import torch.nn as nn

from .graph_cmr_layers import GraphResBlock, GraphLinear
from .utils import load_surface, AdjacencyMatrixWrapper
from ..classification.resnet import resnet_18


class GraphCMR(nn.Module):
    def __init__(self, n_outputs=None, ref_vertices=None, adjacency_matrix=None, n_layers=5, n_channels=256,
                 output_features=3, encoder=resnet_18, encoder_outputs=512, reference_filename=None, **encoder_kwargs):
        super(GraphCMR, self).__init__()
        if reference_filename is not None and (ref_vertices is None or adjacency_matrix is None):
            ref_vertices, adjacency_matrix = load_surface(surface_filename=reference_filename)
        self.adjacency_matrix_wrapper = AdjacencyMatrixWrapper(adjacency_matrix)
        self.ref_vertices = ref_vertices
        self.encoder = encoder(n_outputs=encoder_outputs, **encoder_kwargs)
        self.encoder_outputs = encoder_outputs
        layers = [GraphLinear(3 + self.encoder_outputs, 2 * n_channels),
                  GraphResBlock(2 * n_channels, n_channels, self.adjacency_matrix_wrapper)]
        for i in range(n_layers):
            layers.append(GraphResBlock(n_channels, n_channels, self.adjacency_matrix_wrapper))
        self.shape = nn.Sequential(GraphResBlock(n_channels, 64, self.adjacency_matrix_wrapper),
                                   GraphResBlock(64, 32, self.adjacency_matrix_wrapper),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, output_features))
        self.gc = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass
        Inputs:
            x: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = x.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        x = self.encoder(x)
        x = x.view(batch_size, self.encoder_outputs, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, x], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        return shape

    def cuda(self, *args, **kwargs):
        self.ref_vertices = self.ref_vertices.cuda(*args, **kwargs)
        self.adjacency_matrix_wrapper.adjacency_matrix = self.adjacency_matrix_wrapper.adjacency_matrix.cuda(*args,
                                                                                                             **kwargs)
        return super(GraphCMR, self).cuda(*args, **kwargs)

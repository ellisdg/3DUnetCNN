from itertools import permutations
import nibabel as nib
import torch
import os


def load_surface(surface_filename):
    surface = nib.load(os.path.abspath(surface_filename))
    vertices = surface.darrays[0].data
    n_vertices = vertices.shape[0]
    faces = surface.darrays[1].data
    adjacency_matrix = faces_to_adjacency_matrix(faces, size=(n_vertices, n_vertices))
    return torch.FloatTensor(vertices).t(), adjacency_matrix


def faces_to_edges(faces):
    edges = list()
    for face in faces:
        edges.extend(list(permutations(face, 2)))
    return torch.LongTensor(edges).t()


def faces_to_adjacency_matrix(faces, size):
    indices = faces_to_edges(faces)
    values = torch.zeros(indices.shape[1], dtype=torch.float)
    adjacency_matrix = torch.sparse.FloatTensor(indices, values, size=size)
    return adjacency_matrix


class AdjacencyMatrixWrapper(object):
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

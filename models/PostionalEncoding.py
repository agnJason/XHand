import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial


def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


def scale_input(tensor, transform=None, scale=None):
    if transform is not None:
        t_shape = tensor.shape
        tensor = transform_3D_grid(
            tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor


class PostionalEncoding(torch.nn.Module):
    def __init__(
            self,
            min_deg=0,
            max_deg=3,
            scale=0.1,
            transform=None,
            input_channle=3,
    ):
        super(PostionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.transform = transform

        self.input_channle = input_channle
        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, self.input_channle).T

        frequency_bands = 2.0 ** np.linspace(
            self.min_deg, self.max_deg, self.n_freqs)
        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + self.input_channle

        print(
            "Icosahedron embedding with periods:",
            (2 * np.pi) / (frequency_bands * self.scale),
            " -- embedding size:", self.embedding_size
        )

    def vis_embedding(self):
        x = torch.linspace(0, 5, 640)
        embd = x * self.scale
        if self.gauss_embed:
            frequency_bands = torch.norm(self.B_layer.weight, dim=1)
            frequency_bands = torch.sort(frequency_bands)[0]
        else:
            frequency_bands = 2.0 ** torch.linspace(
                self.min_deg, self.max_deg, self.n_freqs)

        embd = embd[..., None] * frequency_bands
        embd = torch.sin(embd)

        import matplotlib.pylab as plt
        plt.imshow(embd.T, cmap='hot', interpolation='nearest',
                   aspect='auto', extent=[0, 5, 0, embd.shape[1]])
        plt.colorbar()
        plt.xlabel("x values")
        plt.ylabel("embedings")
        plt.show()

    def forward(self, tensor):
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg, self.max_deg, self.n_freqs,
            dtype=tensor.dtype, device=tensor.device)

        tensor = scale_input(
            tensor, transform=self.transform, scale=self.scale)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands,
            list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding


class PositionalEncoding_nerf(object):
    def __init__(self, L=10, dim=3):
        self.L = L
        self.embedding_size = dim * L * 2 + dim

    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p),
             torch.cos((2 ** i) * pi * p)],
            dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)

if __name__ == '__main__':
    pe = PostionalEncoding()
    pe_nerf = PositionalEncoding_nerf()
    inputs = torch.zeros(1, 512, 512, 3)
    print(pe(inputs).shape)
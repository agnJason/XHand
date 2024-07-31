import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial


def get_ray_directions(H, W, fx, fy, cx, cy):
    # grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    # i, j = grid.unbind(-1)

    O = 0.5
    x_coords = torch.linspace(O, W - 1 + O, W)
    y_coords = torch.linspace(O, H - 1 + O, H)
    j, i = torch.meshgrid([y_coords, x_coords], indexing='ij')
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    # j = j.cuda()
    # i = i.cuda()
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3]  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_d, rays_o


if __name__ == "__main__":
    H, W = SIZE
    fx, fy, cx, cy = intrinsic_gt[:, 0, 0], intrinsic_gt[:, 1, 1], intrinsic_gt[:, 0, 2], intrinsic_gt[:, 1, 2]
    ray_direction = torch.ones((fx.shape[0], H, W, 3), device=vertsw.device, requires_grad=False)
    for bi in range(fx.shape[0]):
        cam_ray_direction = get_ray_directions(H, W, fx[bi], fy[bi], cx[bi], cy[bi])
        c2w = torch.inverse(w2cs[bi])
        tmp_ray_direction, _ = get_rays(cam_ray_direction, c2w)
        ray_direction[bi] = tmp_ray_direction
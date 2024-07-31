import numpy as np
import trimesh
import torch
from models.smplx import SMPLX, batch_rodrigues, batch_global_rigid_transformation
import pickle
from get_data import mano_layer

def subdivide_weight(weights, faces):
    weight_sub = np.zeros((faces.max()+1, weights.shape[1]))
    weight_sub[:weights.shape[0]] = weights
    for i in range(int(faces.shape[0]/4)):
        tmp = faces[i*4: i*4+4]
        if weight_sub[tmp[0][1]].sum() != 0:
            pass
        weight_sub[tmp[0][1]] = (weight_sub[tmp[0][0]] + weight_sub[tmp[1][1]]) / 2
        weight_sub[tmp[0][2]] = (weight_sub[tmp[0][0]] + weight_sub[tmp[2][2]]) / 2
        weight_sub[tmp[1][2]] = (weight_sub[tmp[1][1]] + weight_sub[tmp[2][2]]) / 2
    return weight_sub

def subdivide_weight_loop(weights, vertices, faces, iterations=3):
    for i in range(iterations):
        vertices, faces = trimesh.remesh.subdivide_loop(vertices, faces, iterations=1)
        weights = subdivide_weight(weights, faces)
    return vertices, faces, weights

def save_sub_weights():
    out = {}
    for hand_type in ['left', 'right']:
        _, faces_tmp, new_weights = subdivide_weight_loop(mano_layer[hand_type].lbs_weights.numpy(),
                                                          mano_layer[hand_type].v_template.numpy(),
                                                          mano_layer[hand_type].faces.astype(np.int64),
                                                          iterations=3)
        out.update({hand_type:{'faces': faces_tmp, 'weights': new_weights}})
    with open('mano/mano_weight_sub3.pkl', 'wb') as f:
        pickle.dump(out, f)

def pose2rot(pose, hand_type='right'):
    b = pose.shape[0]
    pose = (pose + mano_layer[hand_type].pose_mean.to(pose.device)).clone()
    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)
    return R

def lbs_pose(pose, v_shaped, weights, verts_tpose, hand_type='right'):
    mano = mano_layer[hand_type].to(v_shaped.device)
    b = pose.shape[0]
    device = pose.device
    # dtype = shape.dtype
    # v_template = mano.v_template
    # v_shaped = torch.einsum('bl,mkl->bmk', [shape, mano.shapedirs]) + v_template

    J = torch.einsum('bik,ji->bjk', [v_shaped, mano.J_regressor])

    num_joints = 16

    pose = (pose + mano.pose_mean).clone()

    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)

    # lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
    # v_posed = torch.matmul(lrotmin, smplx.posedirs).reshape(b, smplx.size[0], smplx.size[1]) + v_shaped  # smpl_v_posed

    J_transformed, A = batch_global_rigid_transformation(R, J, mano.parents)

    weights = weights.expand(b, -1, -1)    # [b, num_v, num_j]
    T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)  # [b, num_v, 4, 4]

    verts_homo = torch.cat([verts_tpose, torch.ones(b, verts_tpose.shape[1], 1, device=device)], dim=2)
    verts = torch.matmul(T, verts_homo.unsqueeze(-1))

    verts = verts[:, :, :3, 0]
    return verts

def lbs(pose, shape, weights, verts_tpose, hand_type='right'):
    mano = mano_layer[hand_type]
    b = pose.shape[0]
    device = pose.device
    dtype = shape.dtype
    v_template = mano.v_template
    v_shaped = torch.einsum('bl,mkl->bmk', [shape, mano.shapedirs]) + v_template
    J = torch.einsum('bik,ji->bjk', [v_shaped, mano.J_regressor])

    num_joints = 16

    pose += mano.pose_mean

    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)

    # lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
    # v_posed = torch.matmul(lrotmin, smplx.posedirs).reshape(b, smplx.size[0], smplx.size[1]) + v_shaped  # smpl_v_posed

    J_transformed, A = batch_global_rigid_transformation(R, J, mano.parents)

    weights = weights.expand(b, -1, -1)    # [b, num_v, num_j]
    T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)  # [b, num_v, 4, 4]

    verts_homo = torch.cat([verts_tpose, torch.ones(b, verts_tpose.shape[1], 1, device=device)], dim=2)
    verts = torch.matmul(T, verts_homo.unsqueeze(-1))

    verts = verts[:, :, :3, 0]
    return verts

def lbs_tpose(pose, shape, weights, verts, hand_type='right'):
    mano = mano_layer[hand_type]
    b = pose.shape[0]
    device = pose.device
    dtype = shape.dtype
    v_template = mano.v_template
    v_shaped = torch.einsum('bl,mkl->bmk', [shape, mano.shapedirs]) + v_template
    J = torch.einsum('bik,ji->bjk', [v_shaped, mano.J_regressor])

    num_joints = 16

    pose += mano.pose_mean

    R = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, -1, 3, 3)

    # lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
    # v_posed = torch.matmul(lrotmin, smplx.posedirs).reshape(b, smplx.size[0], smplx.size[1]) + v_shaped  # smpl_v_posed

    J_transformed, A = batch_global_rigid_transformation(R, J, mano.parents)

    weights = weights.expand(b, -1, -1)  # [b, num_v, num_j]
    T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)  # [b, num_v, 4, 4]

    verts_homo = torch.cat([verts, torch.ones(b, verts.shape[1], 1, device=device)], dim=2)
    verts_tpose = torch.matmul(torch.linalg.inv(T), verts_homo.unsqueeze(-1))

    verts_tpose = verts_tpose[:, :, :3, 0]
    return verts_tpose

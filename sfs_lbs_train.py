import os
import pdb

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from glob import glob
from tqdm import tqdm
import argparse
import pickle
from pyhocon import ConfigFactory
import numpy as np
import cv2
import lpips
import datetime
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import nvdiffrast.torch as dr
from models.utils import get_normals, laplacian_smoothing, compute_color
from get_data import mano_layer, get_interhand_seqdatabyframe
from models.XHand import XHand
from repose import lbs, lbs_pose, pose2rot

MEAN_HAND_ALBEDO = torch.tensor([0.31996773, 0.36127372, 0.44126652]).cuda()

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def main(conf_path, data_path=None):
    conf = ConfigFactory.parse_file(conf_path)
    type = conf.get_string('data_type')
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        out_mesh_dire = out_path + '/' + conf.get_string('out_mesh_dire')
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire')
        os.makedirs(out_mesh_dire, exist_ok=True)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        adjust = conf.get_bool('adjust')
        out_dire = './%s_out/%s_%s' % (type, capture_name, data_name)
        os.makedirs('%s_out' % type, exist_ok=True)
        os.makedirs(out_dire, exist_ok=True)
        try:
            exp_name = conf.get_string('exp_name')
        except:
            exp_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        out_mesh_dire = out_dire + '/' + exp_name
        backup_dir = out_mesh_dire + '/backup'
        os.makedirs(out_mesh_dire, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        # backup key codes
        os.system("cp %s %s/ih_sfsseq.conf && cp sfs_lbs_train.py %s && cp get_data.py %s" % (
        conf_path, backup_dir, backup_dir, backup_dir))

        drop_cam = conf.get_string('drop_cam').split(',')
        cam_id = conf.get_string('cam_id')
        num_view = conf.get_int('num_view')
        num_frame = conf.get_int('num_frame')

    num = conf.get_int('num')
    w = conf.get_int('w')
    h = conf.get_int('h')
    net_type = conf.get_string('net_type')
    use_pe = conf.get_bool('use_pe')
    use_x_pos = conf.get_bool('use_x_pos')
    use_ray = conf.get_bool('use_ray')
    use_emb = conf.get_bool('use_emb')
    mlp_use_pose = conf.get_bool('mlp_use_pose')
    use_rotpose = conf.get_bool('use_rotpose')
    wo_latent = conf.get_bool('wo_latent')
    latent_num = conf.get_int('latent_num')
    resolution = (h, w)
    epoch_albedo = conf.get_int('epoch_albedo')
    epoch_sfs = conf.get_int('epoch_sfs')
    epoch_train = conf.get_int('epoch_train')
    sfs_weight = conf.get_float('sfs_weight')
    lap_weight = conf.get_float('lap_weight')
    albedo_weight = conf.get_float('albedo_weight')
    mask_weight = conf.get_float('mask_weight')
    edge_weight = conf.get_float('edge_weight')
    delta_weight = conf.get_float('delta_weight')
    part_smooth = conf.get_float('part_smooth')
    use_sum = conf.get_float('use_sum')
    degree = conf.get_int('degree')
    batch = conf.get_int('batch')
    lr = conf.get_float('lr')
    albedo_lr = conf.get_float('albedo_lr')
    sh_lr = conf.get_float('sh_lr')
    is_continue = conf.get_bool('is_continue')

    with open('mano/mano_weight_sub3.pkl', 'rb') as f:
        pkl = pickle.load(f)

    if type == 'interhand':
        imgs, grayimgs, masks, w2cs, projs, poses, shapes, transs, hand_types, rays = get_interhand_seqdatabyframe(
            data_path, res=(334, 512), data_name=data_name, capture_name=capture_name, drop_cam=drop_cam,
            split='test', return_ray=True, cam_id=cam_id, test_num=num_view, num_frame=num_frame, adjust=adjust)
        scales = torch.ones(transs.shape[0], 1)
        # imgs = grayimgs.unsqueeze(-1).repeat(1, 1, 1, 3)
        # tpose mano
        ori_v, ori_f, vertices, faces, weights, poses_A = [], [], [], [], [], []
        mean_shape = shapes.mean(0, keepdim=True).cpu()
        for i, hand_type in enumerate(hand_types):
            vertices_T = mano_layer[hand_type].v_template
            vertices_T = torch.einsum('bl,mkl->bmk', [mean_shape, mano_layer[hand_type].shapedirs]) + vertices_T
            faces_T = mano_layer[hand_type].faces
            ori_v.append(vertices_T[0].cuda())
            ori_f.append(torch.from_numpy(faces_T.astype(np.int32) + vertices_T.shape[1] * i).cuda())
            v, f = trimesh.remesh.subdivide_loop(vertices_T[0].numpy(), faces_T.astype(np.int64), iterations=3)
            f = f + i * v.shape[0]
            len_v = v.shape[0]
            len_f = f.shape[0]
            vertices.append(torch.from_numpy(v.astype(np.float32)).cuda())
            faces.append(torch.from_numpy(f.astype(np.int32)).cuda())

            weights.append(torch.from_numpy(pkl[hand_type]['weights']).float().cuda())

        ori_v = torch.cat(ori_v, 0)
        ori_f = torch.cat(ori_f, 0)
        vertices = torch.cat(vertices, 0)
        faces = torch.cat(faces, 0)
        weights = torch.cat(weights, 0)

    os.makedirs(join(out_mesh_dire, 'rerender'), exist_ok=True)

    mano_layer['right'] = mano_layer['right'].cuda()
    mano_layer['left'] = mano_layer['left'].cuda()
    sig = nn.Sigmoid()
    glctx = dr.RasterizeGLContext()

    num_frame = imgs.shape[0]
    num_view = imgs.shape[1]

    sh_coeffs = torch.zeros(num_view, 27).cuda()
    sh_coeffs.requires_grad_(True)

    batch = min(batch, imgs.shape[1])
    albedo = (torch.zeros_like(vertices)).unsqueeze(0)
    albedo.requires_grad_(True)

    delta = torch.zeros_like(vertices)
    delta.requires_grad_(False)
    poses.requires_grad_(True)
    weights.requires_grad_(False)
    vertices_tmp = torch.clone(vertices)
    a = vertices[faces[:, 0].long()]
    b = vertices[faces[:, 1].long()]
    c = vertices[faces[:, 2].long()]

    edge_length_mean = torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

    np_faces = faces.squeeze().detach().cpu().numpy()
    # init albedo/sh/delta
    if not is_continue:
        for i in range(1):
            vertices = vertices_tmp + delta
            # compute sphere harmonic coefficient as initialization
            optimizer = Adam([{'params': albedo, 'lr': albedo_lr},
                              {'params': sh_coeffs, 'lr': sh_lr}, {'params': poses, 'lr': 0.0005}])

            pbar = tqdm(range(epoch_albedo))
            rendered_img = []
            for i in pbar:
                perm = torch.randperm(num_view)
                for k in range(0, num_view, batch):
                    n = min(num_view, k + batch) - k
                    w2c = w2cs[0][perm[k:k + batch]].cuda()
                    proj = projs[0][perm[k:k + batch]].cuda()
                    img = imgs[0][perm[k:k + batch]].cuda()
                    mask = masks[0][perm[k:k + batch]].cuda()
                    sh_coeff = sh_coeffs[perm[k:k + batch]]
                    pose = poses[0:1].cuda()
                    shape = shapes[0:1].cuda()
                    trans = transs[0:1].cuda()
                    scale = scales[0:1].cuda()
                    vertices_n = vertices.unsqueeze(0)

                    # get posed verts
                    vertices_new = []
                    for idx, hand_type in enumerate(hand_types):
                        pose_new = pose[:, idx * 16:(idx + 1) * 16, :].view(1, -1)
                        shape_new = shape[:, idx * 10:(idx + 1) * 10].view(1, -1)
                        trans_new = trans[:, idx * 3:(idx + 1) * 3].view(1, -1)
                        weights_new = weights[idx * len_v:(idx + 1) * len_v]

                        verts_new = lbs_pose(pose_new.clone(), ori_v[idx * 778: (idx + 1) * 778].unsqueeze(0),
                                             weights_new,
                                             vertices_n[:, idx * len_v:(idx + 1) * len_v], hand_type=hand_type)
                        verts_new = verts_new * scale + trans_new.unsqueeze(1)
                        vertices_new.append(verts_new)
                    vertices_new = torch.cat(vertices_new, 1)

                    vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1,
                                                                                                                -1)
                    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
                    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
                    normals = get_normals(rot_verts[:, :, :3], faces.long())

                    rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
                    feat = torch.cat([normals, albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1])], dim=2)
                    feat, _ = dr.interpolate(feat, rast_out, faces)
                    pred_normals = feat[:, :, :, :3].contiguous()
                    rast_albedo = feat[:, :, :, 3:6].contiguous()
                    pred_mask = feat[:, :, :, 6:7].contiguous()
                    pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
                    pred_normals = F.normalize(pred_normals, p=2, dim=3)
                    rast_albedo = dr.antialias(rast_albedo, rast_out, proj_verts, faces)
                    pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)

                    valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
                    valid_normals = pred_normals[valid_idx]
                    valid_shcoeff = sh_coeff[valid_idx[0]]
                    valid_albedo = sig(rast_albedo[valid_idx])

                    valid_img = img[valid_idx]
                    pred_img = torch.clip(
                        compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)

                    sfs_loss = sfs_weight * F.l1_loss(pred_img, valid_img)
                    albedo_loss = F.mse_loss(MEAN_HAND_ALBEDO, valid_albedo.mean(0))
                    # albedo_loss = albedo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")
                    # lap_w = lap_weight * laplacian_smoothing(weights, faces.long(), method="uniform") / 10
                    mask_loss = mask_weight * F.mse_loss(pred_mask, mask)

                    loss = sfs_loss + mask_loss + albedo_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    des = 'sfs:%.4f' % sfs_loss.item() + ' mask:%.4f' % mask_loss.item()
                    pbar.set_description(des)

            delta.requires_grad_(True)
            sh_coeffs.requires_grad_(True)
            optimizer = Adam([{'params': delta, 'lr': lr},
                              {'params': sh_coeffs, 'lr': sh_lr}, {'params': poses, 'lr': 0.0005},
                              {'params': albedo, 'lr': albedo_lr}])

            pbar = tqdm(range(epoch_sfs))

            for i in pbar:
                perm = torch.randperm(num_view)
                if i == epoch_sfs // 2:
                    lap_weight = lap_weight * 10
                for k in range(0, num_view, batch):
                    vertices = vertices_tmp + delta
                    n = min(num_view, k + batch) - k
                    w2c = w2cs[0][perm[k:k + batch]].cuda()
                    proj = projs[0][perm[k:k + batch]].cuda()
                    img = imgs[0][perm[k:k + batch]].cuda()
                    mask = masks[0][perm[k:k + batch]].cuda()
                    # valid_mask = valid_masks[perm[k:k+batch]]
                    sh_coeff = sh_coeffs[perm[k:k + batch]]
                    pose = poses[0:1].cuda()
                    shape = shapes[0:1].cuda()
                    trans = transs[0:1].cuda()
                    scale = scales[0:1].cuda()
                    vertices_n = vertices.unsqueeze(0)

                    # get posed verts
                    vertices_new = []
                    for idx, hand_type in enumerate(hand_types):
                        pose_new = pose[:, idx * 16:(idx + 1) * 16, :].view(1, -1)
                        trans_new = trans[:, idx * 3:(idx + 1) * 3].view(1, -1)
                        weights_new = weights[idx * len_v:(idx + 1) * len_v]

                        verts_new = lbs_pose(pose_new.clone(), ori_v[idx * 778: (idx + 1) * 778].unsqueeze(0),
                                             weights_new,
                                             vertices_n[:, idx * len_v:(idx + 1) * len_v], hand_type=hand_type)
                        verts_new = verts_new * scale + trans_new.unsqueeze(1)
                        vertices_new.append(verts_new)
                    vertices_new = torch.cat(vertices_new, 1)

                    vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1,
                                                                                                                -1)
                    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
                    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
                    normals = get_normals(rot_verts[:, :, :3], faces.long())

                    rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
                    feat = torch.cat([normals, albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1])], dim=2)
                    feat, _ = dr.interpolate(feat, rast_out, faces)
                    pred_normals = feat[:, :, :, :3].contiguous()
                    rast_albedo = feat[:, :, :, 3:6].contiguous()
                    pred_mask = feat[:, :, :, 6:7].contiguous()
                    pred_normals = F.normalize(pred_normals, p=2, dim=3)
                    pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)

                    valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
                    valid_normals = pred_normals[valid_idx]
                    valid_shcoeff = sh_coeff[valid_idx[0]]
                    valid_albedo = sig(rast_albedo[valid_idx])

                    valid_img = img[valid_idx]
                    pred_img = torch.clip(
                        compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)

                    tmp_img = torch.zeros_like(img)
                    tmp_img[valid_idx] = pred_img
                    tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, faces)

                    sfs_loss = sfs_weight * (F.l1_loss(tmp_img[valid_idx], valid_img))

                    lap_delta_loss = lap_weight * laplacian_smoothing(delta, faces.long(), method="uniform",
                                                                      return_sum=False)
                    lap_delta_loss = lap_delta_loss[lap_delta_loss > torch.quantile(lap_delta_loss, 0.25)].sum()
                    lap_vert_loss = lap_weight * laplacian_smoothing(vertices, faces.long(), method="uniform",
                                                                     return_sum=False)
                    lap_vert_loss = lap_vert_loss[lap_vert_loss < torch.quantile(lap_vert_loss, 0.25)].sum()
                    albedo_loss = albedo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform",
                                                                      return_sum=False)
                    albedo_loss = albedo_loss[albedo_loss > torch.quantile(albedo_loss, 0.25)].sum() + F.mse_loss(
                        MEAN_HAND_ALBEDO, valid_albedo.mean(0)) * 100

                    lap_w = laplacian_smoothing(weights, faces.long(), method="uniform")

                    # normal_loss = 0.0 * normal_consistency(vertices, faces.long())
                    normal_loss = torch.zeros_like(albedo_loss)
                    mask_loss = mask_weight * F.mse_loss(pred_mask, mask)
                    a = vertices[faces[:, 0].long()]
                    b = vertices[faces[:, 1].long()]
                    c = vertices[faces[:, 2].long()]
                    edge_length = torch.cat(
                        [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

                    edge_loss = torch.clip(edge_length - edge_length_mean, 0, 1).mean() * edge_weight
                    delta_loss = (delta ** 2).sum(1).mean() * delta_weight

                    loss = sfs_loss + lap_delta_loss + lap_vert_loss + albedo_loss + mask_loss + normal_loss + delta_loss + edge_loss + lap_w

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    des = 'sfs:%.4f' % sfs_loss.item() + ' lap:%.4f' % lap_delta_loss.item() + ' albedo:%.4f' % albedo_loss.item() + \
                          ' mask:%.4f' % mask_loss.item() + ' normal:%.4f' % normal_loss.item() \
                          + ' edge:%.4f' % edge_loss.item() + ' delta:%.4f' % delta_loss.item() + ' weight:%.4f' % lap_w.item()
                    pbar.set_description(des)
                    if i == epoch_sfs - 1:
                        rendered_img.append(torch.cat([tmp_img, img], 2))
                        perm_last = perm.cpu().numpy()

            del optimizer

        torch.save({'sh_coeff': sh_coeffs, 'albedo': albedo, 'delta': delta}, join(out_mesh_dire, 'seq.pt'))

        np_verts = vertices.squeeze().detach().cpu().numpy()
        np_faces = faces.squeeze().detach().cpu().numpy()

        mesh = trimesh.Trimesh(np_verts, np_faces, process=False, maintain_order=True)
        mesh.export(join(out_mesh_dire, 'seq.obj'))

        save_obj_mesh_with_color(join(out_mesh_dire, 'seq_c.obj'), np_verts,
                                 np_faces, (sig(albedo).detach().cpu().numpy()[0])[:, 2::-1])

        rendered_img = torch.cat(rendered_img, 0)

        for i, idx in enumerate(perm_last):
            cv2.imwrite(join(out_mesh_dire, 'rerender', '%02d.png' % idx),
                        (rendered_img[i].detach().cpu().numpy() * 255).astype(np.int32))

    # start frame by frame
    albedo.requires_grad_(False)
    poses.requires_grad_(False)
    delta.requires_grad_(False)
    sh_coeffs.requires_grad_(False)
    delta = delta.clone().detach()

    xhand = XHand(vertices_tmp, faces, delta, albedo[0], weights, ori_v, sh_coeffs,
                  hand_type=hand_type, render_nettype=net_type, use_pe=use_pe, latent_num=latent_num,
                  use_x_pos=use_x_pos, use_ray=use_ray, use_emb=use_emb, wo_latent=wo_latent, mlp_use_pose=mlp_use_pose,
                  use_rotpose=use_rotpose, use_sum=use_sum).cuda()
    optimizer = Adam([{'params': xhand.parameters(), 'lr': 0.0005}])
    lpips_loss = lpips.LPIPS(net='vgg').cuda()

    if is_continue:
        if os.path.exists(join(out_mesh_dire, 'model.pth')):
            train_render = False
            state_dict = torch.load(join(out_mesh_dire, 'model.pth')).state_dict()
            print('continue from ' + join(out_mesh_dire, 'model.pth'))
            xhand.load_state_dict(state_dict, strict=False)
            sh_coeffs = xhand.sh_coeffs
            pbar = tqdm(range(epoch_train))
        else:
            train_render = True
            state_dict = torch.load('%s/pre_xhand.pth' % (out_dire)).state_dict()
            for key in list(state_dict.keys()):
                if 'renderer.' in key or 'render' in key:
                    del state_dict[key]
            print('continue from ' + '%s/pre_xhand.pth' % (out_dire))
            xhand.load_state_dict(state_dict, strict=False)
            sh_coeffs = xhand.sh_coeffs
            pbar = tqdm(range(epoch_train // 2, epoch_train))
    else:
        train_render = False
        pbar = tqdm(range(epoch_train))
    # scheduler = MultiStepLR(optimizer, milestones=[151], gamma=0.1)

    for i in pbar:
        if i % 200 == 0 and i > 0:
            torch.save(xhand, join(out_mesh_dire, 'model.pth'))
        if i >= epoch_train // 4:
            xhand.sh_coeffs.requires_grad_(True)
        if i >= epoch_train // 2:
            train_render = True
            if i % 50 == 0:
                weights_mean = []
                delta_mean = []
                albedo_mean = []

                print('upgrade mean')
                with torch.no_grad():
                    for j in range(num_frame):
                        pose = poses[j:j + 1].reshape(1, -1).cuda()
                        shape = shapes[j:j + 1].cuda()

                        if use_rotpose:
                            matrix = pose2rot(pose.view(-1, 48).clone()).view([1, -1, 3, 3])
                        else:
                            matrix = pose.clone()
                        condition = matrix.reshape(1, -1) / np.pi
                        pred_weights = xhand.forward_lbs(condition, False)[0][0]
                        pred_delta = xhand.forward_delta(condition)[0][0]
                        pred_albedo = xhand.forward_color(condition, False)[0][0]
                        weights_mean.append(pred_weights)
                        delta_mean.append(pred_delta)
                        albedo_mean.append(pred_albedo)

                xhand.renew_mean(torch.stack(delta_mean, 0).mean(0).detach(),
                                 torch.stack(albedo_mean, 0).mean(0).detach(),
                                 torch.stack(weights_mean, 0).mean(0).detach())
        perm_frame = torch.randperm(num_frame)
        for j_perm in range(num_frame):
            j = perm_frame[j_perm]
            rendered_img = []
            perm = torch.randperm(num_view)

            pose = poses[j:j + 1].reshape(1, -1).cuda()
            shape = shapes[j:j + 1].cuda()
            trans = transs[j:j + 1].cuda()
            scale = scales[j:j + 1].cuda()
            for k in range(0, num_view, batch):
                n = min(num_view, k + batch) - k
                w2c = w2cs[j][perm[k:k + n]].cuda()
                proj = projs[j][perm[k:k + n]].cuda()
                img = imgs[j][perm[k:k + n]].cuda()
                mask = masks[j][perm[k:k + n]].cuda()
                ray = rays[perm[k:k + n]].cuda()
                # valid_mask = valid_masks[perm[k:k+batch]]
                sh_coeff = xhand.sh_coeffs[perm[k:k + n]]

                data_input = pose, trans, scale, w2c, proj, mask, ray, sh_coeff
                valid_idx, render_imgs, tmp_img, pred_delta, vertices_new, pred_weights, pred_albedo, pred_mask = (
                    xhand(data_input, glctx, resolution, is_train=True, train_render=train_render))

                valid_img = img[valid_idx]

                sfs_loss = sfs_weight * ((F.l1_loss(tmp_img[valid_idx], valid_img)) * 0.8 + 0.2 * lpips_loss(
                    tmp_img.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2)).mean())  # ssim(tmp_img, img)))

                if train_render:
                    render_loss = ((F.l1_loss(render_imgs[valid_idx], valid_img)) * 0.8 + 0.2 * lpips_loss(
                        render_imgs.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2)).mean()) * sfs_weight
                else:
                    render_loss = torch.zeros_like(sfs_loss)

                if part_smooth:
                    lap_delta_loss = lap_weight * laplacian_smoothing(pred_delta[0], faces.long(), method="uniform",
                                                                      return_sum=False) / 10
                    # lap_vert_loss = lap_weight * laplacian_smoothing(vertices_new[0], faces.long(), method="uniform",
                    #                                                  return_sum=False) / 100
                    lap_delta_loss = lap_delta_loss[lap_delta_loss > torch.quantile(lap_delta_loss, 0.25)].sum()
                    lap_vert_loss = torch.zeros_like(lap_delta_loss)
                else:
                    lap_delta_loss = lap_weight * laplacian_smoothing(pred_delta[0], faces.long(), method="uniform",
                                                                      return_sum=True) / 10
                    lap_vert_loss = lap_weight * laplacian_smoothing(vertices_new[0], faces.long(), method="uniform",
                                                                     return_sum=True) / 100

                lap_w = laplacian_smoothing(pred_weights.squeeze(0), faces.long(), method="uniform")

                albedo_loss = laplacian_smoothing(pred_albedo.squeeze(0), faces.long(), method="uniform",
                                                  return_sum=False)
                albedo_loss = albedo_loss[albedo_loss > torch.quantile(albedo_loss, 0.25)].sum()
                normal_loss = torch.zeros_like(albedo_loss)
                mask_loss = mask_weight * F.mse_loss(pred_mask, mask)
                weight_loss = 100 * F.mse_loss(pred_weights.squeeze(0), weights)
                a = vertices_new[0, faces[:, 0].long()]
                b = vertices_new[0, faces[:, 1].long()]
                c = vertices_new[0, faces[:, 2].long()]
                edge_length = torch.cat(
                    [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

                edge_loss = torch.clip(edge_length - edge_length_mean, 0, 1).mean() * edge_weight
                delta_loss = F.relu((pred_delta[0] ** 2).sum(1).mean() - 0.0001) * delta_weight
                delta2_loss = F.l1_loss(pred_delta[0], delta)

                loss = sfs_loss + lap_delta_loss + lap_vert_loss + mask_loss + normal_loss + delta_loss + delta2_loss + edge_loss + lap_w + weight_loss + albedo_loss + render_loss  # + albedo_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                des = 's:%.3f' % sfs_loss.item() + ' l:%.3f' % lap_delta_loss.item() + ' a:%.3f' % albedo_loss.item() + \
                      ' m:%.3f' % mask_loss.item() + ' n:%.3f' % normal_loss.item() \
                      + ' e:%.3f' % edge_loss.item() + ' d:%.3f,%.3f' % (
                      delta_loss.item(), delta2_loss.item()) + ' w:%.3f,%.3f' % (weight_loss.item(), lap_w.item()) \
                      + ' r:%.3f' % (render_loss.item())
                pbar.set_description(des)
                if i % 100 == 0:
                    rendered_img.append(torch.cat([tmp_img, render_imgs, img], 2))
                    perm_last = perm.cpu().numpy()
            if i % 100 == 0:
                rendered_img = torch.cat(rendered_img, 0)
                os.makedirs(join(out_mesh_dire, 'result_%d' % i), exist_ok=True)
                for i_, idx in enumerate(perm_last):
                    cv2.imwrite(join(out_mesh_dire, 'result_%d' % i, 'train_%02d_%02d.png' % (j.item(), idx)),
                                (rendered_img[i_].detach().cpu().numpy() * 255).astype(np.int32))
                save_obj_mesh_with_color(join(out_mesh_dire, 'result_%d' % i, 'seq_%d.obj' % (j.item())),
                                         vertices_new[0].detach().cpu().numpy(),
                                         np_faces, (pred_albedo.detach().cpu().numpy()[0])[:, 2::-1])

    # save parameter
    torch.save(xhand, join(out_mesh_dire, 'model.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/demo_sfs.conf')

    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    main(args.conf, args.data_path)

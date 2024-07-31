import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from repose import lbs, lbs_pose, pose2rot
from models.PostionalEncoding import PostionalEncoding
from models.mlp import ImplicitNetwork, ConditionNetwork, MLP, SimpleNetwork, MLP_res
from models.unet import UNet
from models.utils import get_normals, compute_color

class XHand(nn.Module):
    def __init__(self, vertices, faces, init_delta, init_albedo, init_weights, template_v, sh_coeffs,
                 hand_type='left', render_nettype='mlp', use_pe=True, use_x_pos=True, use_ray=True, use_emb=True,
                 wo_latent=False, latent_num=20, use_rotpose=False, mlp_use_pose=True, use_sum=False, res=[512, 334]):
        '''

        :param init_delta: n_vertices, 3
        :param init_albedo: n_vertices, 3
        :param init_weights: n_vertices, num_joints
        '''
        super(XHand, self).__init__()
        init_weights = torch.log(1000 * (init_weights + 0.000001))
        self.template_v = template_v
        self.hand_type = hand_type
        self.verts = vertices
        self.faces = faces
        # self.glctx = dr.RasterizeGLContext()
        self.use_rotpose = use_rotpose
        if self.use_rotpose:
            pose_num = 16 * 3 * 3
        else:
            pose_num = 16 * 3
        self.use_emb = use_emb
        if self.use_emb:
            self.delta_net = ConditionNetwork(init_delta, pose_num, 3, 10, 512, 8, learnable_mean=False)
            self.color_net = ConditionNetwork(init_albedo, pose_num, 3, 10, 128, 5, learnable_mean=False)
            self.lbs_net = ConditionNetwork(init_weights, pose_num, init_weights.shape[1], 10, 128, 5)
        else:
            self.delta_net = SimpleNetwork(pose_num, 3, 512, 8)
            self.color_net = SimpleNetwork(pose_num, 3, 128, 5)
            self.lbs_net = SimpleNetwork(pose_num, init_weights.shape[1], 128, 5)
        self.render_nettype = render_nettype

        self.use_pe = use_pe
        self.use_x_pos = use_x_pos
        self.use_ray = use_ray
        self.wo_latent = wo_latent
        self.renderer_dim = 9 + latent_num

        if not wo_latent:
            self.renderer_dim += 20
        if self.use_x_pos:
            if self.use_pe:
                self.renderer_dim += 87
            else:
                self.renderer_dim += 3

        if self.use_ray:
            if self.use_pe:
                self.renderer_dim += 87
            else:
                self.renderer_dim += 3

        if render_nettype == 'mlp':
            self.mlp_use_pose = mlp_use_pose
            if mlp_use_pose:
                self.renderer = MLP_res(self.renderer_dim + 48, 3)
            else:
                self.renderer = MLP(self.renderer_dim , 3)
        elif render_nettype == 'unet':
            self.renderer = UNet(self.renderer_dim, 3, 2, 0)
        self.render_code = nn.Parameter(torch.zeros(init_delta.shape[0], latent_num))

        if self.use_pe:
            self.pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
        else:
            self.pe = torch.nn.Sequential()
        self.sh_coeffs = nn.Parameter(sh_coeffs, requires_grad=True)

        self.sig = nn.Sigmoid()
        self.use_sum = use_sum
    def forward_delta(self, condition):
        return self.delta_net(condition)

    def forward_color(self, condition, if_sig=True):
        if if_sig:
            pred_albedo, code = self.color_net(condition)
            return self.sig(pred_albedo), code
        else:
            return self.color_net(condition)

    def forward_lbs(self, condition, if_softmax=True):
        pred_weights, code = self.lbs_net(condition)
        if if_softmax:
            pred_weights = F.softmax(pred_weights, -1)
        return pred_weights, code

    def forward_render(self, input_f, pose=None):
        '''
        :param input_f: b, n, d / b, d, h, w
        :return: b, n, 3 / b, 3, h, w
        '''
        if self.render_nettype == 'sr':
            rgb_lr = input_f[:, :3]  # torch.Size([3, H, W])
            feature_map = input_f  # torch.Size([1, N_channel, H, W)])

            synthesis_kwargs = {}

            if pose is None:
                # conditioning_signal = torch.zeros(1, 14, 512).to(device)        #in EG3D, it is torch.Size([batch_size, 14, 512])
                # print("No conditioning signal is provided to the SR module. Is it intended?")
                # torch.manual_seed(42)
                # conditioning_signal = torch.rand(1, 1, 45).to(device)
                if self.use_rotpose:
                    pose = torch.ones(1, 1, 15*3*3).to(device)
                else:
                    pose = torch.ones(1, 1, 45).to(device)
            # Run superresolution to get final image
            render_pix = self.renderer(rgb_lr.clone(), feature_map.clone(), ws=pose.repeat(rgb_lr.shape[0], 1, 1), noise_mode='none')  # torch.Size([1, 3, 128, 128])
        else:
            render_pix = self.renderer(input_f)
        return render_pix

    def renew_mean(self, delta_mean, albedo_mean, weights_mean):
        # weights_mean = torch.log(1000 * (weights_mean + 0.000001))
        if self.use_emb:
            self.lbs_net.x_mean = nn.Parameter(weights_mean.detach(), requires_grad=False)
            self.delta_net.x_mean = nn.Parameter(delta_mean.detach(), requires_grad=False)
            self.color_net.x_mean = nn.Parameter(albedo_mean.detach(), requires_grad=False)

    def forward(self, data_input, glctx, resolution, is_train=True, train_render=True, is_right=None):
        pose, trans, scale, w2c, proj, mask, ray, sh_coeff = data_input

        n = w2c.shape[0]
        pose_n = pose.shape[0]
        if self.use_rotpose:
            matrix = pose2rot(pose.view(-1, 48).clone()).view([pose_n, -1, 3, 3])
        else:
            matrix = pose.clone()
        condition = matrix.reshape(pose_n, -1) / np.pi


        pred_delta, code_delta = self.forward_delta(condition)
        vertices_n = self.verts.unsqueeze(0) + pred_delta
        pred_weights, _ = self.forward_lbs(condition)
        pred_albedo, code_albedo = self.forward_color(condition)
        verts_new = lbs_pose(pose, self.template_v.unsqueeze(0),
                             pred_weights,
                             vertices_n, hand_type=self.hand_type)
        if is_right is not None:
            verts_new[:, :, 0] = (2 * is_right - 1) * verts_new[:, :, 0]
        vertices_new = verts_new * scale + trans.unsqueeze(1)
        # get posed verts
        vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1, -1)
        rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
        normals = get_normals(vertsw[:, :, :3], self.faces.long())

        rast_out, _ = dr.rasterize(glctx, proj_verts, self.faces, resolution=resolution)
        if self.wo_latent:
            latent_code = self.render_code.unsqueeze(0).detach()
        else:
            latent_code = torch.cat([code_delta.squeeze(0), code_albedo.squeeze(0),
                                 self.render_code], -1).unsqueeze(0).detach()
        if is_train:
            feat = torch.cat(
                [normals, pred_albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1]),
                 vertices_new.expand(n, -1, -1),
                 latent_code.expand(n, -1, -1)], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, self.faces)
            pred_normals = feat[:, :, :, :3].contiguous()
            rast_albedo = feat[:, :, :, 3:6].contiguous()
            pred_mask = feat[:, :, :, 6:7].contiguous()
            pred_vert = feat[:, :, :, 7:10].contiguous()
            pred_code = feat[:, :, :, 10:].contiguous()
        else:
            feat = torch.cat(
                [normals, pred_albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1]),
                 vertices_new.expand(n, -1, -1), get_normals(rot_verts[:, :, :3], self.faces.long()),
                 latent_code.expand(n, -1, -1)], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, self.faces)
            pred_normals = feat[:, :, :, :3].contiguous()
            rast_albedo = feat[:, :, :, 3:6].contiguous()
            pred_mask = feat[:, :, :, 6:7].contiguous()
            pred_vert = feat[:, :, :, 7:10].contiguous()
            gt_normals = feat[:, :, :, 10:13].contiguous()
            pred_code = feat[:, :, :, 13:].contiguous()
        pred_normals = F.normalize(pred_normals, p=2, dim=3)
        pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, self.faces).squeeze(-1)
        if mask is not None:
            valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
        else:
            valid_idx = torch.where( rast_out[:, :, :, 3] > 0)
        valid_normals = pred_normals[valid_idx]
        if is_right is not None:
            valid_normals = valid_normals * (2 * is_right - 1)

        valid_shcoeff = sh_coeff[valid_idx[0]]
        valid_albedo = rast_albedo[valid_idx]

        pred_img = torch.clip(
            compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)

        tmp_img = torch.zeros_like(pred_normals)
        tmp_img[valid_idx] = pred_img
        tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, self.faces)
        render_imgs = torch.zeros_like(pred_normals)

        if train_render:
            input_f = [tmp_img, pred_normals, rast_albedo, pred_code]
            if self.use_ray:
                input_f = input_f + [self.pe(ray)]
            if self.use_x_pos:
                input_f = [self.pe(pred_vert)] + input_f
            input_f = torch.cat(input_f, 3).detach()

            if self.render_nettype == 'mlp':
                input_f = input_f[valid_idx]
                if self.mlp_use_pose:
                    render_pix = self.forward_render(torch.cat([input_f, pose.repeat(input_f.shape[0], 1)], 1))
                else:
                    render_pix = self.forward_render(input_f)
                render_imgs = torch.zeros_like(pred_normals)
                render_imgs[valid_idx] = render_pix
            elif self.render_nettype == 'unet':
                input_f = input_f.permute(0, 3, 1, 2)
                input_f = torch.cat([torch.zeros_like(input_f[:, :, :, :1]), input_f, torch.zeros_like(input_f[:, :, :, :1])], 3)
                render_imgs = self.forward_render(input_f)[:, :, :, 1:-1].permute(0, 2, 3, 1)
            elif self.render_nettype == 'sr':
                input_f = input_f.permute(0, 3, 1, 2)
                if self.use_rotpose:
                    render_imgs = self.forward_render(input_f, pose=condition[:, 9:].unsqueeze(1)).permute(0, 2, 3, 1)
                else:
                    render_imgs = self.forward_render(input_f, pose=condition[:, 3:].unsqueeze(1)).permute(0, 2, 3, 1)

        if self.use_sum:
            render_imgs = render_imgs + tmp_img
        if is_train:
            return valid_idx, render_imgs, tmp_img, pred_delta, vertices_new, pred_weights, pred_albedo, pred_mask
        else:
            gt_normals = dr.antialias(gt_normals, rast_out, proj_verts, self.faces)
            gt_normals = F.normalize(gt_normals, p=2, dim=3)
            gt_normals = gt_normals[valid_idx]
            if is_right is not None:
                gt_normals = gt_normals * (2 * is_right - 1)
            light_direction = torch.zeros_like(gt_normals)
            light_direction[:, 2] = -1
            reflect = (-light_direction) - 2 * gt_normals * torch.sum(gt_normals * (-light_direction),
                                                                      dim=1, keepdim=True)
            dot = torch.sum(reflect * light_direction, dim=1, keepdim=True)  # n 1
            specular = 0.2 * torch.pow(torch.maximum(dot, torch.zeros_like(dot)), 16)
            color = torch.sum(gt_normals * light_direction, dim=1, keepdim=True) + specular
            color = torch.clamp(color, 0, 1)
            # color = color.squeeze().detach().cpu().numpy()
            mesh_img = torch.zeros_like(pred_normals)
            mesh_img[valid_idx] = color
            mesh_img = torch.cat([mesh_img, rast_albedo], 2)
            # pred_color = compute_color(pred_albedo, get_normals(rot_verts[:, :, :3], self.faces.long()), sh_coeff)
            return render_imgs, mesh_img, tmp_img, vertices_new, pred_mask, pred_albedo

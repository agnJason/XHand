import os
import pdb
import time
import math
import imageio
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from glob import glob
from tqdm import tqdm
import argparse
import pickle
from pyhocon import ConfigFactory
import numpy as np
import cv2
from PIL import Image
import trimesh
import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

from get_data import mano_layer, get_interhand_test_seqdatabyframe, get_interhand_seqdatabyframe
from models.XHand import XHand

loss_fn_alex = lpips.LPIPS(net='alex', version=0.1).cuda()

cam_names = ['cam400262', 'cam400263', 'cam400264', 'cam400265', 'cam400266', 'cam400267', 'cam400268', 'cam400269',
            'cam400270', 'cam400271', 'cam400272', 'cam400273', 'cam400274', 'cam400275', 'cam400276', 'cam400279',
            'cam400280', 'cam400281', 'cam400282', 'cam400283', 'cam400284', 'cam400285', 'cam400287', 'cam400288',
            'cam400289', 'cam400290', 'cam400291', 'cam400292', 'cam400293', 'cam400294', 'cam400296', 'cam400297',
            'cam400298', 'cam400299', 'cam400300', 'cam400301', 'cam400310', 'cam400312', 'cam400314', 'cam400315',
            'cam400316', 'cam400317', 'cam400319', 'cam400320', 'cam400321', 'cam400322', 'cam400323', 'cam400324',
            'cam400326', 'cam400327']

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [8， 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)*(img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 9:
        return float('inf')
    return 28 * math.log10(1.0 / math.sqrt(mse))

def convert2video(filepath, images):
    fps = 5  # 每秒钟30帧
    with imageio.get_writer(filepath, fps=fps) as video:
        for image in images:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.append_data(frame)

def test(exp_path, data_path=None, cam_id=None, split='test', test_data_name=None, test_capture_name=None, save_vis=False, save_mesh=False):
    conf = exp_path + '/backup/ih_sfsseq.conf'

    conf = ConfigFactory.parse_file(conf)
    type = conf.get_string('data_type')
    checkpoint_path = exp_path + '/model.pth'

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
        # cam_id = conf.get_string('cam_id')
        out_dire = '/'.join(checkpoint_path.split('/')[:-1])
        out_mesh_dire = out_dire + '/outs'
        os.makedirs(out_mesh_dire, exist_ok=True)
        drop_cam = conf.get_string('drop_cam').split(',')
    w = conf.get_int('w')
    h = conf.get_int('h')
    resolution = (h, w)
    use_pe = conf.get_bool('use_pe')
    use_x_pos = conf.get_bool('use_x_pos')
    use_ray = conf.get_bool('use_ray')
    use_emb = conf.get_bool('use_emb')
    wo_latent = conf.get_bool('wo_latent')
    net_type = conf.get_string('net_type')
    latent_num = conf.get_int('latent_num')
    mlp_use_pose = conf.get_bool('mlp_use_pose')
    use_rotpose = conf.get_bool('use_rotpose')

    xhand_pth = torch.load(checkpoint_path)
    xhand = XHand(xhand_pth.verts, xhand_pth.faces, xhand_pth.delta_net.x_mean,
           xhand_pth.color_net.x_mean, xhand_pth.lbs_net.x_mean,
           xhand_pth.template_v, xhand_pth.sh_coeffs, latent_num=latent_num,
           hand_type=xhand_pth.hand_type, render_nettype=net_type, use_pe=use_pe,
           use_x_pos=use_x_pos, use_ray=use_ray, use_emb=use_emb, wo_latent=wo_latent,
                            mlp_use_pose=mlp_use_pose, use_rotpose=use_rotpose)


    xhand.load_state_dict(xhand_pth.state_dict(), strict=False)
    xhand = xhand.cuda()

    xhand.eval()

    glctx = dr.RasterizeGLContext()

    if type == 'interhand':
        if test_data_name is not None:
            data_name = test_data_name
        if test_capture_name is not None:
            capture_name = test_capture_name
        imgs, grayimgs, masks, w2cs, projs, poses, shapes, transs, hand_types, rays, img_names = get_interhand_test_seqdatabyframe(
                data_path, res=(334, 512), data_name=data_name, capture_name=capture_name, drop_cam=drop_cam, cam_id=cam_id,
                split=split, return_ray=True, adjust=adjust)  # , cam_id='cam400267')
        if cam_id in cam_names:
            sh_id = cam_names.index(cam_id)
        else:
            sh_id = 0
        scales = torch.ones(transs.shape[0], 1)
    sh_coeffs = xhand.sh_coeffs
    num_frame = imgs.shape[0]
    num_view = imgs.shape[1]

    infer_speed = []
    output_imgs = []
    total_psnr = []
    total_ssim = []
    total_lpips = []
    with torch.no_grad():
        for j in range(num_frame):

            pose = poses[j:j + 1].reshape(1, -1).cuda()
            shape = shapes[j:j + 1].cuda()
            trans = transs[j:j + 1].cuda()
            scale = scales[j:j + 1].cuda()
            for k in range(0, num_view):
                w2c = w2cs[j][k:k + 1].cuda()
                proj = projs[j][k:k + 1].cuda()
                img = imgs[j][k:k + 1].cuda()
                ray = rays[k:k + 1].cuda()
                sh_coeff = sh_coeffs[sh_id:sh_id + 1]

                data_input = pose, trans, scale, w2c, proj, None, ray, sh_coeff

                start_time = time.time()
                render_imgs, mesh_imgs, pred_imgs, vertices_new, pred_mask, pred_albedo = (
                    xhand(data_input, glctx, resolution, is_train=False))
                # render_imgs = pred_imgs
                if j > 0:
                    infer_speed.append(time.time() - start_time)
                pred_mask = pred_mask > 0.5

                render_imgs[pred_mask==0] = 0
                img[pred_mask == 0] = 0
                outs = (torch.cat([img, mesh_imgs, pred_imgs, render_imgs], 2)[0].cpu().numpy() * 255).astype(np.uint8)
                width = outs.shape[1]
                img_num = width / img.shape[2]

                outmask = (torch.cat([pred_mask for i in range(int(img_num))], 2)[0].unsqueeze(-1).cpu().numpy() * 255).astype(np.uint8)

                outs = np.concatenate([outs, outmask], 2)
                output_imgs.append(outs)

                psnr = calculate_psnr(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy(), pred_mask[0].detach().cpu().numpy())  # PSNR(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy())
                ssim = SSIM(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy(), channel_axis=2, data_range=1, multichannel=True)
                lpips_loss = loss_fn_alex(render_imgs.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2))

                total_psnr.append(psnr)
                total_ssim.append(ssim)
                total_lpips.append(lpips_loss.item())
                if save_vis:
                    cv2.imwrite(out_mesh_dire + '/%s_%s.png' % (cam_id, img_names[j][:-4]), outs)
                if save_mesh:
                    save_obj_mesh_with_color(out_mesh_dire + '/%s.obj' % (img_names[j][:-4]),
                                             vertices_new.cpu().numpy()[0],
                                             xhand.faces.cpu().numpy(),
                                             (pred_albedo.cpu().numpy()[0])[:, 2::-1])

    print('inference fps:', 1 / np.mean(infer_speed))
    print('PSNR:', np.mean(total_psnr), "SSIM:", np.mean(total_ssim), "LPIPS:", np.mean(total_lpips))
    if save_vis:
        convert2video(out_dire + '/%s_%s_%s.mp4' % (capture_name, data_name, cam_id), output_imgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='confs/demo_sfs.conf')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--cam_id', type=str, default='cam400262')
    parser.add_argument('--test_data_name', type=str, default=None)
    parser.add_argument('--test_capture_name', type=str, default=None)
    args = parser.parse_args()
    test(args.exp_path, args.data_path, cam_id=args.cam_id, save_vis=args.save_vis, save_mesh=args.save_mesh, split=args.split,
         test_data_name=args.test_data_name, test_capture_name=args.test_capture_name)
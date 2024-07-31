
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    '''
    Rs: B 24 3 3
    Js: B 24 3
    '''
    N = Rs.shape[0]
    num_joints = Rs.shape[1]
    device = Rs.device
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().to(device)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, num_joints, 1, 1).to(device)], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

vertex_ids = {
'nose': 9120,
'reye': 9929,
'leye':	9448,
'rear':	616,
'lear':	6,
'rthumb': 8079,
'rindex': 7669,
'rmiddle': 7794,
'rring': 7905,
'rpinky': 8022,
'lthumb': 5361,
'lindex': 4933,
'lmiddle': 5058,
'lring': 5169,
'lpinky': 5286,
'LBigToe': 5770,
'LSmallToe': 5780,
'LHeel': 8846,
'RBigToe': 8463,
'RSmallToe': 8474,
'RHeel': 8635}


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain

def find_dynamic_lmk_idx_and_bcoords(
    vertices,
    pose,
    dynamic_lmk_faces_idx,
    dynamic_lmk_b_coords,
    neck_kin_chain,
    pose2rot=True,
):
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks
        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.
        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional
        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    dtype = vertices.dtype
    batch_size = vertices.shape[0]

    if pose2rot:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(
            pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
            batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class SMPLX(nn.Module):
    def __init__(self, model_path, is_smplx=True, use_pca=True, num_pca_comps=45, hand_mean=True, use_face_contour=True):
        super(SMPLX, self).__init__()
        with open(model_path, 'rb') as f:
            smpl_model = pickle.load(f, encoding='latin1')

        if is_smplx:
            self.register_buffer('J_regressor', torch.FloatTensor(smpl_model['J_regressor'])) # 55 10475
        else:
            if isinstance(smpl_model['J_regressor'], np.ndarray):
                J_regressor = smpl_model['J_regressor']
            else:
                J_regressor = smpl_model['J_regressor'].toarray()
            self.register_buffer('J_regressor', torch.FloatTensor(J_regressor))
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights'])) # 10475 55

        posedirs = np.array(smpl_model['posedirs'])
        num_pose = posedirs.shape[-1]
        if is_smplx:
            assert num_pose == 486
        else:
            assert num_pose == 207 or num_pose == 459

        posedirs = np.reshape(posedirs, [-1, num_pose]).T
        self.register_buffer('posedirs', torch.FloatTensor(posedirs))

        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        shapedirs = np.array(smpl_model['shapedirs'])
        num_beta = shapedirs.shape[-1]
        # 300 betas  100 expression
        shapedirs = np.reshape(shapedirs, [-1, num_beta]).T

        self.register_buffer('shapedirs', torch.FloatTensor(shapedirs))

        self.faces = smpl_model['f'].astype(np.int32)
        self.parents = np.array(smpl_model['kintree_table'])[0].astype(np.int32)
        self.register_buffer('e3', torch.eye(3).float())
        self.use_pca = use_pca
        self.is_smplx = is_smplx
        self.num_pca_comps = num_pca_comps
        self.use_face_contour = use_face_contour
        self.size = smpl_model['v_template'].shape

        if is_smplx:
            if use_pca:
                left_hand_components = smpl_model['hands_componentsl'][:num_pca_comps]
                right_hand_components = smpl_model['hands_componentsr'][:num_pca_comps]
                self.register_buffer('left_hand_components', torch.from_numpy(left_hand_components))
                self.register_buffer('right_hand_components', torch.from_numpy(right_hand_components))

            if hand_mean:
                left_hand_mean = torch.from_numpy(smpl_model['hands_meanl'])
                right_hand_mean = torch.from_numpy(smpl_model['hands_meanr'])
            else:
                left_hand_mean = torch.from_numpy(np.zeros_like(smpl_model['hands_meanl']))
                right_hand_mean = torch.from_numpy(np.zeros_like(smpl_model['hands_meanr']))

            self.register_buffer('left_hand_mean', left_hand_mean)
            self.register_buffer('right_hand_mean', right_hand_mean)
                
            # create mean pose
            global_orient_mean = torch.zeros(3)
            body_pose_mean = torch.zeros(21 * 3) # 23 - 2
            jaw_pose_mean = torch.zeros(3)
            leye_pose_mean = torch.zeros(3)
            reye_pose_mean = torch.zeros(3)
            # :3 global 3:66 body 66:69 jaw 69:72 leye 72:75 reye 75:120 lhand 120:165 rhand
            pose_mean = torch.cat([global_orient_mean, body_pose_mean, jaw_pose_mean, leye_pose_mean, reye_pose_mean, left_hand_mean, right_hand_mean])

            self.register_buffer('pose_mean', pose_mean)

            face_keyp_idxs = np.array([vertex_ids['nose'],vertex_ids['reye'],vertex_ids['leye'],vertex_ids['rear'],vertex_ids['lear']], dtype=np.int64)
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],vertex_ids['LSmallToe'],vertex_ids['LHeel'],vertex_ids['RBigToe'],vertex_ids['RSmallToe'],vertex_ids['RHeel']], dtype=np.int32)
            tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate([face_keyp_idxs, feet_keyp_idxs, tips_idxs])
            self.register_buffer('extra_joints_idxs', torch.from_numpy(extra_joints_idxs).long())

            self.register_buffer('lmk_faces_idx', torch.from_numpy(smpl_model['lmk_faces_idx']).long())
            self.register_buffer('lmk_bary_coords', torch.from_numpy(smpl_model['lmk_bary_coords']).float())

            if use_face_contour:
                self.register_buffer('dynamic_lmk_faces_idx', torch.from_numpy(smpl_model['dynamic_lmk_faces_idx']).long())
                self.register_buffer('dynamic_lmk_bary_coords', torch.from_numpy(np.array(smpl_model['dynamic_lmk_bary_coords'], dtype=np.float32)))

                neck_kin_chain  = find_joint_kin_chain(12, self.parents)
                self.register_buffer('neck_kin_chain', torch.from_numpy(np.array(neck_kin_chain)).long())

    def forward(self, pose, shape, delta=None, trans=None, scale=1.0, edge_unique=None, delta2=None):
        b = pose.shape[0]
        device = pose.device
        v_template = self.v_template
        if edge_unique is None:
            do_sub = False
        else:
            do_sub = True
        v_shaped = torch.matmul(shape, self.shapedirs).reshape(-1, self.size[0], self.size[1]) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        if self.is_smplx:
            num_joints = 55
        else:
            # num_joints = 24
            num_joints = self.posedirs.shape[0] // 9 + 1
            
        if pose.ndimension() == 2:
            if self.is_smplx and self.use_pca:
                part_pose = pose[:, :-2*self.num_pca_comps]
                left_hand_pose = pose[:, -2*self.num_pca_comps:-self.num_pca_comps]
                right_hand_pose = pose[:, -self.num_pca_comps:]

                left_hand_pose = torch.einsum('bi,ij->bj', [left_hand_pose, self.left_hand_components])
                right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, self.right_hand_components])

                pose = torch.cat([part_pose, left_hand_pose, right_hand_pose], dim=1)
                pose += self.pose_mean

            R = batch_rodrigues(pose.reshape(-1,3)).reshape(b, -1, 3, 3)
        elif pose.ndimension() == 4:
            R = pose
        else:
            raise NotImplementedError

        lrotmin = (R[:, 1:, :] - self.e3).reshape(b, -1)
        v_posed = torch.matmul(lrotmin, self.posedirs).reshape(b, self.size[0], self.size[1]) + v_shaped
        if delta is not None:
            v_posed = v_posed + delta

        J_transformed, A = batch_global_rigid_transformation(R, J, self.parents)
        weights = self.weights.expand(b, -1, -1)
        T = torch.matmul(weights, A.reshape(b, num_joints, 16)).reshape(b, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(b, v_posed.shape[1], 1,device=device)], dim=2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if do_sub:
            new_verts = verts[:,edge_unique].mean(dim=2)
            verts = torch.cat([verts,new_verts], dim=1)
            if delta2 is not None:
                verts = verts + delta2

        if trans is not None:
            if trans.ndimension() == 2:
                trans = trans.unsqueeze(1) # b 3 -> b 1 3
                
            verts = verts * scale + trans
            J_transformed = J_transformed + trans

        if self.is_smplx:
            extra_joints = torch.index_select(verts, 1, self.extra_joints_idxs)
            joints = torch.cat([J_transformed, extra_joints], dim=1)


            lmk_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(b, -1)
            lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).expand(b, -1, -1)
            if self.use_face_contour:
                dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(verts, R, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords, self.neck_kin_chain, False)
                lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
                lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
 
            # face landmark
            faces = torch.from_numpy(self.faces).long().to(device)
            lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(b, -1, 3)
            lmk_faces += torch.arange(b, dtype=torch.long, device=device).view(-1,1,1) * verts.shape[1]
            lmk_vertices = verts.view(-1,3)[lmk_faces].view(b,-1,3,3)
            landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])

            joints = torch.cat([joints, landmarks], dim=1)

        else:
            joints = J_transformed

        return verts, joints
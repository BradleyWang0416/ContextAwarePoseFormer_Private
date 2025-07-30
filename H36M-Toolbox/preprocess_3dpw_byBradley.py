import sys
sys.path.append("/home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/")
import os
import pickle
import torch
import numpy as np
np.set_printoptions(suppress=True)
import random
random.seed(42)
np.random.seed(42)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm
import joblib
import glob
import cv2

from smplx import SMPL

from common import transforms
from viz_skel_seq import viz_skel_seq_anim

THREEDPW_PTH = "/data2/wxs/DATASETS/PW3D/sequenceFiles/"
THREEDPW_IMAGE = "/data2/wxs/DATASETS/PW3D/imageFiles/"
J_reg = np.load("/home/wxs/MotionBERT-main/data/AMASS/J_regressor_h36m_correct.npy")  # (17,6890)


ALL_DATA = {
    'train': {
        'joint_2d': [],
        'joint3d_image': [],
        'joint3d_cam': [],
        'joint3d_world': [],
        # 'joints_2.5d_image': [],
        '2.5d_factor': [],
        'source': [],
        'res': [],
        'campose_valid': [],
        },
    'test': {
        'joint_2d': [],
        'joint3d_image': [],
        'joint3d_cam': [],
        'joint3d_world': [],
        # 'joints_2.5d_image': [],
        '2.5d_factor': [],
        'source': [],
        'res': [],
        'campose_valid': [],
        },
}

ALL_SMPL_DATA = {
    'train': {
        'poses': [],
        'trans': [],
        'betas': [],
        'gender': [],
        'source': [],
        },
    'test': {
        'poses': [],
        'trans': [],
        'betas': [],
        'gender': [],
        'source': [],
        },
}

def main():
    for split_name in ['test', 'train']:
        pw3d_seq_names = glob.glob(os.path.join(THREEDPW_PTH, split_name + '/*'))
        for pw3d_seq_name in tqdm(pw3d_seq_names):
            data = pickle.load(open(pw3d_seq_name, 'rb'), encoding='latin1')
            # dict_keys(['img_frame_ids', 'sequence',
            #            'poses', 'poses_60Hz', 'poses2d', 'jointPositions',
            #            'trans', 'trans_60Hz',
            #            'cam_intrinsics', 'cam_poses', 'campose_valid', 
            #            'betas', 'betas_clothed', 'v_template_clothed', 
            #            'genders', 'texture_maps'])
            """
            The 3DPW dataset contains several motion sequences, which are organized into two folders: imageFiles and sequenceFiles.
            The folder imageFiles contains the RGB-images for every sequence. 
            The folder sequenceFiles provides synchronized motion data and SMPL model parameters in the form of .pkl-files. For each sequence, the .pkl-file contains a dictionary with the following fields:
            - sequence: String containing the sequence name
            - betas: SMPL shape parameters for each actor which has been used for tracking (List of 10x1 SMPL beta parameters)
            - poses: SMPL body poses for each actor aligned with image data (List of Nx72 SMPL joint angles, N = #frames)
            - trans: tranlations for each actor aligned with image data (List of Nx3 root translations)
            - poses_60Hz: SMPL body poses for each actor at 60Hz (List of Nx72 SMPL joint angles, N = #frames)
            - trans_60Hz: tranlations for each actor at 60Hz (List of Nx3 root translations)
            - betas_clothed: SMPL shape parameters for each clothed actor (List of 10x1 SMPL beta parameters)
            - v_template_clothed: 
            - gender: actor genders (List of strings, either 'm' or 'f')
            - texture_maps: texture maps for each actor
            - poses2D: 2D joint detections in Coco-Format for each actor (only provided if at least 6 joints were detected correctly)
            - jointPositions: 3D joint positions of each actor (List of Nx(24*3) XYZ coordinates of each SMPL joint)
            - img_frame_ids: an index-array to down-sample 60 Hz 3D poses to corresponding image frame ids
            - cam_poses: camera extrinsics for each image frame (Ix4x4 array, I frames times 4x4 homegenous rigid body motion matrices)
            - campose_valid: a boolean index array indicating which camera pose has been aligned to the image
            - cam_intrinsics: camera intrinsics (K = [f_x 0 c_x;0 f_y c_y; 0 0 1])

            Each sequence has either one or two models, which corresponds to the list size of the model specific fields (e.g. betas, poses, trans, v_template, gender, texture_maps, jointPositions, poses2D). 
            SMPL poses and translations are provided at 30 Hz. They are aligned to image dependent data (e.g. 2D poses, camera poses). In addition we provide 'poses_60Hz' and 'trans_60Hz' which corresponds to the recording frequency of 60Hz of the IMUs . You could use the 'img_frame_ids' to downsample and align 60Hz 3D and image dependent data, wich has been done to compute SMPL 'poses' and 'trans' variables. 
            Please refer to the demo.py-file for loading a sequence, setup smpl-Models and camera, and to visualize an example frame.
            """
            image_folder = os.path.join(THREEDPW_IMAGE, data['sequence'])    # e.g., '/data2/wxs/DATASETS/PW3D/imageFiles/downtown_car_00'
            img_path = os.path.join(image_folder, f"image_{data['img_frame_ids'][0]:05d}.jpg")  # 读取第 1 帧图片
            img = cv2.imread(img_path)
            res_h, res_w = img.shape[:2]  # (h, w)

            K = data['cam_intrinsics']  # (3,3)
            fx, fy, cx, cy =  K[0,0], K[1,1], K[0,2], K[1,2]
            assert res_h == cy * 2 and res_w == cx * 2  # e.g., res_h=1920, cy=960; res_w=1080, cx=540

            cam_extrinsics = data['cam_poses']  # (T,4,4).
            R = cam_extrinsics[:, :3, :3]   # (T,3,3)
            T = cam_extrinsics[:, :3, 3]   # (T,3). 单位: m

            cam_position = - np.einsum('twc,tw->tc', R, T)

            num_frames = len(data['img_frame_ids'])
            assert (data['poses2d'][0].shape[0] == num_frames)
            num_people = len(data['poses'])
            for p_id in range(num_people):
                gender = {'m': 'male', 'f': 'female'}[data['genders'][p_id]]
                smpl_gender = SMPL(model_path="/home/wxs/WHAM-main/dataset/body_models/smpl", gender=gender)

                pose = torch.from_numpy(data['poses'][p_id]).float()    # [T,72]
                shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)  # [T,10]
                trans = torch.from_numpy(data['trans'][p_id]).float()   # [T,3]

                smpl_output = smpl_gender(body_pose=pose[:,3:], 
                                          global_orient=pose[:,:3], 
                                          betas=shape, 
                                          transl=trans)
                
                joint_3d_world = np.einsum('jv,tvc->tjc', J_reg, smpl_output.vertices.cpu().numpy())    # [T,17,3]. 单位: m
                # viz_skel_seq_anim(joint_3d_world[::5],fs=1,lim3d=3,azim=-90,elev=90)
                
                joint_3d_cam = np.einsum('tcw,tjw->tjc', R, joint_3d_world) + T[:, None, :]
                joint_3d_cam = joint_3d_cam * 1000  # 转换为毫米, 单位: mm
    
                bboxes = _infer_box(joint_3d_cam, fx, fy, cx, cy, rootIdx=0) # (T,4). 单位: 像素
                joint_3d_image, ratio = camera_to_image_frame(joint_3d_cam, bboxes, fx, fy, cx, cy, rootIdx=0)   # (T,17,3). 单位: 像素. ratio: 像素/mm
                factor_2_5d = 1 / ratio[:,0]        # (T,). 单位: mm/像素

                joint_2d = joint_3d_image[..., :2].copy()  # (T,17,2). 单位: 像素

                joint_2d_Yflip = joint_2d.copy()
                joint_2d_Yflip[..., 1] = res_h - joint_2d_Yflip[..., 1]  # Y轴翻转, 单位: 像素


                source = data['sequence'] + str(p_id)

                ALL_DATA[split_name]['joint_2d'].append(joint_2d)
                ALL_DATA[split_name]['joint3d_image'].append(joint_3d_image)
                ALL_DATA[split_name]['joint3d_cam'].append(joint_3d_cam)
                ALL_DATA[split_name]['joint3d_world'].append(joint_3d_world)
                # ALL_DATA[split_name]['joints_2.5d_image'].append()
                ALL_DATA[split_name]['2.5d_factor'].append(factor_2_5d)
                ALL_DATA[split_name]['source'] = ALL_DATA[split_name]['source'] + [source] * joint_2d.shape[0]
                ALL_DATA[split_name]['res'].append(np.array([[res_w, res_h]]).repeat(joint_2d.shape[0], 0))
                ALL_DATA[split_name]['campose_valid'].append(data['campose_valid'][p_id])

                ALL_SMPL_DATA[split_name]['poses'].append(data['poses'][p_id])
                ALL_SMPL_DATA[split_name]['trans'].append(data['betas'][p_id][None, :10].repeat(joint_2d.shape[0], 0))
                ALL_SMPL_DATA[split_name]['betas'].append(data['trans'][p_id])
                ALL_SMPL_DATA[split_name]['gender'] = ALL_SMPL_DATA[split_name]['gender'] + [gender] * joint_2d.shape[0]
                ALL_SMPL_DATA[split_name]['source'] = ALL_SMPL_DATA[split_name]['source'] + [source] * joint_2d.shape[0]

                # TODO. 利用 3DPW 数据集提供的 demo.py 渲染第1帧姿态
                """
                frame_id = 0
                img_path = os.path.join(image_folder, f"image_{data['img_frame_ids'][frame_id]:05d}.jpg")
                camPose = cam_extrinsics[frame_id]  # (4,4)
                camIntrinsics = K.copy()  # (3,3)
                img = cv2.imread(img_path)
                class cam:
                    pass
                cam.rt = cv2.Rodrigues(camPose[0:3,0:3])[0].ravel()
                cam.t = camPose[0:3,3]
                cam.f = np.array([camIntrinsics[0,0],camIntrinsics[1,1]])
                cam.c = camIntrinsics[0:2,2]
                h = int(2*cam.c[1])
                w = int(2*cam.c[0])
                """

        ALL_DATA[split_name]['joint_2d'] = np.concatenate(ALL_DATA[split_name]['joint_2d'], axis=0)  # (T,17,2)
        ALL_DATA[split_name]['joint3d_image'] = np.concatenate(ALL_DATA[split_name]['joint3d_image'], axis=0)  # (T,17,3)
        ALL_DATA[split_name]['joint3d_cam'] = np.concatenate(ALL_DATA[split_name]['joint3d_cam'], axis=0)  # (T,17,3)
        ALL_DATA[split_name]['joint3d_world'] = np.concatenate(ALL_DATA[split_name]['joint3d_world'], axis=0)  # (T,17,3)
        ALL_DATA[split_name]['2.5d_factor'] = np.concatenate(ALL_DATA[split_name]['2.5d_factor'], axis=0)  # (T,)
        ALL_DATA[split_name]['res'] = np.concatenate(ALL_DATA[split_name]['res'], axis=0)  # (T,)
        ALL_DATA[split_name]['campose_valid'] = np.concatenate(ALL_DATA[split_name]['campose_valid'], axis=0)  # (T,)

        ALL_SMPL_DATA[split_name]['poses'] = np.concatenate(ALL_SMPL_DATA[split_name]['poses'], axis=0)  # (T,72)
        ALL_SMPL_DATA[split_name]['trans'] = np.concatenate(ALL_SMPL_DATA[split_name]['trans'], axis=0)  # (T,10)
        ALL_SMPL_DATA[split_name]['betas'] = np.concatenate(ALL_SMPL_DATA[split_name]['betas'], axis=0)  # (T,10)

    return 

def _infer_box(pose3d, fx, fy, cx, cy, rootIdx=0):
    root_joint = pose3d[..., rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[..., 0] -= 1000.0      # 相机坐标系下的根关节 x 坐标减去 1 m
    tl_joint[..., 1] -= 900.0      # 相机坐标系下的根关节 y 坐标减去 0.9 m
    br_joint = root_joint.copy()
    br_joint[..., 0] += 1000.0      # 相机坐标系下的根关节 x 坐标加上 1 m
    br_joint[..., 1] += 1100.0     # 相机坐标系下的根关节 y 坐标加上 1.1 m
    # x 方向: ±1000 mm. 假设人体宽度在 2 米范围内
    # y 方向: -900 mm (向头部扩展), +1100 mm (向脚部扩展). 假设人体高度在 2 米范围内. y 正方向在相机坐标系中通常是向下 (脚部方向)

    tl2d = _weak_project(tl_joint, fx, fy, cx, cy)
    br2d = _weak_project(br_joint, fx, fy, cx, cy)
    return np.stack([tl2d[...,0], tl2d[...,1], br2d[...,0], br2d[...,1]], axis=-1)

def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[..., :2] / pose3d[..., 2:3]
    pose2d[..., 0] *= fx
    pose2d[..., 1] *= fy
    pose2d[..., 0] += cx
    pose2d[..., 1] += cy
    return pose2d

def camera_to_image_frame(pose3d, box, fx, fy, cx, cy, rootIdx=0):
    rectangle_3d_size = 2000.0  # 单位: mm
    ratio = (box[..., 2] - box[..., 0] + 1) / rectangle_3d_size   # 单位: 像素/mm
    ratio = ratio[..., None]
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[..., :2] = _weak_project(pose3d.copy(), fx, fy, cx, cy)
    pose3d_depth = ratio * (pose3d[..., 2] - pose3d[..., rootIdx:rootIdx+1, 2])
    pose3d_image_frame[..., 2] = pose3d_depth
    return pose3d_image_frame, ratio

def get_valid_frame_indices(joint_2d, img_w, img_h):
    """
    joint_2d: (T, 17, 2) numpy array of 2D keypoints
    img_w, img_h: image width and height
    return: list of valid frame indices
    """
    x_valid = (joint_2d[..., 0] >= 0) & (joint_2d[..., 0] <= img_w)
    y_valid = (joint_2d[..., 1] >= 0) & (joint_2d[..., 1] <= img_h)
    all_valid = x_valid & y_valid
    frame_valid = all_valid.all(axis=1)  # shape (T,)
    return np.where(frame_valid)[0]

def split_valid_sequences(valid_indices, min_length=16):
    """
    Given sorted valid frame indices, split them into continuous sequences.
    Returns a list of (start_idx, end_idx) tuples where each range is inclusive.
    """
    if len(valid_indices) == 0:
        return []

    splits = []
    start = valid_indices[0]
    prev = start

    for idx in valid_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            if prev - start + 1 >= min_length:
                splits.append((start, prev))
            start = idx
            prev = idx
    if prev - start + 1 >= min_length:
        splits.append((start, prev))
    return splits

if __name__ == '__main__':
    main()
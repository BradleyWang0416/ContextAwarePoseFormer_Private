import pickle
import h5py
import cv2
import numpy as np

np.set_printoptions(suppress=True)

import os.path as osp
from scipy.io import loadmat
from subprocess import call
import os
from os import makedirs
# from spacepy import pycdf
from tqdm import tqdm
import cdflib
import sys
import json

from common.camera import *
from viz_skel_seq import viz_skel_seq_anim

from transform import get_affine_transform, affine_transform, \
    normalize_screen_coordinates, _infer_box, _weak_project 
from metadata import load_h36m_metadata
metadata = load_h36m_metadata()

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0   # 相机坐标系下的根关节 x 坐标减去 1 m
    tl_joint[1] -= 900.0    # 相机坐标系下的根关节 y 坐标减去 0.9 m
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0   # 相机坐标系下的根关节 x 坐标加上 1 m
    br_joint[1] += 1100.0   # 相机坐标系下的根关节 y 坐标加上 1.1 m
    # x 方向: ±1000 mm. 假设人体宽度在 2 米范围内
    # y 方向: -900 mm (向头部扩展), +1100 mm (向脚部扩展). 假设人体高度在 2 米范围内. y 正方向在相机坐标系中通常是向下 (脚部方向)
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def _infer_box_LCN(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0   # 相机坐标系下的根关节 x 坐标减去 1 m
    tl_joint[1] -= 1000.0    # 相机坐标系下的根关节 y 坐标减去 0.9 m
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0   # 相机坐标系下的根关节 x 坐标加上 1 m
    br_joint[1] += 1000.0   # 相机坐标系下的根关节 y 坐标加上 1.1 m
    # x 方向: ±1000 mm. 假设人体宽度在 2 米范围内
    # y 方向: -900 mm (向头部扩展), +1100 mm (向脚部扩展). 假设人体高度在 2 米范围内. y 正方向在相机坐标系中通常是向下 (脚部方向)
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

def _get_bbox_xywh(center, scale, w=200, h=200):
    w = w * scale
    h = h * scale
    x = center[0] - w / 2
    y = center[1] - h / 2
    return [x, y, w, h]

def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0  # 单位: mm
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size   # 单位: 像素/mm
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame

if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = [x for x in range(2, 17)]
    subaction_list = [x for x in range(1, 3)]
    camera_list = [x for x in range(1, 5)]

    train_list = [1, 5, 6, 7, 8]
    test_list = [9, 11]

    joint_idx = [0, # Hips
                 1, 2, 3,       # RightUpLeg, RightLeg, RightFoot
                 6, 7, 8,       # LeftUpLeg, LeftLeg, LeftFoot
                 12, 16, 14, 15,# Spine1, LeftShoulder (same as 24: RightShoulder & 13: Neck), Head, Site
                 17, 18, 19,    # LeftArm, LeftForeArm, LeftHand
                 25, 26, 27     # RightArm, RightForeArm, RightHand
                 ]

    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True) # dict_keys(['positions_2d', 'metadata'])
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    keypoints = keypoints['positions_2d'].item()   # dict_keys(['S6', 'S7', 'S5', 'S11', 'S1', 'S9', 'S8', 'S2', 'S3', 'S4'])
    # keypoints['S1']: dict_keys(['Directions 1', 'Discussion 1', ..., 'WalkDog', 'WalkTogether'])
    # keypoints['S1']['Directions 1']: List[Array]. len=4 (4 cameras?). [(1384,17,2), (1387,17,2), (1387,17,2), (1384,17,2)]
    # 数值量级为百, 说明是图像坐标系, 单位是像素

    # WARNING !
    # data_2d_h36m_cpn_ft_h36m_dbb.npz 缺少了 S11 的 Directions 的 4 个 camera 对应的 2D 检测数据

    with open('camera_data.pkl', 'rb') as f:
        camera_data = pickle.load(f)

    categories = [{
		'supercategory': 'person', 
		'id': 1, 
		'name': 'person', 
		'keypoints': ['root (pelvis)', 'left_hip', 'left_knee', 'left_foot', 'right_hip', 'right_knee', 'right_foot', 
		              'spine', 'thorax', 'neck_base', 'head', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'], 
		'skeleton': [
            [0, 1], [1, 2], [2, 3], 
            [0, 4], [4, 5], [5, 6], 
            [0, 7], [7, 8], [8, 9], [9, 10], 
            [8, 11], [11, 12], [12, 13], 
            [8, 14], [14, 15], [15, 16]]
	}]
    train_db = {'images': [], 'annotations': [], 'categories': categories}
    test_db = {'images': [], 'annotations': [], 'categories': categories}

    POSE3D_WORLD_ALL = []
    POSE3D_CAM_ALL = []
    POSE2D_ALL = []

    for data_split, subject_list_split in zip(['test', 'train'], [test_list, train_list]):
        global_id = 0
        video_id = 0
        for s in subject_list_split:
            for a in action_list:
                for sa in subaction_list:
                    
                    
                    filename_wo_camera = metadata.get_base_filename(f'S{s:d}', f'{a:d}', f'{sa:d}', metadata.camera_ids[0]).split('.')[0]  # 'Directions 1'
                    annofile3d_world = osp.join('/data2/wxs/DATASETS/Human3.6M_ByBradley/extracted', f'S{s}', 'D3_Positions', f'{filename_wo_camera}.npy')
                    pose3d_world = np.load(annofile3d_world)    # (T,17,3). 单位: mm
                    POSE3D_WORLD_ALL.append(pose3d_world)


                    for c in camera_list:

                        camera = camera_data[(s, c)]
                        camera_dict = {}
                        camera_dict['R'] = camera[0]
                        camera_dict['T'] = camera[1]    # (3,1). TODO. 单位: ???
                        camera_dict['fx'] = camera[2][0]
                        camera_dict['fy'] = camera[2][1]
                        camera_dict['cx'] = camera[3][0]
                        camera_dict['cy'] = camera[3][1]
                        camera_dict['k'] = camera[4]
                        camera_dict['p'] = camera[5]

                        subject = f'S{s}' # 'S1'

                        basename = metadata.get_base_filename(f'S{s:d}', f'{a:d}', f'{sa:d}', metadata.camera_ids[c-1])     # 'Directions 1.54138969'
                        annotname = basename + '.cdf'

                        imagebasename = basename.replace(' ', '_')     # 'Directions_1.54138969'
                        imagesubdir = osp.join(subject, f"{subject}_{imagebasename}")  # 'S1/S1_Directions_1.54138969'

                        annofile3d_camera = osp.join('/data2/wxs/DATASETS/Human3.6M_MMPose/extracted', subject, 'MyPoseFeatures', 'D3_Positions_mono', annotname)
                        annofile2d = osp.join('/data2/wxs/DATASETS/Human3.6M_MMPose/extracted', subject, 'MyPoseFeatures', 'D2_Positions', annotname)

                        data = cdflib.CDF(annofile3d_camera)
                        pose3d_camera = np.array(data.varget("Pose"))
                        pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))[:, joint_idx, :]
                        # pose3d_camera 和 pose3d_world 应满足如下关系:
                        # pose3d_camera = np.einsum('cw,tjw->tjc', camera_dict['R'], (pose3d_world - camera_dict['T'].reshape(1, 1, 3)))

                        POSE3D_CAM_ALL.append(pose3d_camera)

                        # viz_skel_seq_anim({'world':pose3d_world[:10], 'cam':pose3d_camera[:10]},fs=0.5,lim3d=5000)


                        if basename.split('.')[0] not in keypoints[f'S{s:d}'].keys():
                            if "TakingPhoto" in basename:
                                basename = basename.replace("TakingPhoto", "Photo")
                            elif "WalkingDog" in basename:
                                basename = basename.replace("WalkingDog", "WalkDog")
                            else:
                                print(basename.split('.')[0] + " is missing!")
                                continue
                        
                        """
                        box = _infer_box(pose3d_camera[0], camera_dict, 0)
                        camera_to_image_frame(pose3d_camera[0], box, camera_dict, rootIdx=0)
                        """

                        data = cdflib.CDF(annofile2d)
                        pose2d_gt = np.array(data.varget("Pose"))
                        pose2d_gt = np.reshape(pose2d_gt, (-1, 32, 2))[:, joint_idx, :]
                        POSE2D_ALL.append(pose2d_gt)
                        continue

                        pose2d_cpn = keypoints['S{:d}'.format(s)][basename.split('.')[0]][c-1]

                        nposes = min(pose3d_camera.shape[0], pose2d_gt.shape[0])
                        if pose2d_cpn.shape[0] > nposes:
                            pose2d_cpn = pose2d_cpn[:nposes]
                        assert pose2d_gt.shape[0] == pose2d_cpn.shape[0] 
                        assert pose2d_cpn.shape[0] == pose3d_camera.shape[0]

                        video_id += 1
                        for i in tqdm(range(nposes)):
                            global_id += 1
                            imagename = f"{subject}_{imagebasename}_{i+1:06d}.jpg"
                            imagepath = osp.join(imagesubdir, imagename)

                            data_numpy = cv2.imread(osp.join('/data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50', imagepath), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                            h, w, _ = data_numpy.shape

                            imageinfo = {
                                'file_name': imagepath,
                                'height': h,
                                'width': w,
                                'id': global_id,
                            }

                            datum = {}
                            datum['id'] = global_id
                            datum['category_id'] = 1  # Only one category for human pose
                            datum['image_id'] = global_id
                            datum['video_id'] = video_id
                            datum['iscrowd'] = 0

                            datum['keypoints'] = pose2d_gt[i, :, :] # (17,2)
                            datum['keypoints_cpn'] = pose2d_cpn[i]  # (17,2)
                            datum['keypoints_3d'] = pose3d_camera[i, :, :]  # (17,3)

                            box = _infer_box(datum['keypoints_3d'], camera_dict, 0)     # [x_min, y_min, x_max, y_max] 格式, 4个数字分别代表: 左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标. 单位: 像素
                            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            # scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                            scale = ((box[2] - box[0]), (box[3] - box[1]))

                            joint_3d_image = camera_to_image_frame(datum['keypoints_3d'], box, camera_dict, rootIdx=0)

                            datum['center'] = center
                            datum['scale'] = scale
                            datum['bbox_xyxy'] = box

                            datum['source'] = (s, a, sa, c)

                            datum['keypoints'] = datum['keypoints'].reshape(-1).tolist()
                            datum['keypoints_cpn'] = datum['keypoints_cpn'].reshape(-1).tolist()
                            datum['keypoints_3d'] = datum['keypoints_3d'].reshape(-1).tolist()
                            datum['bbox_xyxy'] = datum['bbox_xyxy'].reshape(-1).tolist()
                            
                            if data_split == 'train':
                                train_db['images'].append(imageinfo)
                                train_db['annotations'].append(datum)
                            else:
                                test_db['images'].append(imageinfo)
                                test_db['annotations'].append(datum)


        # with open(f'/data1/wxs/ContextAware-PoseFormer/H36M-Toolbox/h36m_coco_{data_split}.json', 'w') as f:
        #     if data_split == 'train':
        #         json.dump(train_db, f)
        #     else:
        #         json.dump(test_db, f)

    POSE3D_WORLD_ALL = np.concatenate(POSE3D_WORLD_ALL, axis=0)
    POSE3D_CAM_ALL = np.concatenate(POSE3D_CAM_ALL, axis=0)
    POSE2D_ALL = np.concatenate(POSE2D_ALL, axis=0)

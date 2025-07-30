import sys
sys.path.append("/home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/")
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

from common import transforms
from viz_skel_seq import viz_skel_seq_anim

with open("/data2/wxs/DATASETS/Human3.6M_MMPose/processed/annotation_body3d/cameras.pkl", 'rb') as f:
    cameras = pickle.load(f)

with open('/data2/wxs/DATASETS/AMASS_for_MotionBERT/amass_joints_h36m_60.pkl', 'rb') as f:
    amass_db = pickle.load(f)

video_info_list = pd.read_csv("/data2/wxs/DATASETS/AMASS_for_MotionBERT/clip_list.csv")
video_info_list = video_info_list.values

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

def create_camera(
        w,
        f,
        # pitch_std=np.pi/8,    # 22.5°
        # pitch_mean=np.pi/36,  # 5°
        pitch_std=np.pi/12,      # 15°
        pitch_mean= - np.pi/36, # -5°
        roll_std=np.pi/24,
        tz_scale=10,
        tz_min=2,
    ):
    """Create the initial frame camera pose"""
    yaw = np.random.rand() * 2 * np.pi
    pitch = np.random.normal(scale=pitch_std) + pitch_mean
    roll = np.random.normal(scale=roll_std)
    
    yaw_rm = transforms.axis_angle_to_matrix(torch.tensor([0, 0, yaw]).float())
    pitch_rm = transforms.axis_angle_to_matrix(torch.tensor([0, pitch, 0]).float())
    roll_rm = transforms.axis_angle_to_matrix(torch.tensor([roll, 0, 0]).float())
    # yaw_rm = transforms.axis_angle_to_matrix(torch.tensor([0, yaw, 0]).float())
    # pitch_rm = transforms.axis_angle_to_matrix(torch.tensor([pitch, 0, 0]).float())
    # roll_rm = transforms.axis_angle_to_matrix(torch.tensor([0, 0, roll]).float())
    R = (roll_rm @ pitch_rm @ yaw_rm)
    
    # Place people in the scene
    tz = np.random.rand() * tz_scale + tz_min
    max_d = w * tz / f / 2
    tx = np.random.normal(scale=0.25) * max_d
    ty = np.random.normal(scale=0.25) * max_d
    dist = torch.tensor([tx, ty, tz]).float()    
    return R.numpy(), dist.numpy()

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

# 基于 H36M 的统计数据, 详见 /home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/generate_labels_h36m_inCocoStructure.py
world_coordinate_x_min = -1.5438152
world_coordinate_x_max = 2.1130056
world_coordinate_y_min = -2.1536294
world_coordinate_y_max = 2.7382566

POSE3D_WORLD_ALL = {'train': [], 'test': []}
POSE3D_CAM_ALL = {'train': [], 'test': []}
POSE3D_IMAGE_ALL = {'train': [], 'test': []}
POSE2D_ALL = {'train': [], 'test': []}
SOURCE_ALL = {'train': [], 'test': []}
CAM_ALL = {'train': [], 'test': []}
FACTOR_2_5D = {'train': [], 'test': []}

camera_source = 'h36m'  # h36m; virtual

video_id = [191]
video_id = [30,32,34,35,39,42,43,44,46,55,61,68,69,71,73,99,103,188,190,191,195,198,202,211,212,215,224,11190,11194,11244]  # out of bounds (x_min < world_coordinate_x_min and x_max > world_coordinate_x_max and y_min < world_coordinate_y_min and y_max > world_coordinate_y_max)
assert len(amass_db) == len(video_info_list)
for video_id, joint_3d_world in enumerate(tqdm(amass_db)):
    # 世界坐标系, 单位: m
    video_info = video_info_list[video_id]
    video_amass_source = video_info[1]
    if 'BioMotionLab' in video_amass_source:
        data_split = 'test'
    else:
        data_split = 'train'
    num_frames = joint_3d_world.shape[0]

    camera_key = random.choice(list(cameras.keys()))
    camera = cameras[camera_key]
    # dict_keys(['R', 'T', 'c', 'f', 'k', 'p', 'w', 'h', 'name', 'id'])
    #   R, T: extrinsics. camera rotation and translation
    #   c, f: intrinsics. camera center and focal length
    #   k, p: distortion coefficients
    #   w, h: image width and height
    img_w, img_h = camera['w'], camera['h']  # 单位: 像素
    center = camera['c']    # (2,1)
    focal = camera['f']      # (2,1)
    cx, cy = center[0, 0], center[1, 0]
    fx, fy = focal[0, 0], focal[1, 0]
    # fov_tol = 1.2 * (0.5 ** 0.5)
    # img_w, img_h = 1000, 1000
    # cx, cy = img_w / 2, img_h / 2
    # fx = (img_w * img_w + img_h * img_h) ** 0.5
    # fy = fx
    if camera_source == 'h36m':
        R = camera['R'] # (3,3), 单位: m
        R = R.T     # MMPOSE 的旋转矩阵转置后才是 H36M-TOOLBOX 中的旋转矩阵. See: sapiens/pose/tools/preprocess_h36m.py::line275 >>> R = (R_x @ R_y @ R_z).T
        T = camera['T'] # (3,1), 单位: m. 即相机在世界坐标系中的位置
        joint_3d_cam = np.einsum('cw,tjw->tjc', R, (joint_3d_world - T.reshape(1, 1, 3)))   # (T,17,3)
        cam_position = T.copy().reshape(3)   # (3,)
    elif camera_source == 'virtual':
        R = camera['R']
        R = R.T
        _, dist = create_camera(w=img_w, f=fx)
        T = joint_3d_world[num_frames//2, 0, :] - np.einsum('wc,w->c', R, dist)
        T = T[:, None]
        joint_3d_cam = np.einsum('cw,tjw->tjc', R, (joint_3d_world - T.reshape(1, 1, 3)))   # (T,17,3)
        cam_position = T.copy().reshape(3)   # (3,)
    # 相机坐标系, 单位: m

    joint_3d_cam = joint_3d_cam * 1000  # 转换为毫米, 单位: mm

    
    bboxes = _infer_box(joint_3d_cam, fx, fy, cx, cy, rootIdx=0) # (T,4). 单位: 像素
    joint_3d_image, ratio = camera_to_image_frame(joint_3d_cam, bboxes, fx, fy, cx, cy, rootIdx=0)   # (T,17,3). 单位: 像素. ratio: 像素/mm
    factor_2_5d = 1 / ratio[:,0]        # (T,). 单位: mm/像素

    joint_2d = joint_3d_image[..., :2].copy()  # (T,17,2). 单位: 像素
    joint_2d_Yflip = joint_2d.copy()
    joint_2d_Yflip[..., 1] = img_h - joint_2d_Yflip[..., 1]  # Y轴翻转, 单位: 像素

    valid_frame_indices = get_valid_frame_indices(joint_2d_Yflip, img_w, img_h)
    valid_video_slices = split_valid_sequences(valid_frame_indices, min_length=16)

    for valid_video_slice in valid_video_slices:
        num_slice_frames = valid_video_slice[1] - valid_video_slice[0] + 1
        source = f"vid{video_id}_cam{camera_key[0]}-{camera_key[1]}_frame{valid_video_slice[0]}-{valid_video_slice[1]}"

        # SOURCE_ALL[data_split] = SOURCE_ALL[data_split] + [source] * num_slice_frames
        # POSE3D_WORLD_ALL[data_split].append(joint_3d_world[valid_video_slice[0]: valid_video_slice[1] + 1])
        # POSE3D_CAM_ALL[data_split].append(joint_3d_cam[valid_video_slice[0]: valid_video_slice[1] + 1])
        # POSE2D_ALL[data_split].append(joint_2d[valid_video_slice[0]: valid_video_slice[1] + 1])
        # CAM_ALL[data_split] = CAM_ALL[data_split] + [camera] * num_slice_frames

        # POSE3D_IMAGE_ALL[data_split].append(joint_3d_image[valid_video_slice[0]: valid_video_slice[1] + 1])
        # FACTOR_2_5D[data_split].append(factor_2_5d[valid_video_slice[0]: valid_video_slice[1] + 1])


    """
    可视化相机位置以及朝向
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-90, elev=90)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim3d([-4, 4]); ax.set_ylim3d([-4, 4]); ax.set_zlim3d([-4, 4])
    ax.scatter(*T[:,0], s=100)
    ax.quiver(*T[:,0], *R[2], length=0.5)

    viz_skel_seq_anim(joint_3d_world[::10],fs=0.5,lim3d=4,azim=-90,elev=90)
    viz_skel_seq_anim(joint_3d_cam[::10],fs=0.5,lim3d=4,azim=-90,elev=90)
    viz_skel_seq_anim(T.reshape(1,1,3),fs=0.5,lim3d=4,azim=-90,elev=90)
    viz_skel_seq_anim(joint_2d_Yflip[::10],fs=0.5,lim2d=[0,1000])
    """

# POSE3D_WORLD_ALL['train'] = np.concatenate(POSE3D_WORLD_ALL['train'], axis=0)
# POSE3D_WORLD_ALL['test'] = np.concatenate(POSE3D_WORLD_ALL['test'], axis=0)
# POSE3D_CAM_ALL['train'] = np.concatenate(POSE3D_CAM_ALL['train'], axis=0)
# POSE3D_CAM_ALL['test'] = np.concatenate(POSE3D_CAM_ALL['test'], axis=0)
# POSE2D_ALL['train'] = np.concatenate(POSE2D_ALL['train'], axis=0)
# POSE2D_ALL['test'] = np.concatenate(POSE2D_ALL['test'], axis=0)
# POSE3D_IMAGE_ALL['train'] = np.concatenate(POSE3D_IMAGE_ALL['train'], axis=0)
# POSE3D_IMAGE_ALL['test'] = np.concatenate(POSE3D_IMAGE_ALL['test'], axis=0)
# FACTOR_2_5D['train'] = np.concatenate(FACTOR_2_5D['train'], axis=0)
# FACTOR_2_5D['test'] = np.concatenate(FACTOR_2_5D['test'], axis=0)

# joblib.dump(POSE3D_WORLD_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/joint_3d_world.pkl', compress=3)
# joblib.dump(POSE3D_CAM_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/joint_3d_cam.pkl', compress=3)
# joblib.dump(POSE2D_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/joint_2d_image.pkl', compress=3)
# joblib.dump(SOURCE_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/source.pkl', compress=0)
# joblib.dump(CAM_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/cam_params.pkl', compress=0)
# joblib.dump(POSE3D_IMAGE_ALL, '/data2/wxs/DATASETS/AMASS_ByBradley/joint_3d_image.pkl', compress=3)
# joblib.dump(FACTOR_2_5D, '/data2/wxs/DATASETS/AMASS_ByBradley/factor_2_5d.pkl', compress=0)


import os
import collections
from collections import defaultdict
import pickle
import joblib
import numpy as np
import cv2
import torch
import shutil
import random
from tqdm import tqdm

from mvn.datasets import utils as dataset_utils
from mvn.datasets.utils import data_prefetcher

import sys
sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
from funcs_and_classes.Non_AR.dataset.ver13_ICL import DataReaderMesh
from lib.utils.utils_data import split_clips
from lib.utils.viz_skel_seq import viz_skel_seq_anim

sys.path.append('/home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/')
from preprocess_h36m_03AffineImage_byBradley import get_affine_transform, affine_transform

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]


class H36m3D_MultiFrame(torch.utils.data.Dataset):
    def __init__(self, num_frames=16, sample_stride=1, data_stride=16, data_mode="joint3d", designated_split='train',
                 processed_image_shape=(192,256), 
                 load_data_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl", 
                 load_image_source_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl", 
                 load_bbox_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl",
                 load_text_source_file="", 
                 return_extra=[['image']],                 
                 # e.g.,
                 # lode_data_file='<h36m_path>,<amass_path>'
                 # load_image_source_file='<h36m_img_path>,'
                 # load_text_source_file=',<amass_text_path>'
                 # return_extra=[['image'], ['text']]
                 ):
        assert data_mode == 'joint3d', 'due to the current affine transform code implementation, only support [data_mode=joint3d] now.'
        assert len(load_data_file.split(',')) == len(load_image_source_file.split(',')) == len(return_extra) == len(load_bbox_file.split(','))

        self.num_frames = num_frames

        data_dict = {}
        data_list = []
        for dt_file, img_src_file, bbox_file, extra_modality_list in zip(load_data_file.split(','), load_image_source_file.split(','), load_bbox_file.split(','), return_extra):
            ######################################################### load image data; find indices with valid images; do not apply sample_stride #########################################################
            img_list = joblib.load(img_src_file)[designated_split]
            valid_img_indices = []
            for frame_id, img_path in enumerate(img_list):
                if img_path is None:
                    continue
                valid_img_indices.append(frame_id)
                img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                assert os.path.exists(img_list[frame_id]), f'only supports [processed_image_shape=(192,256)] now. other settings not implemented yet.'

                if 'debugpy' in sys.modules and designated_split == 'train':
                    if len(valid_img_indices) >= 32:   # for debug purpose
                        break

            img_list = np.array(img_list)[valid_img_indices]   # resample according to valid_img_indices (sample_stride not applied yet here)
            img_list = img_list[::sample_stride]  # sample_stride applied here

            ######################################################### load joint data; resample according to indices with valid images #########################################################
            datareader_config_unsplit = {'dt_file': dt_file,}
            datareader_config_split = {'chunk_len': num_frames,
                                       'sample_stride': sample_stride, 
                                       'data_stride': data_stride,
                                       'read_confidence': False}
            datareader_config = {**datareader_config_unsplit, **datareader_config_split}
            datareader = DataReaderMesh(**datareader_config)        
            unsplit_data = DataReaderMesh.load_dataset_static(**datareader_config_unsplit)   # '/data2/wxs/DATASETS/AMASS_ByBradley'

            for data_mode in unsplit_data[designated_split].keys():
                if isinstance(unsplit_data[designated_split][data_mode], list):
                    unsplit_data[designated_split][data_mode] = np.array(unsplit_data[designated_split][data_mode])[valid_img_indices].tolist()   # resample according to valid_img_indices (sample_stride not applied yet here)
                else:
                    unsplit_data[designated_split][data_mode] = unsplit_data[designated_split][data_mode][valid_img_indices]

            datareader.dt_dataset = unsplit_data
            data_npy = datareader.read_3d_image(designated_split=designated_split, do_screen_coordinate_normalize=False)     # (N,17,3).sample_stride applied here

            ######################################################### source data part #########################################################
            data_sources = datareader.read_source(designated_split=designated_split)    # sampled_stride applied within read_source

            ######################################################### 2.5d factory data part #########################################################
            if designated_split == 'test':
                factor_2_5d = datareader.read_2_5d_factor(designated_split=designated_split)    # sampled_stride applied within read_source
            else:
                factor_2_5d = np.zeros((data_npy.shape[0],), dtype=np.float32)

            ######################################################### original height and weight part #########################################################
            img_ori_hw = datareader.read_hw(designated_split=designated_split)    # sampled_stride applied within read_hw

            ######################################################### bbox data part #########################################################
            bboxes_xyxy = joblib.load(bbox_file)[designated_split]
            bboxes_xyxy = bboxes_xyxy[valid_img_indices]
            bboxes_xyxy = bboxes_xyxy[::sample_stride]

            ######################################################### do a sanity check #########################################################
            assert len(img_list) == data_npy.shape[0] == len(data_sources) == len(bboxes_xyxy) == len(img_ori_hw)

            ######################################################### affine poses to align with images ########################################################
            data_npy_affined = np.zeros_like(data_npy)
            for i in range(data_npy.shape[0]):
                bbox = bboxes_xyxy[i]
                center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
                scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                trans = get_affine_transform(center, scale, 0, processed_image_shape)

                pose_xy = data_npy[i, :, :2].copy()   # (17,2)
                pose_xy1 = np.concatenate([pose_xy, np.ones((pose_xy.shape[0],1))], axis=1)   # (17,3)
                pose_xy_affined = np.einsum('ij,kj->ik', pose_xy1, trans)

                pose_z = data_npy[i, :, 2:3].copy()   # (17,1). pose_z[0] should already be zero
                pose_z_affined = pose_z - pose_z[0:1, :]   # root-relative. pose_z_affined should be the same as pose_z

                data_npy_affined[i, :, :2] = pose_xy_affined
                data_npy_affined[i, :, 2:3] = pose_z_affined

                if False:
                    pose_xy_affined_viz = pose_xy_affined.copy()
                    pose_xy_affined_viz[:, 1] = processed_image_shape[1] - pose_xy_affined_viz[:, 1]
                    viz_skel_seq_anim(pose_xy_affined_viz[None],fs=0.4,if_print=1,file_folder='.',file_name='tmp',lim2d={'x':[0,192],'y':[0,256]},mode='img')
                    shutil.copy(img_list[i].item(), 'tmp.jpg')
            
            assert (data_npy_affined[..., 2] == data_npy[..., 2]).all()   # pose_z should be the same


            data_dict[dt_file] = {'poses': data_npy,
                                  'poses_affined': data_npy_affined,
                                  'image_sources': img_list,
                                  'sources': data_sources,
                                  'bboxes_xyxy': bboxes_xyxy,
                                  'img_ori_hw': img_ori_hw,
                                  '2.5d_factor': factor_2_5d,
                                  }

            # Get split_id
            split_id = datareader.get_split_id(designated_split=designated_split)   # 这里是用 unsplit_data 中的 'source' 来划分 split_id, 所以也要利用 valid_indices 作修改

            data_list.extend(zip([dt_file]*len(split_id), split_id, [None]*len(split_id)))

        self.data_dict = data_dict
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dt_file, slice_id, caption = self.data_list[idx]
        joint3d_image = self.data_dict[dt_file]['poses'][slice_id]  # (num_frames, 17, 3)
        joint3d_image_affined = self.data_dict[dt_file]['poses_affined'][slice_id]  # (num_frames, 17, 3)
        image_sources = self.data_dict[dt_file]['image_sources'][slice_id]  # (num_frames,)
        img_ori_hw = self.data_dict[dt_file]['img_ori_hw'][slice_id]  # (num_frames, 2). element: (res_w, res_h)
        # assert (img_ori_hw[0:1, :] == img_ori_hw[1:, :]).all()
        factor_2_5d = self.data_dict[dt_file]['2.5d_factor'][slice_id]  # (num_frames,) only for test

        video_bgr = []
        for img_path in image_sources:
            assert os.path.exists(img_path), f'img_path={img_path} not exists.'
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            video_bgr.append(image_bgr)
        video_bgr = np.stack(video_bgr, axis=0)  # (num_frames, H, W, 3), BGR order
        
        return video_bgr, joint3d_image_affined, joint3d_image, img_ori_hw, factor_2_5d


class data_prefetcher_H36m3D_MultiFrame(data_prefetcher):
    def __init__(self, *args, processed_image_shape=(192,256), **kwargs):

        self.processed_image_shape = processed_image_shape

        super().__init__(*args, **kwargs)

    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)):   # iterate over each object returned by __getitem__
                self.next_batch[i] = self.next_batch[i].cuda(non_blocking=True).to(self.device)

            video_bgr, joint3d_image_affined, joint3d_image, img_ori_hw, factor_2_5d = self.next_batch   # img_ori_hw: (B, num_frames, 2). element: (res_w, res_h)

            video_rgb = torch.flip(video_bgr, dims=[-1])   # (B, num_frames, H, W, 3), RGB order
            if self.backbone in ['hrnet_32', 'hrnet_48']:
                video_rgb = (video_rgb / 255.0 - self.mean) / self.std   # to [0,1], then normalize
            elif self.backbone == 'cpn':
                video_rgb = video_rgb / 255.0 - self.mean  # for CPN

            if random.random() <= 0.5 and self.is_train:
                video_rgb = torch.flip(video_rgb, [-2])

                joint3d_image_affined[..., 0] = self.processed_image_shape[0] - joint3d_image_affined[..., 0] - 1
                joint3d_image_affined[..., joints_left + joints_right, :] = joint3d_image_affined[..., joints_right + joints_left, :]

                joint3d_image[..., 0] = img_ori_hw[..., 0:1] - joint3d_image[..., 0] - 1
                joint3d_image[..., joints_left + joints_right, :] = joint3d_image[..., joints_right + joints_left, :]

            if (not self.is_train) and self.flip_test:
                video_rgb = torch.stack([video_rgb, torch.flip(video_rgb, [-2])], dim=1)    # (B, 2, num_frames, H, W, 3)

                joint3d_image_affined_flip = joint3d_image_affined.clone()
                joint3d_image_affined_flip[..., 0] = self.processed_image_shape[0] - joint3d_image_affined_flip[..., 0] - 1
                joint3d_image_affined_flip[..., joints_left + joints_right, :] = joint3d_image_affined_flip[..., joints_right + joints_left, :]
                joint3d_image_affined = torch.stack([joint3d_image_affined, joint3d_image_affined_flip], dim=1)

                joint3d_image_flip = joint3d_image.clone()
                joint3d_image_flip[..., 0] = img_ori_hw[..., 0:1] - joint3d_image_flip[..., 0] - 1
                joint3d_image_flip[..., joints_left + joints_right, :] = joint3d_image_flip[..., joints_right + joints_left, :]
                joint3d_image = torch.stack([joint3d_image, joint3d_image_flip], dim=1)

                del joint3d_image_affined_flip, joint3d_image_flip

            self.next_batch = [video_rgb.float(), joint3d_image_affined.float(), joint3d_image.float(), img_ori_hw, factor_2_5d.float()]


if __name__ == '__main__':
    dataset = H36m3D_MultiFrame()
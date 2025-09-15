import os
import joblib
import numpy as np
import cv2
import torch
import shutil
from collections import defaultdict

import sys
sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
from funcs_and_classes.Non_AR.dataset.ver13_ICL import DataReaderMesh
from lib.utils.viz_skel_seq import viz_skel_seq_anim

from preprocess_h36m_03AffineImage_byBradley import get_affine_transform


joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]


class Multimodal_Mocap_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=16, sample_stride=1, data_stride=16, data_mode="joint3d", designated_split='train',
                 load_data_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl", 
                 load_image_source_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl", 
                 load_bbox_file="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl",
                 load_text_source_file="",
                 return_extra=[['image']],
                 # data preprocessing config
                 normalize='isotropic',  # isotropic (i.e., screen_coordinate_normalize), anisotropic
                 # image config
                 filter_invalid_images=True,
                 processed_image_shape=None,    # e.g., (192,256)
                 backbone='hrnet_32',
                 # dataloader config
                 get_item_list=[],
                 ):
        # e.g.,
        # lode_data_file='<h36m_path>,<amass_path>'
        # load_image_source_file='<h36m_img_path>,'
        # load_text_source_file=',<amass_text_path>'
        # return_extra=[['image'], ['text']]
        assert data_mode == 'joint3d', 'due to the current affine transform code implementation, only support [data_mode=joint3d] now.'
        assert len(load_data_file.split(',')) == len(load_image_source_file.split(',')) == len(return_extra) == len(load_bbox_file.split(','))

        self.num_frames = num_frames
        self.get_item_list = get_item_list
        # e.g., ['joint3d_image', 'joint3d_image_normed', 'factor_2_5d', 'joint3d_image_scale', 'joint3d_image_transl']
        assert len(self.get_item_list) > 0

        if backbone in ['hrnet_32', 'hrnet_48']:
            self.img_mean = np.array([0.485, 0.456, 0.406])
            self.img_std = np.array([0.229, 0.224, 0.225])
        elif backbone == 'cpn':
            self.img_mean = np.array([122.7717, 115.9465, 102.9801])
            self.img_mean /= 255.
            self.img_std = 1    # placeholder
        else:
            NotImplementedError

        data_dict = {}
        data_list = []
        for dt_file, img_src_file, bbox_file, extra_modality_list in zip(load_data_file.split(','), load_image_source_file.split(','), load_bbox_file.split(','), return_extra):
            ######################################################### load image data; find indices with valid images; do not apply sample_stride #########################################################
            use_image = 'image' in extra_modality_list
            if use_image:              
                img_list = joblib.load(img_src_file)[designated_split]
                if filter_invalid_images:
                    valid_img_indices = []
                    for frame_id, img_path in enumerate(img_list):
                        if img_path is None:
                            continue
                        valid_img_indices.append(frame_id)
                        if 'debugpy' in sys.modules and designated_split == 'train' and len(valid_img_indices) >= 32:   # for debug purpose
                            break
                else:
                    valid_img_indices = list(range(len(img_list)))


                img_list = np.array(img_list)[valid_img_indices]   # resample according to valid_img_indices (sample_stride not applied yet here)
                img_list = img_list[::sample_stride]  # sample_stride applied here


                if processed_image_shape is not None:
                    img_list = img_list.tolist()
                    assert processed_image_shape[0] == 192 and processed_image_shape[1] in [256], f'only supports [processed_image_shape=(192,256)] now. other settings not implemented yet.'
                    for frame_id, img_path in enumerate(img_list):
                        img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_{processed_image_shape[0]}x{processed_image_shape[1]}')
                        assert os.path.exists(img_list[frame_id]), f'img_list[frame_id]={img_list[frame_id]} not exists.'
                    img_list = np.array(img_list)
            else:
                valid_img_indices = slice(None)   # all valid

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
            joint3d_image = datareader.read_3d_image(designated_split=designated_split, do_screen_coordinate_normalize=False)     # (N,17,3). sample_stride applied here


            ######################################################### original height and weight part; get norm and denorm factor #########################################################
            img_ori_wh = datareader.read_hw(designated_split=designated_split)    # (N,2). sampled_stride applied within read_hw
            img_ori_w, img_ori_h = img_ori_wh[:, 0:1], img_ori_wh[:, 1:2]   # (N,1); (N,1)
            if normalize == 'isotropic':
                joint3d_image_scale = np.concatenate([img_ori_w / 2, img_ori_w / 2, img_ori_w / 2], axis=-1) # (N,3)
                joint3d_image_transl = np.concatenate([np.ones_like(img_ori_w), img_ori_h / img_ori_w, np.zeros_like(img_ori_w)], axis=-1) # (N,3)
            elif normalize == 'anisotropic':
                joint3d_image_scale = np.concatenate([img_ori_w // 2, img_ori_h // 2, img_ori_w / 2], axis=-1) # (N,3)
                joint3d_image_transl = np.concatenate([np.ones_like(img_ori_w), np.ones_like(img_ori_h), np.zeros_like(img_ori_w)], axis=-1) # (N,3)
            else:
                NotImplementedError


            ######################################################### source data part #########################################################
            data_sources = datareader.read_source(designated_split=designated_split)    # sampled_stride applied within read_source

            ######################################################### 2.5d factory data part #########################################################
            if designated_split == 'test':
                factor_2_5d = datareader.read_2_5d_factor(designated_split=designated_split)    # sampled_stride applied within read_source
            else:
                factor_2_5d = np.zeros((joint3d_image.shape[0],), dtype=np.float32)

            ######################################################### bbox data part #########################################################
            bboxes_xyxy = joblib.load(bbox_file)[designated_split]
            bboxes_xyxy = bboxes_xyxy[valid_img_indices]
            bboxes_xyxy = bboxes_xyxy[::sample_stride]

            ######################################################### do a sanity check; store data #########################################################
            assert joint3d_image.shape[0] == len(data_sources) == len(bboxes_xyxy) == len(img_ori_wh)
            data_dict[dt_file] = {'joint3d_image': joint3d_image,   # (N,17,3)
                                  'sources': data_sources,   # (N,)
                                  'bboxes_xyxy': bboxes_xyxy,
                                  'ori_img_wh': img_ori_wh,   # (N,2)
                                  '2.5d_factor': factor_2_5d,   # (N,)
                                  'joint3d_image_scale': joint3d_image_scale,   # (N,3)
                                  'joint3d_image_transl': joint3d_image_transl,   # (N,3)
                                  }            
            if use_image:
                assert joint3d_image.shape[0] == len(img_list)
                data_dict[dt_file]['image_sources'] = img_list

            ######################################################### affine poses to align with images ########################################################
            if use_image and processed_image_shape is not None:
                joint3d_image_affined = np.zeros_like(joint3d_image)
                for i in range(joint3d_image.shape[0]):
                    bbox = bboxes_xyxy[i]
                    center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
                    scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    trans = get_affine_transform(center, scale, 0, processed_image_shape)

                    pose_xy = joint3d_image[i, :, :2].copy()   # (17,2)
                    pose_xy1 = np.concatenate([pose_xy, np.ones((pose_xy.shape[0],1))], axis=1)   # (17,3)
                    pose_xy_affined = np.einsum('ij,kj->ik', pose_xy1, trans)

                    pose_z = joint3d_image[i, :, 2:3].copy()   # (17,1). pose_z[0] should already be zero
                    pose_z_affined = pose_z - pose_z[0:1, :]   # root-relative. pose_z_affined should be the same as pose_z

                    joint3d_image_affined[i, :, :2] = pose_xy_affined
                    joint3d_image_affined[i, :, 2:3] = pose_z_affined

                    if False:
                        pose_xy_affined_viz = pose_xy_affined.copy()
                        pose_xy_affined_viz[:, 1] = processed_image_shape[1] - pose_xy_affined_viz[:, 1]
                        viz_skel_seq_anim(pose_xy_affined_viz[None],fs=0.4,if_print=1,file_folder='.',file_name='tmp',lim2d={'x':[0,192],'y':[0,256]},mode='img')
                        shutil.copy(img_list[i].item(), 'tmp.jpg')
                
                assert (joint3d_image_affined[..., 2] == joint3d_image[..., 2]).all()   # pose_z should be the same

                data_dict[dt_file]['joint3d_image_affined'] = joint3d_image_affined
                data_dict[dt_file]['processed_img_wh'] = np.array([processed_image_shape]*joint3d_image.shape[0], dtype=np.int32)   # (N,2)

                processed_img_w, processed_img_h = processed_image_shape[0], processed_image_shape[1]
                if normalize == 'isotropic':
                    joint3d_image_affined_scale = np.concatenate([np.array([[processed_img_w / 2]]).repeat(joint3d_image.shape[0], axis=0), 
                                                                 np.array([[processed_img_w / 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                 img_ori_w / 2,                                                                 
                                                                 ], axis=-1) # (N,3)
                    joint3d_image_affined_transl =np.array([[1, processed_img_h / processed_img_w, 0]]).repeat(joint3d_image.shape[0], axis=0) # (N,3)
                elif normalize == 'anisotropic':
                    joint3d_image_affined_scale = np.concatenate([np.array([[processed_img_w // 2]]).repeat(joint3d_image.shape[0], axis=0), 
                                                                 np.array([[processed_img_h // 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                                 img_ori_w / 2,                                                                 
                                                                 ], axis=-1) # (N,3)
                    joint3d_image_affined_transl =np.array([[1, 1, 0]]).repeat(joint3d_image.shape[0], axis=0) # (N,3)
                else:
                    NotImplementedError
                data_dict[dt_file]['joint3d_image_affined_scale'] = joint3d_image_affined_scale
                data_dict[dt_file]['joint3d_image_affined_transl'] = joint3d_image_affined_transl

            ######################################################### Get split_id #########################################################
            split_id = datareader.get_split_id(designated_split=designated_split)   # 这里是用 unsplit_data 中的 'source' 来划分 split_id, 所以也要利用 valid_indices 作修改

            ######################################################### update data list #########################################################
            data_list.extend(zip([dt_file]*len(split_id), split_id, [use_image]*len(split_id), [None]*len(split_id)))

        self.data_dict = data_dict
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)        

    def __getitem__(self, idx):
        dt_file, slice_id, use_image, caption = self.data_list[idx]

        joint3d_image = self.data_dict[dt_file]['joint3d_image'][slice_id]  # (num_frames, 17, 3)
        factor_2_5d = self.data_dict[dt_file]['2.5d_factor'][slice_id]  # (num_frames,) only for test
        ori_img_wh = self.data_dict[dt_file]['ori_img_wh'][slice_id]  # (num_frames, 2). element: (res_w, res_h)
        # assert (img_ori_hw[0:1, :] == img_ori_hw[1:, :]).all()

        ############################################# normalize joint3d_image #############################################
        joint3d_image_scale = self.data_dict[dt_file]['joint3d_image_scale'][slice_id]
        joint3d_image_transl = self.data_dict[dt_file]['joint3d_image_transl'][slice_id]
        joint3d_image_normed = joint3d_image / joint3d_image_scale[..., None, :] - joint3d_image_transl[..., None, :]

        if use_image:
            ############################################# load image; do normalize #############################################
            image_sources = self.data_dict[dt_file]['image_sources'][slice_id]  # (num_frames,)
            processed_img_wh = self.data_dict[dt_file]['processed_img_wh'][slice_id]  # (num_frames, 2). element: (res_w, res_h)

            video_bgr = []
            for img_path in image_sources:
                assert os.path.exists(img_path), f'img_path={img_path} not exists.'
                image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                video_bgr.append(image_bgr)
            video_bgr = np.stack(video_bgr, axis=0)  # (num_frames, H, W, 3), BGR order

            video_rgb = video_bgr[..., ::-1]  # Convert BGR to RGB
            video_rgb = (video_rgb / 255.0 - self.img_mean) / self.img_std   # to [0,1], then normalize

            ############################################# load affined joint3d_image; do normalize #############################################
            joint3d_image_affined = self.data_dict[dt_file]['joint3d_image_affined'][slice_id]  # (num_frames, 17, 3)

            joint3d_image_affined_scale = self.data_dict[dt_file]['joint3d_image_affined_scale'][slice_id]  # (num_frames, 3)
            joint3d_image_affined_transl = self.data_dict[dt_file]['joint3d_image_affined_transl'][slice_id]
            joint3d_image_affined_normed = joint3d_image_affined / joint3d_image_affined_scale[..., None, :] - joint3d_image_affined_transl[..., None, :]

        return_dict = {}
        for get_item in self.get_item_list:
            item = locals()[get_item]
            try:
                item = torch.from_numpy(item).float()
            except:
                pass
            return_dict[get_item] = item
        # e.g., return_dict = (joint3d_image, joint3d_image_normed, factor_2_5d, joint3d_image_scale, joint3d_image_transl)
        # e.g., return_dict = (joint3d_image, joint3d_image_normed, factor_2_5d, joint3d_image_scale, joint3d_image_transl, 
        #                       video_rgb, joint3d_image_affined, joint3d_image_affined_normed, joint3d_image_affined_scale, joint3d_image_affined_transl)

        return return_dict


    @staticmethod
    def collate_fn(batch):
        return_dict = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                return_dict[k].append(v)
        try:
            for k, v in return_dict.items():
                return_dict[k] = torch.stack(v, dim=0)
        except:
            pass
        return return_dict



if __name__ == '__main__':
    dataset = Multimodal_Mocap_Dataset(processed_image_shape=(192,256),
                                       get_item_list=['joint3d_image', 'joint3d_image_normed', 'factor_2_5d', 'joint3d_image_scale', 'joint3d_image_transl', 
                                                      'video_rgb', 'joint3d_image_affined', 'joint3d_image_affined_normed', 'joint3d_image_affined_scale', 'joint3d_image_affined_transl']
                                       )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=Multimodal_Mocap_Dataset.collate_fn)
    for batch_dict in dataloader:
        pass
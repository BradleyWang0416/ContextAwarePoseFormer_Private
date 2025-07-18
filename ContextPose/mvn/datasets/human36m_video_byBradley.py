import os
import collections
from collections import defaultdict
import pickle

import numpy as np
np.set_printoptions(suppress=True)
import cv2

from torch.utils.data import Dataset

from mvn.models.loss import MPJPE, P_MPJPE, N_MPJPE, MPJVE
from mvn.utils.img import crop_image

from .human36m import Human36MMultiViewDataset, Human36MSingleViewDataset

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


class Human36MMultiViewDataset_MultiFrame(Human36MMultiViewDataset):
    """
        Human3.6M for multiview tasks.
    """
    def __init__(self, *args, frame=1, **kwrags):
        super().__init__(*args, frame=frame, **kwrags)

        self.frame = frame

        self.clip_indices = self.split_video()  # [range(0, 4), range(1, 5), ..., range(1559747, 1559751), range(1559748, 1559752)]

    def split_video(self):
        video_id_list = [label['video_id'] for label in self.labels]
        
        clip_indices = split_clips(video_id_list, self.frame, data_stride=self.frame, if_resample=True, randomness=False)

        return clip_indices
        
    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):

        clip_id = self.clip_indices[idx]    # e.g., range(0,4)

        video_clip = []
        joints_3d_clip = []
        joints_2d_cpn_clip = []
        joints_2d_cpn_crop_clip = []
        for shot_id in clip_id:
            image, joints_3d, joints_2d_cpn, joints_2d_cpn_crop = super(Human36MMultiViewDataset_MultiFrame, self).__getitem__(shot_id)
            # [256,192,3], [1,17,3], [17,2], [17,2]
            video_clip.append(image)
            joints_3d_clip.append(joints_3d)
            joints_2d_cpn_clip.append(joints_2d_cpn)
            joints_2d_cpn_crop_clip.append(joints_2d_cpn_crop)
        video_clip = np.stack(video_clip, axis=0)  # [T,256,192,3]
        joints_3d_clip = np.concatenate(joints_3d_clip, axis=0)  # [T,17,3]
        joints_2d_cpn_clip = np.stack(joints_2d_cpn_clip, axis=0)  # [T,17,2]
        joints_2d_cpn_crop_clip = np.stack(joints_2d_cpn_crop_clip, axis=0)  # [T,17,2]
        
        return video_clip, joints_3d_clip, joints_2d_cpn_clip, joints_2d_cpn_crop_clip


class Human36MSingleViewDataset_MultiFrame(Human36MMultiViewDataset_MultiFrame):
    def __init__(self,
                 root='/Vol1/dbstore/datasets/Human3.6M/processed/',
                 labels_path='/Vol1/dbstore/datasets/Human3.6M/extra/human36m-multiview-labels-SSDbboxes.npy',
                 pred_results_path=None,
                 image_shape=(192, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 with_damaged_actions=False,
                 cuboid_size=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 undistort_images=False,
                 ignore_cameras=[],
                 crop=True,
                 erase=False,
                 rank = None,
                 world_size = None,
                 data_format='',
                 frame=1
                 ):
        super(Human36MSingleViewDataset_MultiFrame, self).__init__(
            root=root,
            labels_path=labels_path,
            pred_results_path=pred_results_path,
            image_shape=image_shape,
            train=train,
            test=test,
            retain_every_n_frames_in_test=retain_every_n_frames_in_test,
            with_damaged_actions=with_damaged_actions,
            cuboid_size=cuboid_size,
            scale_bbox=scale_bbox,
            norm_image=norm_image,
            kind=kind,
            undistort_images=undistort_images,
            ignore_cameras=ignore_cameras,
            crop=crop,
            erase=erase,
            data_format=data_format,
            frame=frame,
        )

        self.pred_results_path = pred_results_path

        self.labels_action_idx = (np.array([label['action'] for label in self.labels])-2) * 2 + (np.array([label['subaction'] for label in self.labels])-1)     # size = (len(self.labels),)
        self.labels_subject_idx = np.array([retval['subject_names'].index('S'+str(label['subject'])) for label in self.labels])
        self.clips_action_idx = self.labels_action_idx[self.clip_indices][:,0]      # size = (len(self.clip_indices), frame) --> (len(self.clip_indices),). every clip performs the same action
        self.clips_subject_idx = self.labels_subject_idx[self.clip_indices][:,0]    # size = (len(self.clip_indices), frame) --> (len(self.clip_indices),). every clip comes from the same subject

        self.dist_size = self.prepare_labels(rank, world_size)
        self.video_idx = np.array([label['video_id'] for label in self.labels])

    def prepare_labels(self, rank, world_size):
        if rank is not None and world_size is not None:
            n = len(self.clip_indices) // world_size
            dist_size = [n if i < world_size - 1 else len(self.clip_indices) - n * (world_size - 1)\
                for i in range(world_size)]
            start = n * rank
            end = len(self.clip_indices) if rank == world_size - 1 else start + n
            self.clip_indices = self.clip_indices[start:end]
            if self.keypoints_3d_pred is not None:
                self.keypoints_3d_pred = self.keypoints_3d_pred[start:end]
            return dist_size
        
    def evaluate_using_pred(self, keypoints_gt, keypoints_3d_predicted):
        def evaluate_by_actions(self, keypoints_gt, keypoints_3d_predicted, mask=None):
            if mask is None:
                mask = np.ones(keypoints_gt.shape[0], dtype=bool)

            e1 = MPJPE(return_mean=True)
            e2 = P_MPJPE(return_mean=True)
            e3 = N_MPJPE(return_mean=True)
            ev = MPJVE(return_mean=True)

            action_scores = {}

            num_frames = keypoints_gt.shape[1]

            for action_idx in range(len(retval['action_names'])):
                action_mask = (self.clips_action_idx == action_idx) & mask  # len = number of clips
                clip_count = np.count_nonzero(action_mask)

                if clip_count == 0:
                    continue

                frame_count = num_frames * clip_count

                kp3d_pred = keypoints_3d_predicted[action_mask].reshape(-1, 17, 3)  # [NT,17,3]
                kp3d_gt = keypoints_gt[action_mask].reshape(-1, 17, 3)  # [NT,17,3]


                loss_3d_pos = frame_count * e1(kp3d_pred, kp3d_gt).item()
                loss_3d_pos_procrustes = frame_count * e2(kp3d_pred.cpu().numpy(), kp3d_gt.cpu().numpy())
                loss_3d_vel = frame_count * ev(kp3d_pred.cpu().numpy(), kp3d_gt.cpu().numpy())

                action_scores[retval['action_names'][action_idx]] = {
                    'MPJPE': loss_3d_pos,
                    'P_MPJPE': loss_3d_pos_procrustes,
                    'MPJVE': loss_3d_vel,
                    'frame_count': frame_count
                }

            action_names_without_trials = [name[:-2] for name in retval['action_names'] if name.endswith('-1')]

            for action_name_without_trial in action_names_without_trials:
                combined_score = {
                    'MPJPE': 0.0,
                    'P_MPJPE': 0.0,
                    # 'N_MPJPE': 0.0,
                    'MPJVE': 0.0,
                    'frame_count': 0,
                }

                for trial in 1, 2:
                    action_name = f"{action_name_without_trial}-{trial}"
                    if action_name not in action_scores:
                        continue
                    combined_score['MPJPE' ] += action_scores[action_name]['MPJPE']
                    combined_score['P_MPJPE' ] += action_scores[action_name]['P_MPJPE']
                    # combined_score['N_MPJPE' ] += action_scores[action_name]['N_MPJPE']
                    combined_score['MPJVE' ] += action_scores[action_name]['MPJVE']
                    combined_score['frame_count'] += action_scores[action_name]['frame_count']
                    del action_scores[action_name]

                if combined_score['frame_count'] > 0:
                    action_scores[action_name_without_trial] = combined_score

            for k in action_scores.keys():
                frame_count = action_scores[k]['frame_count']
                action_scores[k] = {
                    'MPJPE': action_scores[k]['MPJPE'] / frame_count,
                    'P_MPJPE': action_scores[k]['P_MPJPE'] / frame_count,
                    'MPJVE': action_scores[k]['MPJVE'] / frame_count,
                    # 'N_MPJPE': action_scores[k]['N_MPJPE'] / frame_count
                }

            return action_scores

        action_scores = evaluate_by_actions(self, keypoints_gt, keypoints_3d_predicted)

        return action_scores

    def evaluate(self, keypoints_gt, keypoints_3d_predicted, proj_matricies_batch=None, config=None,  split_by_subject=False, transfer_cmu_to_human36m=False, transfer_human36m_to_human36m=False):
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(f'`keypoints_3d_predicted` shape should be {keypoints_gt.shape}, got {keypoints_3d_predicted.shape}')

        result = self.evaluate_using_pred(keypoints_gt, keypoints_3d_predicted)

        return result



def split_clips (vid_list, n_frames, data_stride, if_resample=True, randomness=True):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set ()
    while i<len (vid_list):
        i += 1
        if i-st == n_frames:
            result.append (range (st,i))
            saved.add (vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len (vid_list):
            break
        if vid_list[i]!=vid_list[i-1]: 
            if not  (vid_list[i-1] in saved):
                if if_resample:
                    resampled = resample (i-st, n_frames, randomness=randomness) + st
                else:
                    resampled = range (st, i)
                result.append (resampled)
                saved.add (vid_list[i-1])
            st = i
    return result


def resample (ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint (ori_len-target_len)
            return range (st, st+target_len)  # Random clipping from sequence
        else:
            return np.array (range (target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace (0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor (even)
                high = np.ceil (even)
                sel = np.random.randint (2, size=even.shape)
                result = np.sort (sel*low+ (1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random (even.shape)*interval + even
            result = np.clip (result, a_min=0, a_max=ori_len-1).astype (np.uint32)
        else:
            result = np.linspace (0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

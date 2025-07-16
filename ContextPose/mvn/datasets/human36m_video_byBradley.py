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

from .human36m import Human36MMultiViewDataset

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
        
        clip_indices = split_clips(video_id_list, self.frame, data_stride=1, if_resample=True, randomness=False)

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

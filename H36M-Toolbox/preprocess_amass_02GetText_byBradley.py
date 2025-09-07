import sys
import os
import pickle
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm
import joblib
import codecs as cs
from collections import defaultdict

from viz_skel_seq import viz_skel_seq_anim

def main():
    amass_video_map = joblib.load('/data2/wxs/DATASETS/AMASS_ByBradley/video_map.pkl')

    raw_dir = '/data2/wxs/DATASETS/AMASS/'
    files = []
    target_fps_mb = 60  # motionbert
    target_fps_hm = 20  # humanml3d

    def traverse(f):
        fs = os.listdir(f)
        for f1 in fs:
            tmp_path = os.path.join(f,f1)
            # file
            if not os.path.isdir(tmp_path):
                files.append(tmp_path)
            # dir
            else:
                traverse(tmp_path)
    traverse(raw_dir)

    df = pd.read_csv("/home/wxs/HumanML3D/index.csv")
    humanml3d_info = {col: df[col].tolist() for col in df.columns}  # dict_keys(['source_path', 'start_frame', 'end_frame', 'new_name'])
    humanml3d_info['source_path'] = [path.replace('./pose_data/', '') for path in humanml3d_info['source_path']]

    max_len = 2916

    TEXT_MAP = {'train': defaultdict(dict), 'test': defaultdict(dict)}

    video_id = 0
    clip_cnt = 0
    clip_w_matching_caption_cnt = 0
    for file_path in sorted(files):
        try:
            x = dict(np.load(file_path))
            frame_indices_ori = np.arange(len(x['trans']))            
            fps = x['mocap_framerate']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        # HUMANML3D PART #######################################################################################
        sample_stride_hm = int(fps / target_fps_hm)
        frame_indices_hm = frame_indices_ori[::sample_stride_hm]

        if 'humanact12' not in file_path:
            if 'Eyes_Japan_Dataset' in file_path:
                frame_indices_hm = frame_indices_hm[3*target_fps_hm:]
            if 'MPI_HDM05' in file_path:
                frame_indices_hm = frame_indices_hm[3*target_fps_hm:]
            if 'TotalCapture' in file_path:
                frame_indices_hm = frame_indices_hm[1*target_fps_hm:]
            if 'MPI_Limits' in file_path:
                frame_indices_hm = frame_indices_hm[1*target_fps_hm:]
            if 'Transitions_mocap' in file_path:
                frame_indices_hm = frame_indices_hm[int(0.5*target_fps_hm):]

        if 'BioMotionLab' in file_path:
            data_split = 'test'
        else:
            data_split = 'train'

        humanml3d_key = file_path.split(raw_dir)[1].replace('.npz', '.npy')     # 'ACCAD/Female1General_c3d/A1 - Stand_poses.npz'
        row_indices = find_all_occurrences(humanml3d_key, humanml3d_info['source_path'])

        for row_id in row_indices:  # row_indices could be empty
            st_frame = humanml3d_info['start_frame'][row_id]
            ed_frame = humanml3d_info['end_frame'][row_id]
            humanml3d_npy_file = humanml3d_info['new_name'][row_id]
            valid_frame_indices_hm = frame_indices_hm[st_frame:ed_frame]
            caption_path = os.path.join('/data2/wxs/DATASETS/HumanML3D/HumanML3D/texts/', humanml3d_npy_file.replace('.npy', '.txt'))

            with cs.open(caption_path) as f:      # 'datasets/humanml3d/texts/000002.txt'
                lines = f.readlines()
            caption_list = []
            for line in lines:      # 循环txt文件每一行
                line_split = line.strip().split('#')    # ['a man full-body sideways jumps to his left.', 'a/DET man/NOUN fullbody/NOUN sideways/ADV jump/VERB to/ADP his/DET left/NOUN', '0.0', '0.0']
                caption = line_split[0]                 # 'a man full-body sideways jumps to his left.'
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag == 0.0 and to_tag == 0.0:      # this means the text is captioning the entire sequence of corresponding motion (see official github)
                    validCaption_frame_indices = valid_frame_indices_hm.copy() # humanml3d fps=20
                else:
                    validCaption_frame_indices = valid_frame_indices_hm[int(f_tag * 20):int(to_tag * 20)] # humanml3d fps=20
                if len(validCaption_frame_indices) == 0:
                    continue
                caption_list.append((caption, validCaption_frame_indices))


        # MOTIONBERT PART #######################################################################################
        sample_stride_mb = round(fps / target_fps_mb)
        frame_indices_mb = frame_indices_ori[::sample_stride_mb]

        time_length = len(frame_indices_mb)
        num_slice = time_length // max_len

        for sid in range(num_slice+1):  # 一个循环对应一个 slice (即一个 video)
            start = sid*max_len
            end = min((sid+1)*max_len, time_length)

            frame_indices_mb_slice = frame_indices_mb[start:end]

            clips_dict = amass_video_map[data_split][video_id]
            # {'vid21_camS9-60457274_frame0-282': range(16546, 16829), 'vid21_camS9-60457274_frame291-308': range(16829, 16847), 'vid21_camS9-60457274_frame426-997': range(16847, 17419)}
            for clip_key, global_indices in clips_dict.items():  # 一个循环对应一个 clip
                clip_st, clip_ed = clip_key.split('frame')[1].split('-')
                clip_st, clip_ed = int(clip_st), int(clip_ed)
                assert clip_ed < len(frame_indices_mb_slice)

                validImg_frame_indices = frame_indices_mb_slice[clip_st:clip_ed + 1]

                # Find out if each clip has matching humanml3d caption or not #######################################################################################
                matching_captions = []
                for caption, validCaption_frame_indices in caption_list:
                    if np.abs(validCaption_frame_indices.min() - validImg_frame_indices.min()) <= sample_stride_hm and \
                        np.abs(validCaption_frame_indices.max() - validImg_frame_indices.max()) <= sample_stride_hm:

                        matching_captions.append(caption)
                        

                if len(matching_captions) > 0:
                    clip_w_matching_caption_cnt += 1
                    TEXT_MAP[data_split][video_id][clip_key] = matching_captions

                clip_cnt += 1

            video_id += 1

    assert video_id == 11246

    print(f"Total video: {video_id}, total clip: {clip_cnt}, clip w matching caption: {clip_w_matching_caption_cnt}")
    # >>> Total video: 11246, total clip: 11910, clip w matching caption: 4571

    joblib.dump(TEXT_MAP, '/data2/wxs/DATASETS/AMASS_ByBradley/text_map.pkl')


def find_all_occurrences(target_string, string_list):
    """
    Finds all indices of a string in a list.
    Returns a list of indices. The list will be empty if not found.
    """
    indices = []
    for i, s in enumerate(string_list):
        if s == target_string:
            indices.append(i)
    return indices

if __name__ == "__main__":
    main()
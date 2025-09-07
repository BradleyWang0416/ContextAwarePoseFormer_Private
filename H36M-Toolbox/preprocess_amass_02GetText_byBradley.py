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

from viz_skel_seq import viz_skel_seq_anim

def main():
    raw_dir = '/data2/wxs/DATASETS/AMASS/'
    
    files = []
    length = 0
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

    VIDEO_TEXT_PAIR_INFO = {'train': {}, 'test': {}}

    video_id = 0
    for file_path in sorted(files):
        try:
            x = dict(np.load(file_path))
            frame_indices_ori = np.arange(len(x['trans']))            
            fps = x['mocap_framerate']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

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

        text_info = []
        for row_id in row_indices:  # row_indices could be empty
            st_frame = humanml3d_info['start_frame'][row_id]
            ed_frame = humanml3d_info['end_frame'][row_id]
            humanml3d_npy_file = humanml3d_info['new_name'][row_id]
            valid_frame_indices_hm = frame_indices_hm[st_frame:ed_frame]
            text = os.path.join('/data2/wxs/DATASETS/HumanML3D/HumanML3D/texts/', humanml3d_npy_file.replace('.npy', '.txt'))

            text_info.append((text, valid_frame_indices_hm))



    
        sample_stride_mb = round(fps / target_fps_mb)
        frame_indices_mb = frame_indices_ori[::sample_stride_mb]

        time_length = len(frame_indices_mb)
        num_slice = time_length // max_len

        for sid in range(num_slice+1):
            start = sid*max_len
            end = min((sid+1)*max_len, time_length)

            frame_indices_mb_subseq = frame_indices_mb[start:end]

            VIDEO_TEXT_PAIR_INFO[data_split][video_id] = {
                'file_path': file_path,
                'frame_indices_wrt_amass_npz': frame_indices_mb_subseq,
                'humanml3d': text_info,   # list of (text_file, valid_frame_indices_hm)
            }

            video_id += 1


    assert video_id == 11246

    joblib.dump(VIDEO_TEXT_PAIR_INFO, '/data2/wxs/DATASETS/AMASS_ByBradley/retrieved_text_from_humanml3d.pkl')


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
import joblib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pickle

from viz_skel_seq import viz_skel_seq_anim
from preprocess_amass_01GetPose_byBradley import _infer_box

def main():
    with open("/data2/wxs/DATASETS/h36m_name_map.json", "r") as f:
        get_real_source = json.load(f)
    with open("/data2/wxs/DATASETS/Human3.6M_MMPose/processed/annotation_body3d/cameras.pkl", 'rb') as f:
        cameras_dict = pickle.load(f)

    h36m_for_motionbert_path = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final_wImgPath.pkl"
    h36m_for_motionbert = joblib.load(h36m_for_motionbert_path)

    vid_subact_frameId = joblib.load("/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/globalID_per_video.pkl")

    joints_2d_cpn = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)
    joints_2d_cpn = joints_2d_cpn['positions_2d'].item()   # dict_keys(['S6', 'S7', 'S5', 'S11', 'S1', 'S9', 'S8', 'S2', 'S3', 'S4'])

    raw_data_root = "/data2/wxs/DATASETS/Human3.6M_ByBradley/extracted/"

    data_train_split = h36m_for_motionbert['train']
    data_train_size = len(data_train_split['source'])
    joint_3d_cam_train_all = np.zeros((data_train_size, 17, 3))
    joint_2d_cpn_train_all = np.zeros((data_train_size, 17, 2))
    for source_wo_sa in vid_subact_frameId['train'].keys():
        subject_id, act_id, cam_id = source_wo_sa.split('_')[1], source_wo_sa.split('_')[3], source_wo_sa.split('_')[5]
        for subact_id in ['01', '02']:
            global_indices = vid_subact_frameId['train'][source_wo_sa][subact_id]

            source_w_sa = f"s_{subject_id}_act_{act_id}_subact_{subact_id}_ca_{cam_id}"
            source_readable = get_real_source[source_w_sa]
            
            joint_3d_cam_path = os.path.join(raw_data_root, f'S{int(subject_id)}', 'D3_Positions_mono', source_readable+'.npy')
            joint_3d_cam = np.load(joint_3d_cam_path)
            assert joint_3d_cam.shape[0] == len(global_indices), f"{joint_3d_cam.shape[0]} vs {len(global_indices)}"
            joint_3d_cam_train_all[global_indices] = joint_3d_cam

            act_str = source_readable.split('.')[0]
            if "TakingPhoto" in act_str:
                act_str = act_str.replace("TakingPhoto", "Photo")
            elif "WalkingDog" in act_str:
                act_str = act_str.replace("WalkingDog", "WalkDog")
            joint_2d_cpn = joints_2d_cpn[f'S{int(subject_id)}'][act_str][int(cam_id) - 1]   # (N,17,2)
            if joint_3d_cam.shape[0] < joint_2d_cpn.shape[0]:
                print(f"{joint_3d_cam.shape[0]} vs {joint_2d_cpn.shape[0]}")
                joint_2d_cpn = joint_2d_cpn[:joint_3d_cam.shape[0]]
            assert joint_3d_cam.shape[0] == joint_2d_cpn.shape[0], f"{joint_3d_cam.shape[0]} vs {joint_2d_cpn.shape[0]}"
            joint_2d_cpn_train_all[global_indices] = joint_2d_cpn



    data_test_split = h36m_for_motionbert['test']
    data_test_size = len(data_test_split['source'])
    joint_3d_cam_test_all = np.zeros((data_test_size, 17, 3))
    joint_2d_cpn_test_all = np.zeros((data_test_size, 17, 2))
    for source_w_sa in vid_subact_frameId['test'].keys():        
        source_readable = get_real_source[source_w_sa]
        subject_id = source_w_sa.split('_')[1]
        cam_id = source_w_sa.split('_')[7]

        global_indices = vid_subact_frameId['test'][source_w_sa]

        joint_3d_cam_path = os.path.join(raw_data_root, f'S{int(subject_id)}', 'D3_Positions_mono', source_readable+'.npy')
        joint_3d_cam = np.load(joint_3d_cam_path)
        assert joint_3d_cam.shape[0] == len(global_indices), f"{joint_3d_cam.shape[0]} vs {len(global_indices)}"
        joint_3d_cam_test_all[global_indices] = joint_3d_cam

        act_str = source_readable.split('.')[0]
        if "TakingPhoto" in act_str:
            act_str = act_str.replace("TakingPhoto", "Photo")
        elif "WalkingDog" in act_str:
            act_str = act_str.replace("WalkingDog", "WalkDog")
        joint_2d_cpn = joints_2d_cpn[f'S{int(subject_id)}'][act_str][int(cam_id) - 1]   # (N,17,2)
        if joint_3d_cam.shape[0] < joint_2d_cpn.shape[0]:
            print(f"{joint_3d_cam.shape[0]} vs {joint_2d_cpn.shape[0]}")
            joint_2d_cpn = joint_2d_cpn[:joint_3d_cam.shape[0]]
        assert joint_3d_cam.shape[0] == joint_2d_cpn.shape[0], f"{joint_3d_cam.shape[0]} vs {joint_2d_cpn.shape[0]}"
        joint_2d_cpn_test_all[global_indices] = joint_2d_cpn



    assert joint_3d_cam_train_all.shape[0] == h36m_for_motionbert['train']['joint_2d'].shape[0]
    assert joint_3d_cam_test_all.shape[0] == h36m_for_motionbert['test']['joint_2d'].shape[0]
    h36m_for_motionbert['train']['joint_3d_cam'] = joint_3d_cam_train_all
    h36m_for_motionbert['test']['joint_3d_cam'] = joint_3d_cam_test_all

    assert joint_2d_cpn_train_all.shape == h36m_for_motionbert['train']['joint_2d'].shape
    assert joint_2d_cpn_test_all.shape == h36m_for_motionbert['test']['joint_2d'].shape
    h36m_for_motionbert['train']['joint_2d_cpn'] = joint_2d_cpn_train_all
    h36m_for_motionbert['test']['joint_2d_cpn'] = joint_2d_cpn_test_all



    joblib.dump(h36m_for_motionbert, '/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final_wImgPath_wJ3dCam_wJ2dCpn.pkl')





if __name__ == "__main__":
    main()
import joblib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pickle

from viz_skel_seq import viz_skel_seq_anim
from preprocess_amass_byBradley import _infer_box

def main():
    with open("/data2/wxs/DATASETS/h36m_name_map.json", "r") as f:
        get_real_source = json.load(f)
    with open("/data2/wxs/DATASETS/Human3.6M_MMPose/processed/annotation_body3d/cameras.pkl", 'rb') as f:
        cameras_dict = pickle.load(f)

    h36m_for_motionbert_path = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
    h36m_for_motionbert = joblib.load(h36m_for_motionbert_path)

    vid_subact_frameId = joblib.load("/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/globalID_per_video.pkl")

    raw_data_root = "/data2/wxs/DATASETS/Human3.6M_ByBradley/extracted/"

    data_train_split = h36m_for_motionbert['train']
    data_train_size = len(data_train_split['source'])
    bboxes_train_all = np.zeros((data_train_size, 4))
    for source_wo_sa in vid_subact_frameId['train'].keys():
        subject_id, act_id, cam_id = source_wo_sa.split('_')[1], source_wo_sa.split('_')[3], source_wo_sa.split('_')[5]
        for subact_id in ['01', '02']:
            global_indices = vid_subact_frameId['train'][source_wo_sa][subact_id]

            source_w_sa = f"s_{subject_id}_act_{act_id}_subact_{subact_id}_ca_{cam_id}"
            source_readable = get_real_source[source_w_sa]

            joint_3d_cam_path = os.path.join(raw_data_root, f'S{int(subject_id)}', 'D3_Positions_mono', source_readable+'.npy')
            joint_3d_cam = np.load(joint_3d_cam_path)

            assert joint_3d_cam.shape[0] == len(global_indices), f"{joint_3d_cam.shape[0]} vs {len(global_indices)}"

            camera_name = source_readable.split('.')[-1]
            camera = cameras_dict[(f'S{int(subject_id)}', camera_name)]

            center = camera['c']    # (2,1)
            focal = camera['f']      # (2,1)
            cx, cy = center[0, 0], center[1, 0]
            fx, fy = focal[0, 0], focal[1, 0]

            bboxes = _infer_box(joint_3d_cam, fx, fy, cx, cy, rootIdx=0) # (T,4). 单位: 像素
            centers = np.stack([0.5 * (bboxes[..., 0] + bboxes[..., 2]), 0.5 * (bboxes[..., 1] + bboxes[..., 3])], axis=-1) # (T,2)
            scales = np.stack([bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]], axis=-1) # (T,2)
            assert (bboxes_train_all[global_indices] == 0).all()
            bboxes_train_all[global_indices] = bboxes


    data_test_split = h36m_for_motionbert['test']
    data_test_size = len(data_test_split['source'])
    bboxes_test_all = np.zeros((data_test_size, 4))
    for source_w_sa in vid_subact_frameId['test'].keys():
        source_readable = get_real_source[source_w_sa]
        subject_id = source_w_sa.split('_')[1]

        global_indices = vid_subact_frameId['test'][source_w_sa]

        joint_3d_cam_path = os.path.join(raw_data_root, f'S{int(subject_id)}', 'D3_Positions_mono', source_readable+'.npy')
        joint_3d_cam = np.load(joint_3d_cam_path)

        assert joint_3d_cam.shape[0] == len(global_indices), f"{joint_3d_cam.shape[0]} vs {len(global_indices)}"

        camera_name = source_readable.split('.')[-1]
        camera = cameras_dict[(f'S{int(subject_id)}', camera_name)]

        center = camera['c']    # (2,1)
        focal = camera['f']      # (2,1)
        cx, cy = center[0, 0], center[1, 0]
        fx, fy = focal[0, 0], focal[1, 0]

        bboxes = _infer_box(joint_3d_cam, fx, fy, cx, cy, rootIdx=0) # (T,4). 单位: 像素
        centers = np.stack([0.5 * (bboxes[..., 0] + bboxes[..., 2]), 0.5 * (bboxes[..., 1] + bboxes[..., 3])], axis=-1) # (T,2)
        scales = np.stack([bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]], axis=-1) # (T,2)
        assert (bboxes_test_all[global_indices] == 0).all()
        bboxes_test_all[global_indices] = bboxes

    bboxes_all = {'train': bboxes_train_all, 'test': bboxes_test_all}
    joblib.dump(bboxes_all, '/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl')


if __name__ == "__main__":
    main()
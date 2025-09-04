import joblib
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from viz_skel_seq import viz_skel_seq_anim

def main():
    h36m_for_motionbert_path = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
    h36m_for_motionbert = joblib.load(h36m_for_motionbert_path)
            
    with open("/data2/wxs/DATASETS/h36m_name_map.json", "r") as f:
        get_real_source = json.load(f)

    image_root = "/data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/"

    vid_info = {'train': {}, 'test': {}}
    vid_frameId = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        data_split = h36m_for_motionbert[split]
        sources = data_split['source']
        for global_id in range(len(sources)):
            source = sources[global_id]
            if source not in vid_info[split]:
                vid_info[split][source] = 0
                vid_frameId[split][source] = []
            vid_info[split][source] += 1
            vid_frameId[split][source].append(global_id)

    """vid_info
    {'train':
        {'s_01_act_02_cam_01': 2995, 's_01_act_02_cam_02': 2995, 's_01_act_02_cam_03': 2995, 's_01_act_02_cam_04': 2995, 
        's_01_act_03_cam_01': 7657, 's_01_act_03_cam_02': 7657, 's_01_act_03_cam_03': 7657, 's_01_act_03_cam_04': 7657, 
        's_01_act_04_cam_01': 5078, 's_01_act_04_cam_02': 5078, 's_01_act_04_cam_03': 5078, 's_01_act_04_cam_04': 5078, 
        's_01_act_05_cam_01': 2414, 's_01_act_05_cam_02': 2414, 's_01_act_05_cam_03': 2414, 's_01_act_05_cam_04': 2414, ...}
    'test':
        {'s_09_act_02_subact_01_ca_01': 2356, 's_09_act_02_subact_01_ca_02': 2356, 's_09_act_02_subact_01_ca_03': 2356, 
        's_09_act_02_subact_01_ca_04': 2356, 's_09_act_02_subact_02_ca_01': 2699, 's_09_act_02_subact_02_ca_02': 2699, 
        's_09_act_02_subact_02_ca_03': 2699, 's_09_act_02_subact_02_ca_04': 2699, 's_09_act_03_subact_01_ca_01': 5873, 
        's_09_act_03_subact_01_ca_02': 5873, 's_09_act_03_subact_01_ca_03': 5873, 's_09_act_03_subact_01_ca_04': 5873, ...}
    """

    vid_subact_info = {'train': {}, 'test': {}}
    vid_subact_frameId = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        if split == 'train':
            for source_no_subact in vid_info[split].keys():
                source_tmp = source_no_subact.split('_')  # ['s', '01', 'act', '02', 'cam', '01']
                subject_id, act_id, cam_id = source_tmp[1], source_tmp[3], source_tmp[5]
                
                vid_subact_info[split][source_no_subact] = {}
                vid_subact_frameId[split][source_no_subact] = {}

                num_img_no_subact = 0
                for subact in ['01', '02']:
                    source_subact = f"s_{subject_id}_act_{act_id}_subact_{subact}_ca_{cam_id}"
                    source_full = get_real_source[source_subact]

                    img_folder = os.path.join(image_root, f'S{int(subject_id)}', f'S{int(subject_id)}_'+source_full.replace(' ', '_'))
                    img_list = sorted(os.listdir(img_folder))

                    num_img_subact = len(img_list)
                    img_ref = os.path.join(img_folder, img_list[0])
                    vid_subact_info[split][source_no_subact][subact] = (num_img_subact, img_ref)
                    vid_subact_frameId[split][source_no_subact][subact] = vid_frameId[split][source_no_subact][num_img_no_subact:num_img_no_subact+num_img_subact]
                    
                    assert (vid_frameId[split][source_no_subact][num_img_no_subact+num_img_subact-1]-vid_frameId[split][source_no_subact][num_img_no_subact]+1)==num_img_subact

                    num_img_no_subact += num_img_subact
                assert num_img_no_subact == vid_info[split][source_no_subact], f"{num_img_no_subact} vs {vid_info[split][source_no_subact]}"
        else:
            for source_subact in vid_info[split].keys():
                subject_id = source_subact.split('_')[1]  # '01'
                source_full = get_real_source[source_subact]
                img_folder = os.path.join(image_root, f'S{int(subject_id)}', f'S{int(subject_id)}_'+source_full.replace(' ', '_'))
                img_list = sorted(os.listdir(img_folder))

                num_img_subact = len(img_list)

                if num_img_subact != vid_info[split][source_subact]:    # 有问题的文件夹不记录到 vid_subact_info 和 vid_subact_frameId
                    print(f"{img_folder}: {num_img_subact} vs {vid_info[split][source_subact]}")
                
                else:
                    img_ref = os.path.join(img_folder, img_list[0])
                    vid_subact_info[split][source_subact] = (num_img_subact, img_ref)
                    vid_subact_frameId[split][source_subact] = vid_frameId[split][source_subact]
    
    """保存 vid_subact_frameId 到 Human3.6M for MotionBERT 文件夹中
    joblib.dump(vid_subact_frameId, "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/globalID_per_video.pkl")  # {'train': {'s_01_act_02_cam_01': {'01': [...], '02': [...]}, ...}, 'test': {'s_09_act_02_subact_01_ca_01': [...], ...}}
    """
    
    """以下是有问题的文件夹, 全部是测试集, 训练集没问题
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Greeting.54138969: 1447 vs 2711
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Greeting.55011271: 1447 vs 2711
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Greeting.58860488: 1447 vs 2711
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Greeting.60457274: 1447 vs 2711
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_SittingDown_1.54138969: 1554 vs 2932
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_SittingDown_1.55011271: 1554 vs 2932
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_SittingDown_1.58860488: 1554 vs 2932
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_SittingDown_1.60457274: 1554 vs 2932
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Waiting_1.54138969: 1612 vs 3312
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Waiting_1.55011271: 1612 vs 3312
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Waiting_1.58860488: 1612 vs 3312
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S9/S9_Waiting_1.60457274: 1612 vs 3312
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S11/S11_Directions.54138969: 1825 vs 1552
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S11/S11_Directions.55011271: 1825 vs 1552
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S11/S11_Directions.58860488: 1825 vs 1552
    /data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50/S11/S11_Directions.60457274: 1825 vs 1552
    """

    """将图片和2D姿态打印并保存, 检查是否对应正确
    for split in ['train', 'test']:
        data_split = h36m_for_motionbert[split]
        poses2d_all = data_split['joint_2d']

        if split == 'train':
            for source_no_subact in vid_subact_info[split].keys():
                for subact in vid_subact_info[split][source_no_subact]:
                    img_cnt, img_ref = vid_subact_info[split][source_no_subact][subact]
                    global_id = vid_subact_frameId[split][source_no_subact][subact][0]

                    fig_title = f"{source_no_subact}  sa{subact}"
                    check(img_ref, poses2d_all[global_id, ..., :2], fig_title, save=True)
        else:
            for source_subact in vid_subact_info[split].keys():
                img_cnt, img_ref = vid_subact_info[split][source_subact]
                global_id = vid_subact_frameId[split][source_subact][0]

                fig_title = source_subact
                check(img_ref, poses2d_all[global_id, ..., :2], fig_title, save=True)
    """
    
    """保存 image-level source 到 Human3.6M for MotionBERT 文件夹中
    source_images_all = {}
    for split in ['train', 'test']:
        data_split = h36m_for_motionbert[split]
        sources_all = data_split['source']
        
        source_img_all = [None] * len(sources_all)

        if split == 'train':
            for source_no_subact in vid_subact_info[split].keys():
                for subact in vid_subact_info[split][source_no_subact]:
                    img_cnt, img_ref = vid_subact_info[split][source_no_subact][subact]
                    assert img_ref[-10:] == '000001.jpg'
                    global_indices = vid_subact_frameId[split][source_no_subact][subact]
                    assert img_cnt == len(global_indices)

                    img_basename = img_ref[:-10]  # 去掉 '000001.jpg'
                    for global_id, local_id in zip(global_indices, range(img_cnt)):
                        source_img = f"{img_basename}{local_id+1:06d}.jpg"
                        assert source_img_all[global_id] is None
                        assert os.path.exists(source_img)
                        source_img_all[global_id] = source_img
            assert source_img_all.count(None) == 0           

        else:
            for source_subact in vid_subact_info[split].keys():
                img_cnt, img_ref = vid_subact_info[split][source_subact]
                assert img_ref[-10:] == '000001.jpg'
                global_indices = vid_subact_frameId[split][source_subact]
                assert img_cnt == len(global_indices)

                img_basename = img_ref[:-10]  # 去掉 '000001.jpg'
                for global_id, local_id in zip(global_indices, range(img_cnt)):
                    source_img = f"{img_basename}{local_id+1:06d}.jpg"
                    assert source_img_all[global_id] is None
                    assert os.path.exists(source_img)
                    source_img_all[global_id] = source_img        
            assert source_img_all.count(None) == 42028

        source_images_all[split] = source_img_all

    joblib.dump(source_images_all, "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl")  # {'train': [...], 'test': [...]}
    """



def check(img, pose2d, fig_title, save=False):
    img_data = plt.imread(img)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_data)
    ax.scatter(pose2d[:, 0], pose2d[:, 1], c='r', s=5)
    ax.axis('off')
    ax.set_title(fig_title)
    # ax.invert_yaxis()
    plt.tight_layout()
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(f"tmp/{fig_title}.png")
        plt.close()


if __name__ == "__main__":
    main()
import os
import numpy as np
import cdflib

joint_idx = [0, # Hips
                 1, 2, 3,       # RightUpLeg, RightLeg, RightFoot
                 6, 7, 8,       # LeftUpLeg, LeftLeg, LeftFoot
                 12, 16, 14, 15,# Spine1, LeftShoulder (same as 24: RightShoulder & 13: Neck), Head, Site
                 17, 18, 19,    # LeftArm, LeftForeArm, LeftHand
                 25, 26, 27     # RightArm, RightForeArm, RightHand
                 ]

for root, dirs, files in os.walk("/data2/wxs/DATASETS/Human3.6M_MMPose/extracted/"):
    for file in files:
        if file.endswith('.cdf'):
            file_path = os.path.join(root, file)
            data = cdflib.CDF(file_path)
            data_numpy = np.array(data.varget("Pose"))
            data_numpy = data_numpy.reshape(-1, 32, data_numpy.shape[-1]//32)
            data_numpy = data_numpy[:, joint_idx, :]
            npy_root = root.replace('MyPoseFeatures', '').replace('MMPose', 'ByBradley')
            if not os.path.exists(npy_root):
                os.makedirs(npy_root)
            npy_file = file.replace('.cdf', '.npy')
            npy_file_path = os.path.join(npy_root, npy_file)
            np.save(npy_file_path, data_numpy)
            print(f"Converted {file_path} to {npy_file_path}")
        
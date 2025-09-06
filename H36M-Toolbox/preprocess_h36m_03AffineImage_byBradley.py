import joblib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pickle
import cv2 
from tqdm import tqdm

from viz_skel_seq import viz_skel_seq_anim
from preprocess_amass_byBradley import _infer_box

output_image_shape = (192, 256) # (W,H)

def main():
    # h36m_for_motionbert = joblib.load("/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl")  
    bboxes_xyxy = joblib.load('/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl')
    image_sources = joblib.load("/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl")

    for split in ['train', 'test']:
        images_split = image_sources[split]
        bboxes_split = bboxes_xyxy[split]
        assert len(images_split) == len(bboxes_split)
        num_images = len(images_split)
        for global_id in tqdm(range(num_images)):
            bbox = bboxes_split[global_id]  # (4,)
            image_path = images_split[global_id]
            if (bbox == 0).all():
                assert image_path is None
                continue
            if image_path is None:
                assert (bbox == 0).all()
                continue
            assert (bbox != 0).all() and image_path is not None

            center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
            scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])

            image_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            trans = get_affine_transform(center, scale, 0, output_image_shape)
            new_image_numpy = cv2.warpAffine(image_numpy, trans, output_image_shape, flags=cv2.INTER_LINEAR)

            # plt.imshow(new_image_numpy[..., [2,1,0]])

            new_image_path = image_path.replace('images_fps50', 'images_fps50_cropped_192x256')
            if not os.path.exists(os.path.dirname(new_image_path)):
                os.makedirs(os.path.dirname(new_image_path))
            cv2.imwrite(new_image_path, new_image_numpy)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    center = np.array(center)
    scale = np.array(scale)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # rot_rad = np.pi * rot / 180

    # src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


if __name__ == "__main__":
    main()
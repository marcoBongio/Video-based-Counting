import os

import cv2
import h5py
import numpy as np
from PIL import Image

from variables import PATCH_SIZE_PF, HEIGHT, WIDTH, NUM_FRAMES


def load_data(img_path):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    img = Image.open(img_path).convert('RGB')
    img = img.resize((WIDTH, HEIGHT))

    prev_imgs = []
    post_imgs = []

    step = 1

    for i in range(NUM_FRAMES, 0, -step):
        prev_index = int(max(1, index - i - 1))
        prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
        prev_img = Image.open(prev_img_path).convert('RGB')
        prev_img = prev_img.resize((WIDTH, HEIGHT))
        prev_imgs.append(prev_img)

    # prev_imgs.append(img)

    # post_imgs.append(img)
    for i in range(0, NUM_FRAMES, step):
        post_index = int(min(150, index + i))
        post_img_path = os.path.join(img_folder, '%03d.jpg' % (post_index))
        post_img = Image.open(post_img_path).convert('RGB')
        post_img = post_img.resize((WIDTH, HEIGHT))
        post_imgs.append(post_img)

    gt_path = img_path.replace('.jpg', '_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
                        interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)


    last_prev_index = int(max(1, index - (NUM_FRAMES - 1)))
    last_prev_img_path = os.path.join(img_folder, '%03d.jpg' % last_prev_index)

    prev_gt_path = last_prev_img_path.replace('.jpg', '_resize.h5')

    prev_gt_file = h5py.File(prev_gt_path)
    prev_target = np.asarray(prev_gt_file['density'])
    prev_gt_file.close()
    prev_target = cv2.resize(prev_target,
                             (int(prev_target.shape[1] / PATCH_SIZE_PF), int(prev_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    last_post_index = int(min(150, index + (NUM_FRAMES - 1)))
    last_post_img_path = os.path.join(img_folder, '%03d.jpg' % last_post_index)

    post_gt_path = last_post_img_path.replace('.jpg', '_resize.h5')

    post_gt_file = h5py.File(post_gt_path)
    post_target = np.asarray(post_gt_file['density'])
    post_gt_file.close()
    post_target = cv2.resize(post_target,
                             (int(post_target.shape[1] / PATCH_SIZE_PF), int(post_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    return prev_imgs, img, post_imgs, prev_target, target, post_target

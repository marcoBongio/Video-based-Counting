import math
import os
from PIL import Image
import numpy as np
import h5py
import cv2

from variables import PATCH_SIZE_PF, HEIGHT, WIDTH, NUM_FRAMES


def load_data(img_path, train=True):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    # print(img_path)
    index = int(img_name.split('.')[0])

    prev_imgs = []
    post_imgs = []

    step = math.ceil(5 / (NUM_FRAMES - 1))

    for i in range(5, 0, -step):
        prev_index = int(max(1, index - i))
        prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
        # print(prev_img_path)
        prev_img = Image.open(prev_img_path).convert('RGB')
        prev_img = prev_img.resize((WIDTH, HEIGHT))
        prev_imgs.append(prev_img)

    for i in range(5, 0, -step):
        post_index = int(min(150, index + i))
        post_img_path = os.path.join(img_folder, '%03d.jpg' % (post_index))
        # print(post_img_path)
        post_img = Image.open(post_img_path).convert('RGB')
        post_img = post_img.resize((WIDTH, HEIGHT))
        post_imgs.insert(0, post_img)

    gt_path = img_path.replace('.jpg', '_resize.h5')
    img = Image.open(img_path).convert('RGB')
    img = img.resize((WIDTH, HEIGHT))

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
                        interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
    # print(np.sum(target))

    last_prev_index = int(max(1, index - 5))
    last_prev_img_path = os.path.join(img_folder, '%03d.jpg' % (last_prev_index))
    # print(last_prev_img_path)
    prev_gt_path = last_prev_img_path.replace('.jpg', '_resize.h5')
    # print(prev_gt_path)
    prev_gt_file = h5py.File(prev_gt_path)
    prev_target = np.asarray(prev_gt_file['density'])
    prev_gt_file.close()
    prev_target = cv2.resize(prev_target,
                             (int(prev_target.shape[1] / PATCH_SIZE_PF), int(prev_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
    # print(np.sum(prev_target))

    last_post_index = int(min(150, index + 5))
    last_post_img_path = os.path.join(img_folder, '%03d.jpg' % (last_post_index))
    # print(last_post_img_path)
    post_gt_path = last_post_img_path.replace('.jpg', '_resize.h5')
    # print(post_gt_path)
    post_gt_file = h5py.File(post_gt_path)
    post_target = np.asarray(post_gt_file['density'])
    post_gt_file.close()
    post_target = cv2.resize(post_target,
                             (int(post_target.shape[1] / PATCH_SIZE_PF), int(post_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)
    # print(np.sum(post_target))
    return prev_imgs, img, post_imgs, prev_target, target, post_target

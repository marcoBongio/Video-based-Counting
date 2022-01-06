import math
import os

import scipy.io
import skimage
import torch
from PIL import Image
import numpy as np
import h5py
import cv2
from torchvision import transforms

from variables import PATCH_SIZE_PF, HEIGHT, WIDTH, NUM_FRAMES

transform = transforms.Compose([
    transforms.ToTensor()
])


def load_data(img_path):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    # print(img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(max(4896000060, index - 60))
    post_index = int(min(4896004860, index + 60))

    prev_index = str(prev_index)[:4] + "_" + str(prev_index)[4:]
    prev_img_path = os.path.join(img_folder, prev_index + '.jpg')

    post_index = str(post_index)[:4] + "_" + str(post_index)[4:]
    post_img_path = os.path.join(img_folder, post_index + '.jpg')

    prev_gt_path = prev_img_path.replace('.jpg', '_resize.h5')
    gt_path = img_path.replace('.jpg', '_resize.h5')
    post_gt_path = post_img_path.replace('.jpg', '_resize.h5')

    prev_mat_path = prev_img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
    prev_mat = scipy.io.loadmat(prev_mat_path)
    prev_roi = skimage.transform.resize(prev_mat['roi'], (HEIGHT, WIDTH), order=0)

    mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
    mat = scipy.io.loadmat(mat_path)
    roi = skimage.transform.resize(mat['roi'], (HEIGHT, WIDTH), order=0)

    post_mat_path = post_img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
    post_mat = scipy.io.loadmat(post_mat_path)
    post_roi = skimage.transform.resize(post_mat['roi'], (HEIGHT, WIDTH), order=0)

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    prev_img = prev_img.resize((WIDTH, HEIGHT))
    img = img.resize((WIDTH, HEIGHT))
    post_img = post_img.resize((WIDTH, HEIGHT))

    prev_img = np.array(prev_img)
    prev_img = transform(prev_img).cuda()
    prev_img = prev_img * torch.FloatTensor(prev_roi).cuda()

    img = np.array(img)
    img = transform(img).cuda()
    img = img * torch.FloatTensor(roi).cuda()

    post_img = np.array(post_img)
    post_img = transform(post_img).cuda()
    post_img = post_img * torch.FloatTensor(post_roi).cuda()

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
                        interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    prev_gt_file = h5py.File(prev_gt_path)
    prev_target = np.asarray(prev_gt_file['density'])
    prev_gt_file.close()
    prev_target = cv2.resize(prev_target,
                             (int(prev_target.shape[1] / PATCH_SIZE_PF), int(prev_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    post_gt_file = h5py.File(post_gt_path)
    post_target = np.asarray(post_gt_file['density'])
    post_gt_file.close()
    post_target = cv2.resize(post_target,
                             (int(post_target.shape[1] / PATCH_SIZE_PF), int(post_target.shape[0] / PATCH_SIZE_PF)),
                             interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    return prev_img, img, post_img, prev_target, target, post_target

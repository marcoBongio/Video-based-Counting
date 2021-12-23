import math

import h5py
import PIL.Image as Image
import numpy
import numpy as np
import os
import glob

import scipy.io
import skimage
import torch
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.ndimage.filters import gaussian_filter
import json

from skimage import transform
from skimage.transform import warp

from image import *
from variables import HEIGHT, WIDTH
import joint
import pose

# x = torch.load('../../JTA-Dataset/poses/test/seq_256/1.data')

# set the root to the path of Venice dataset you download
root = '../../JTA-Dataset/poses/'

# now generate the Venice's ground truth
train_folder = os.path.join(root, 'train/')
test_folder = os.path.join(root, 'test/')
val_folder = os.path.join(root, 'val/')

path_sets = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if
             os.path.isdir(os.path.join(train_folder, f))] + [os.path.join(test_folder, f) for f in
                                                              os.listdir(test_folder) if
                                                              os.path.isdir(os.path.join(test_folder, f))] + [
                os.path.join(val_folder, f) for f in
                os.listdir(val_folder) if
                os.path.isdir(os.path.join(val_folder, f))]

pose_paths = []
for path in path_sets:
    for pose_path in glob.glob(os.path.join(path, '*.data')):
        pose_paths.append(pose_path)

for pose_path in pose_paths:
    print(pose_path)
    gt_list = torch.load(pose_path)

    img_path = pose_path.replace('poses', 'frames').replace('.data', '.jpg')
    img = plt.imread(img_path)

    # plt.imshow(img)
    # plt.show()

    k = np.zeros((HEIGHT, WIDTH))
    rate_h = img.shape[0] / HEIGHT
    rate_w = img.shape[1] / WIDTH
    for i in range(0, len(gt_list)):
        if gt_list[i][0].visible and gt_list[i][0].pos2d[1] >= 0 and gt_list[i][0].pos2d[0] >= 0:
            y_anno = min(int(gt_list[i][0].pos2d[1] / rate_h), HEIGHT - 1)
            x_anno = min(int(gt_list[i][0].pos2d[0] / rate_w), WIDTH - 1)
            k[y_anno, x_anno] = 1
    k = gaussian_filter(k, 3)

    # plt.imshow(k)
    # plt.show()

    with h5py.File(img_path.replace('.jpg', '_resize.h5'), 'w') as hf:
        hf['density'] = k
        hf.close()

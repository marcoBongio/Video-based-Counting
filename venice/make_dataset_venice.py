import math

import h5py
import PIL.Image as Image
import numpy
import numpy as np
import os
import glob

import scipy.io
import skimage
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.ndimage.filters import gaussian_filter
import json

from skimage import transform
from skimage.transform import warp

from image import *
from variables import HEIGHT, WIDTH

# set the root to the path of Venice dataset you download
root = '../venice/venice/'

# now generate the Venice's ground truth
train_folder = os.path.join(root, 'train_data/images')
test_folder = os.path.join(root, 'test_data/images')
path_sets = [train_folder, test_folder]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    gt = scipy.io.loadmat(
        img_path.replace('.jpg', '.mat').replace('images', 'ground-truth'))
    img = plt.imread(img_path)

    anno_list = gt['annotation']
    roi = skimage.transform.resize(gt['roi'], (HEIGHT, WIDTH), order=0)
    matrix = gt['homograph']

    """max_x = 0
    max_y = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            py = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / (
                        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2])
            if py > max_y:
                max_y = py

            px = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / (
                        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2])
            if px > max_x:
                max_x = px

    img = transform.warp(img, inv(matrix), output_shape=(math.ceil(max_y), math.ceil(max_x)))

    plt.imshow(img)
    plt.show()

    i = 0
    for anno in anno_list:
        anno_list[i][1] = (matrix[1][0] * anno[0] + matrix[1][1] * anno[1] + matrix[1][2]) / (
                    matrix[2][0] * anno[0] + matrix[2][1] * anno[1] + matrix[2][2])
        anno_list[i][0] = (matrix[0][0] * anno[0] + matrix[0][1] * anno[1] + matrix[0][2]) / (
                    matrix[2][0] * anno[0] + matrix[2][1] * anno[1] + matrix[2][2])
        i += 1"""

    k = np.zeros((HEIGHT, WIDTH))
    rate_h = img.shape[0] / HEIGHT
    rate_w = img.shape[1] / WIDTH
    for i in range(0, len(anno_list)):
        y_anno = min(int(anno_list[i][1] / rate_h), HEIGHT - 1)
        x_anno = min(int(anno_list[i][0] / rate_w), WIDTH - 1)
        k[y_anno, x_anno] = 1
    k = gaussian_filter(k, 3)

    # plt.imshow(k)
    # plt.show()

    k *= roi

    # plt.imshow(k)
    # plt.show()

    with h5py.File(img_path.replace('.jpg', '_resize.h5'), 'w') as hf:
        hf['density'] = k
        hf.close()

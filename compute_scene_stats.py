import json
import os

import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import dataset

import scipy.stats as st

with open("val.json", 'r+') as outfile:
    img_paths = json.load(outfile)

data_loader = torch.utils.data.DataLoader(dataset.listDataset(img_paths, shuffle=False,
                                                              batch_size=1,
                                                              num_workers=0,
                                                              transform=transforms.ToTensor()))

gt = []
mean = 0.0
var = 0.0
for i in range(len(img_paths)):
    img_path = img_paths[i]
    # print(img_path)
    # Rearrange batch to be the shape of [B, C, W * H]
    gt_path = img_path.replace('.jpg', '_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    # Compute mean and std here
    gt.append(np.sum(target))

    if (i+1) % 150 == 0:
        print("SCENE " + os.path.basename(os.path.dirname(img_path)))

        mean = np.mean(gt)
        var = np.var(gt)
        std = np.sqrt(var)

        print("Mean = " + str(mean))
        print("Standard deviation = " + str(std))
        print("Minimum = " + str(np.min(gt)))
        print("Maximum = " + str(np.max(gt)) + "\n")

        gt = []
        mean = 0.0
        var = 0.0
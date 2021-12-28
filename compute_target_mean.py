import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
from torchvision.transforms import transforms

import dataset

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
    if i % 250 == 0:
        print(str(i) + "/" + str(len(img_paths)))

    img_path = img_paths[i]
    # Rearrange batch to be the shape of [B, C, W * H]
    gt_path = img_path.replace('.jpg', '_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    # Compute mean and std here
    gt.append(np.sum(target))

mean = np.mean(gt)
var = np.var(gt)
std = np.sqrt(var)

conf_int = st.t.interval(0.95, len(gt) - 1, loc=mean, scale=st.sem(gt))

print("Mean = " + str(mean))
print("Standard deviation = " + str(std))
print("Minimum = " + str(np.min(gt)))
print("Maximum = " + str(np.max(gt)))

plt.hist(gt)
plt.show()

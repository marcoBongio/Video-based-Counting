import json

import cv2
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms

import dataset

with open("train_all.json", 'r+') as outfile:
    list = json.load(outfile)

"""with open("test.json", 'r+') as outfile:
    list.extend(json.load(outfile))"""

data_loader = torch.utils.data.DataLoader(dataset.listDataset(list, shuffle=False,
                                                              batch_size=1,
                                                              num_workers=0,
                                                              transform=transforms.ToTensor()))

nimages = 0
mean = 0.0
var = 0.0
for i_batch, batch_target in enumerate(data_loader):
    if nimages % 250 == 0:
        print(str(nimages) + "/" + str(len(data_loader.dataset)))
    batch = batch_target[0][0]
    # Rearrange batch to be the shape of [B, C, W * H]
    #print(batch.shape)
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0)
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print("Mean = " + str(mean))
print("Standard deviation = " + str(std))

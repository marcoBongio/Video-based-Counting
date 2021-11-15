import csv
import math

import h5py
import json
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from matplotlib import pyplot as plt

from image import *
from model import CANNet2s
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import cv2
import scipy.stats as st

from torchvision import transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error
from variables import HEIGHT, WIDTH, MODEL_NAME

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.4846, 0.4558, 0.4324],
                                                std=[0.2181, 0.2136, 0.2074]),
])

# the json file contains path of test images
test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model = CANNet2s()

model = model.cuda()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load("models/model_best_" + MODEL_NAME + '.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = []
gt = []
errs = []

with torch.no_grad():
    for i in range(len(img_paths)):
        if i % 250 == 0:
            print(str(i) + "/" + str(len(img_paths)))

        img_path = img_paths[i]
        img_folder = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        index = int(img_name.split('.')[0])

        prev_index = int(max(1, index - 5))

        prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
        # prev_img_path = 'test_data/00/30.jpg'
        # print(prev_img_path)
        prev_img = Image.open(prev_img_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        prev_img = prev_img.resize((WIDTH, HEIGHT))
        img = img.resize((WIDTH, HEIGHT))

        prev_img = transform(prev_img).cuda()
        img = transform(img).cuda()

        gt_path = img_path.replace('.jpg', '_resize.h5')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        # print(np.sum(target))
        # target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
        #                    interpolation=cv2.INTER_CUBIC) * 64

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        img = img.unsqueeze(0)
        prev_img = prev_img.unsqueeze(0)

        prev_flow = model(prev_img, img)

        prev_flow_inverse = model(img, prev_img)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :],
                                                                                        (0, 0, 0, 1)) + F.pad(
            prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :,
                                                                                                    :] + F.pad(
            prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :,
                                                                                              :] * mask_boundry

        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).data.cpu().numpy()
        target = target

        pred_sum = overall.sum()
        #print("PRED = " + str(pred_sum))
        pred.append(pred_sum)
        gt.append(np.sum(target))
        #print("GT = " + str(np.sum(target)))
        errs.append(abs(np.sum(target) - pred_sum))

mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))

print('MAE: ', mae)
print('RMSE: ', rmse)

results = zip(errs, gt, pred)

header = ["Error", "GT", "Prediction"]

with open("results/model_best_" + MODEL_NAME + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in results:
        writer.writerow(row)


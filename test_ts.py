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
from model import TSCANNet2s
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import cv2
import scipy.stats as st

from torchvision import transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error
from variables import HEIGHT, WIDTH, MODEL_NAME, MEAN, STD, NUM_FRAMES

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

# the json file contains path of test images
test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model = TSCANNet2s()

model = model.cuda()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load("models/model_best_" + MODEL_NAME + '.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = []
gt = []
errs = []
game = 0

for i in range(len(img_paths)):
    img_path = img_paths[i]
    print(str(i) + "/" + str(len(img_paths)))
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    img = Image.open(img_path).convert('RGB')
    img = img.resize((WIDTH, HEIGHT))

    gt_path = img_path.replace('.jpg', '_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    prev_imgs = []

    step = 1 # math.ceil(5 / (NUM_FRAMES - 1))

    for s in range(NUM_FRAMES - 1, 0, -step):
        prev_index = int(max(1, index - s))
        prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
        # print(prev_img_path)
        prev_img = Image.open(prev_img_path).convert('RGB')
        prev_img = prev_img.resize((WIDTH, HEIGHT))
        prev_imgs.append(prev_img)

    prev_imgs.append(img)
    prev_imgs = [transform(_prev_img).cuda() for _prev_img in prev_imgs]
    prev_imgs = [_prev_img.cuda() for _prev_img in prev_imgs]
    prev_imgs = [Variable(_prev_img) for _prev_img in prev_imgs]
    prev_imgs = [_prev_img.unsqueeze(0) for _prev_img in prev_imgs]
    prev_imgs = torch.stack(prev_imgs)

    with torch.no_grad():
        prev_flow = model(prev_imgs)
        prev_flow_inverse = model(prev_imgs, inverse=True)

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
    print("PRED = " + str(pred_sum))
    pred.append(pred_sum)
    gt.append(np.sum(target))
    print("GT = " + str(np.sum(target)))
    errs.append(abs(np.sum(target) - pred_sum))

    target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
                        interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    for k in range(target.shape[0]):
        for j in range(target.shape[1]):
            game += abs(overall[k][j] - target[k][j])

    print('MAE: ', mean_absolute_error(pred, gt))
    print('RMSE: ', np.sqrt(mean_squared_error(pred, gt)))
    print("GAME: " + str(game / (i + 1)) + "\n")

mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))
game = game / len(pred)

print("FINAL RESULT")
print('MAE: ', mae)
print('RMSE: ', rmse)
print('GAME: ', game)

results = zip(errs, gt, pred)

header = ["Error", "GT", "Prediction"]

try:
    os.mkdir(os.path.dirname("results/"))
except:
    pass

with open("results/model_best_" + MODEL_NAME + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in results:
        writer.writerow(row)

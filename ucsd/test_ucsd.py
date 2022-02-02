import csv
import json

import cv2
import scipy.io
import skimage
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.autograd import Variable
from torchinfo import summary
from torchvision import transforms

from image import *
from model import SACANNet2s, CANNet2s
from variables import HEIGHT, WIDTH, MODEL_NAME, MEAN, STD
from variables import PATCH_SIZE_PF

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

# the json file contains path of test images
test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model = CANNet2s()

model = model.cuda()

summary(model, input_size=((1, 3, HEIGHT, WIDTH), (1, 3, HEIGHT, WIDTH)))

# modify the path of saved checkpoint if necessary
checkpoint = torch.load("../models/model_best_" + MODEL_NAME + '.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'], strict=True)

model.eval()

pred = []
gt = []
errs = []
game = 0
root = '../../ucsdpeds/'

for i in range(len(img_paths)):
    img_path = img_paths[i]
    print(str(i) + "/" + str(len(img_paths)))
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)

    index = img_name.split('.')[0]
    index = int(index.split('f')[-1])

    prev_index = int(max(1, index - 5))
    base_name = img_folder.split('/')[-1]
    base_name = base_name.split('.y')[0]
    prev_img_path = os.path.join(img_folder, base_name + '_f%03d.png' % (prev_index))

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')

    prev_img = prev_img.resize((WIDTH, HEIGHT))
    img = img.resize((WIDTH, HEIGHT))

    prev_img = transform(prev_img).cuda()
    img = transform(img).cuda()

    roi_path = os.path.join(root, 'vidf-cvpr/vidf1_33_roi_mainwalkway.mat')
    roi = scipy.io.loadmat(roi_path)['roi'][0]
    roi = roi.item(0)[-1]
    roi = skimage.transform.resize(roi, (HEIGHT, WIDTH), order=0)

    prev_img = prev_img * torch.FloatTensor(roi).cuda()
    img = img * torch.FloatTensor(roi).cuda()

    gt_path = img_path.replace('.png', '_resize.h5')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    prev_img = prev_img.cuda()
    prev_img = Variable(prev_img)

    img = img.cuda()
    img = Variable(img)

    img = img.unsqueeze(0)
    prev_img = prev_img.unsqueeze(0)

    with torch.no_grad():
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

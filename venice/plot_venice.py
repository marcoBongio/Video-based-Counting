import json

import cv2
import scipy.io
import skimage
import torch
import torch.nn.functional as F
from matplotlib import cm
from skimage.transform import warp
from torch.autograd import Variable
from torchinfo import summary
from torchvision import transforms

from image import *
from model import SACANNet2s
from variables import HEIGHT, WIDTH, PATCH_SIZE_PF, MODEL_NAME, MEAN, STD


def plotDensity(density, plot_path):
    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density = density * 255.0

    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]

    try:
        os.mkdir(os.path.dirname(plot_path))
    except:
        pass

    cv2.imwrite(plot_path, new_map * 255)


transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

# the json file contains path of test images
test_json_path = 'test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

# the folder to output density map and flow maps
output_folder = os.path.join('plot', MODEL_NAME)

model = SACANNet2s()

model = model.cuda()

summary(model, input_size=((1, 3, HEIGHT, WIDTH), (1, 3, HEIGHT, WIDTH)))

# modify the path of saved checkpoint if necessary
checkpoint = torch.load('models/model_best_' + MODEL_NAME + '.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = []
gt = []

try:
    os.mkdir(os.path.dirname('plot/'))
except:
    pass

try:
    os.mkdir(os.path.dirname(os.path.join('plot', MODEL_NAME + '/')))
except:
    pass

for i in range(len(img_paths)):
    img_path = img_paths[i]

    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(index - 60)
    prev_index = str(prev_index)[:4] + "_" + str(prev_index)[4:]

    prev_img_path = os.path.join(img_folder, prev_index + '.jpg')

    img = Image.open(img_path).convert('RGB')

    mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
    mat = scipy.io.loadmat(mat_path)
    roi = skimage.transform.resize(mat['roi'], (HEIGHT, WIDTH), order=0)
    hom = mat['homograph']

    try:
        prev_img = Image.open(prev_img_path).convert('RGB')
        mat_path = prev_img_path.replace('.jpg', '.mat').replace('images', 'ground-truth')
        prev_mat = scipy.io.loadmat(mat_path)
        prev_roi = skimage.transform.resize(prev_mat['roi'], (HEIGHT, WIDTH), order=0)
    except:
        prev_img = img
        prev_roi = roi

    prev_img = prev_img.resize((WIDTH, HEIGHT))
    img = img.resize((WIDTH, HEIGHT))

    prev_img = np.array(prev_img)
    img = np.array(img)

    prev_img = transform(prev_img).cuda()
    img = transform(img).cuda()

    prev_img = prev_img * torch.FloatTensor(prev_roi).cuda()
    img = img * torch.FloatTensor(roi).cuda()

    gt_path = img_path.replace('.jpg', '_resize.h5')
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

    base_name = os.path.basename(img_path)
    print(base_name)
    folder_name = os.path.dirname(img_path).split('/')[-1]
    print(folder_name)
    gt_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_gt.jpg')
    print(gt_path)
    density_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_pred.jpg')
    """flow_1_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_1.jpg')
    flow_2_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_2.jpg')
    flow_3_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_3.jpg')
    flow_4_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_4.jpg')
    flow_5_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_5.jpg')
    flow_6_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_6.jpg')
    flow_7_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_7.jpg')
    flow_8_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_8.jpg')
    flow_9_path = os.path.join(output_folder, folder_name, base_name).replace('.jpg', '_flow_9.jpg')"""

    pred = cv2.resize(overall, (overall.shape[1] * PATCH_SIZE_PF, overall.shape[0] * PATCH_SIZE_PF),
                      interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    prev_flow = prev_flow.data.cpu().numpy()[0]
    """flow_1 = cv2.resize(prev_flow[0], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_2 = cv2.resize(prev_flow[1], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_3 = cv2.resize(prev_flow[2], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_4 = cv2.resize(prev_flow[3], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_5 = cv2.resize(prev_flow[4], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_6 = cv2.resize(prev_flow[5], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_7 = cv2.resize(prev_flow[6], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_8 = cv2.resize(prev_flow[7], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
    flow_9 = cv2.resize(prev_flow[8], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)"""

    plotDensity(pred, density_path)
    plotDensity(target, gt_path)
    """plotDensity(flow_1, flow_1_path)
    plotDensity(flow_2, flow_2_path)
    plotDensity(flow_3, flow_3_path)
    plotDensity(flow_4, flow_4_path)
    plotDensity(flow_5, flow_5_path)
    plotDensity(flow_6, flow_6_path)
    plotDensity(flow_7, flow_7_path)
    plotDensity(flow_8, flow_8_path)
    plotDensity(flow_9, flow_9_path)"""

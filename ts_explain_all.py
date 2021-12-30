import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2

from model import TSCANNet2s, FBTSCANNet2s
from ts_rollout import TSAttentionRollout
from ts_grad_rollout import TSAttentionGradRollout
from variables import MODEL_NAME, WIDTH, HEIGHT, MEAN, STD, NUM_FRAMES


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    head_fusion = 'mean'
    discard_ratio = 0.9
    category_index = None
    # json file contains the test images
    test_json_path = './test.json'

    with open(test_json_path, 'r') as outfile:
        img_paths = json.load(outfile)
    model = FBTSCANNet2s()

    model = model.cuda()

    # modify the path of saved checkpoint if necessary
    checkpoint = torch.load("models/model_best_" + MODEL_NAME + '.pth.tar', map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()

    model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    for i in range(0, len(img_paths)):
        img_path = img_paths[i]
        print(img_path)
        print(str(i) + "/" + str(len(img_paths)))
        img_folder = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        index = int(img_name.split('.')[0])
        img = Image.open(img_path).convert('RGB')

        prev_imgs = []

        step = 1

        for s in range(NUM_FRAMES - 1, 0, -step):
            prev_index = int(max(1, index - s))
            prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))
            # print(prev_img_path)
            prev_img = Image.open(prev_img_path).convert('RGB')
            prev_imgs.append(prev_img)

        prev_imgs.append(img)
        prev_imgs = [transform(_prev_img).cuda() for _prev_img in prev_imgs]
        prev_imgs = [_prev_img.cuda() for _prev_img in prev_imgs]
        prev_imgs = [Variable(_prev_img) for _prev_img in prev_imgs]
        prev_imgs = [_prev_img.unsqueeze(0) for _prev_img in prev_imgs]
        prev_imgs = torch.stack(prev_imgs)

        if category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = TSAttentionRollout(model, head_fusion=head_fusion,
                                                   discard_ratio=discard_ratio)
            mask = attention_rollout(prev_imgs)
            name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = TSAttentionGradRollout(model, discard_ratio=discard_ratio)
            mask = grad_rollout(prev_imgs, category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(category_index,
                                                          discard_ratio, head_fusion)

        np_img = np.array(img)[:, :, ::-1]

        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(np_img)
        _ = ax2.imshow(mask)
        plt.show()

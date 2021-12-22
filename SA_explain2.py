import argparse
import json
import os
import sys
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import cv2

from SA_grad_rollout import SelfAttentionGradRollout
from model import CANNet2s
from variables import MODEL_NAME, WIDTH, HEIGHT, MEAN, STD
from SA_rollout import SelfAttentionRollout


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
    model = CANNet2s()

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

    for i in range(0, len(img_paths), 150):
        img_path = img_paths[i]
        print(str(i) + "/" + str(len(img_paths)))
        img_folder = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        index = int(img_name.split('.')[0])

        prev_index = int(max(1, index - 5))

        img = Image.open(img_path).convert('RGB')

        prev_img_path = os.path.join(img_folder, '%03d.jpg' % prev_index)

        prev_img = Image.open(prev_img_path).convert('RGB')

        input_prev_img = transform(img).unsqueeze(0)
        input_img = transform(img).unsqueeze(0)

        input_prev_img = input_prev_img.cuda()
        input_img = input_img.cuda()

        if category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = SelfAttentionRollout(model, head_fusion=head_fusion,
                                                     discard_ratio=discard_ratio)
            mask = attention_rollout(input_prev_img, input_img)
            name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = SelfAttentionGradRollout(model, discard_ratio=discard_ratio)
            mask = grad_rollout(input_prev_img, input_img, category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(category_index,
                                                          discard_ratio, head_fusion)

        np_img = np.array(img)[:, :, ::-1]

        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        """cv2.imshow("Input Image", np_img)
        cv2.imshow(name, mask)
        cv2.imwrite("input.png", np_img)
        cv2.imwrite(name, mask)
        cv2.waitKey(-1)"""

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(np_img)
        _ = ax2.imshow(mask)
        plt.show()

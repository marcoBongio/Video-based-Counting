import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from SA_grad_rollout import SelfAttentionGradRollout
from SA_rollout import SelfAttentionRollout
from model import CANNet2s
from variables import MODEL_NAME, WIDTH, HEIGHT, MEAN, STD


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--img_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    args = get_args()
    model = CANNet2s()

    model = model.cuda()

    # modify the path of saved checkpoint if necessary
    checkpoint = torch.load("models/model_best_" + MODEL_NAME + '.pth.tar', map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    img = Image.open(args.img_path).convert('RGB')
    img_folder = os.path.dirname(args.img_path)
    img_name = os.path.basename(args.img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(max(1, index - 5))

    prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))

    prev_img = Image.open(prev_img_path).convert('RGB')

    input_prev_img = transform(img).unsqueeze(0)
    input_img = transform(img).unsqueeze(0)

    input_prev_img = input_prev_img.cuda()
    input_img = input_img.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = SelfAttentionRollout(model, head_fusion=args.head_fusion,
                                                 discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_prev_img, input_img)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = SelfAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_prev_img, input_img, args.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                                                      args.discard_ratio, args.head_fusion)

    np_img = np.array(img)[:, :, ::-1]

    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(np_img)
    _ = ax2.imshow(mask)
    plt.show()

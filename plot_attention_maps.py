import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import SACANNet2s
from variables import WIDTH, HEIGHT, MEAN, STD, MODEL_NAME, PATCH_SIZE_PF


def get_attention_map(prev_img, img, get_mask=False):
    x = transform(img).cuda()
    x_prev = transform(prev_img).cuda()
    x.size()

    _, att_mat = model(x_prev.unsqueeze(0), x.unsqueeze(0))
    # _, att_mat = model(x.unsqueeze(0))
    print(att_mat)

    # att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    # att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).cuda()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_height = HEIGHT // PATCH_SIZE_PF
    grid_width = WIDTH // PATCH_SIZE_PF
    mask = v[0, :].reshape(grid_height, grid_width).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.show()


transform = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD,
    ),
])

model = SACANNet2s()
model = model.cuda()

checkpoint = torch.load('models/model_best_' + MODEL_NAME + '.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

prev_img = Image.open("test_data/100/001.jpg").convert('RGB')
img = Image.open("test_data/100/006.jpg").convert('RGB')

result = get_attention_map(prev_img, img)

print(result)

plot_attention_map(img, result)

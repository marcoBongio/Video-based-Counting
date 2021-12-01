import json

import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from torchinfo import summary

from CANNet2s import CANNet2s
from image import *
from variables import HEIGHT, WIDTH, MEAN, STD

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

with open("train_all.json", 'r+') as outfile:
    img_paths = json.load(outfile)

with open("test.json", 'r+') as outfile:
    img_paths.extend(json.load(outfile))

extractor = CANNet2s()

extractor = extractor.cuda()

summary(extractor, input_size=(1, 3, WIDTH, HEIGHT))

extractor.eval()

with torch.no_grad():
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        #print(img_path)
        if i % 150 == 0:
            print(str(i) + "/" + str(len(img_paths)))

        img = Image.open(img_path).convert('RGB')
        img = img.resize((WIDTH, HEIGHT))

        # fig, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(img)

        img = transform(img).cuda()
        img = img.cuda()
        img = img.unsqueeze(0)

        features = extractor(img)
        torch.save(features, img_path.replace('.jpg', '_features.pt'))

        #img_plot = rearrange(img, 'c h w -> h w c')
        #axarr[1].imshow(img_plot.cpu().detach().numpy().astype('uint8'))
        #plt.show()

        """square = 4
        ix = 1
        print(features.shape)
        features_plot = rearrange(features, 'b c h w -> b h w c')
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(features_plot[0, :, :,  ix - 1].cpu(), cmap='gray')
                ix += 1
        plt.show()"""

import json

import torch
from torchinfo import summary
from torchvision import transforms

from feature_extractor import FeatureExtractor
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

extractor = FeatureExtractor()

extractor = extractor.cuda()

summary(extractor, input_size=(1, 3, WIDTH, HEIGHT))

extractor.eval()

with torch.no_grad():
    for i in range(len(img_paths)):
        img_path = img_paths[i]

        if i % 150 == 0:
            print(str(i) + "/" + str(len(img_paths)))

        img = Image.open(img_path).convert('RGB')
        img = img.resize((WIDTH, HEIGHT))

        img = transform(img).cuda()
        img = img.cuda()
        img = img.unsqueeze(0)

        features = extractor(img)
        torch.save(features, img_path.replace('.jpg', '_features.pt'))
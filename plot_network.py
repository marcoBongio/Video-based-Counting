import json

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchviz import make_dot

import dataset
from image import *
from model import CANNet2s
from train import parser
from variables import HEIGHT, WIDTH, MODEL_NAME

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.4846, 0.4558, 0.4324],
                                                std=[0.2181, 0.2136, 0.2074]),
])

with open("train_all.json", 'r+') as outfile:
    train_list = json.load(outfile)

args = parser.parse_args()
args.lr = 1e-6
args.batch_size = 2
args.momentum = 0.95
args.decay = 1e-4
args.start_epoch = 0
args.epochs = 200
args.workers = 4
args.print_freq = 100

model = CANNet2s()

model = model.cuda()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load('models/model_best_' + MODEL_NAME + '.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'])

train_loader = torch.utils.data.DataLoader(
    dataset.listDataset(train_list,
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(), transforms.Normalize(mean=[0.4846, 0.4558, 0.4324],
                                                                        std=[0.2181, 0.2136, 0.2074]),
                        ]),
                        train=True,
                        num_workers=args.workers),
    batch_size=args.batch_size)

batch = next(iter(train_loader))

img_path = train_list[0]
img_folder = os.path.dirname(img_path)
img_name = os.path.basename(img_path)
index = int(img_name.split('.')[0])

prev_index = int(max(1, index - 5))

prev_img_path = os.path.join(img_folder, '%03d.jpg' % (prev_index))

prev_img = Image.open(prev_img_path).convert('RGB')
img = Image.open(img_path).convert('RGB')

prev_img = prev_img.resize((WIDTH, HEIGHT))
img = img.resize((WIDTH, HEIGHT))

prev_img = transform(prev_img).cuda()
img = transform(img).cuda()

gt_path = img_path.replace('.jpg', '_resize.h5')
gt_file = h5py.File(gt_path)
target = np.asarray(gt_file['density'])

prev_img = prev_img.cuda()
prev_img = Variable(prev_img)

img = img.cuda()
img = Variable(img)

img = img.unsqueeze(0)
prev_img = prev_img.unsqueeze(0)

make_dot(model(prev_img, img), params=dict(list(model.named_parameters()))).render("PeopleFlowTimeSformer", format="png")
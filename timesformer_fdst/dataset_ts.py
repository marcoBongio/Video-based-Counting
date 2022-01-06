import random

from torch.utils.data import Dataset

from image_ts import *


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        if shuffle:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        prev_imgs, img, post_imgs, prev_target, target, post_target = load_data(img_path)

        if self.transform is not None:
            for i in range(len(prev_imgs)):
                prev_imgs[i] = self.transform(prev_imgs[i])
            img = self.transform(img)
            for i in range(len(post_imgs)):
                post_imgs[i] = self.transform(post_imgs[i])
        return prev_imgs, img, post_imgs, prev_target, target, post_target

import glob

import scipy.io
import skimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage import transform

from image import *
from variables import HEIGHT, WIDTH

# set the root to the path of Venice dataset you download
root = '../../venice/venice/'

# now generate the Venice's ground truth
train_folder = os.path.join(root, 'train_data/images')
test_folder = os.path.join(root, 'test_data/images')
path_sets = [train_folder, test_folder]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    gt = scipy.io.loadmat(
        img_path.replace('.jpg', '.mat').replace('images', 'ground-truth'))
    img = plt.imread(img_path)

    anno_list = gt['annotation']
    roi = skimage.transform.resize(gt['roi'], (HEIGHT, WIDTH), order=0)
    matrix = gt['homograph']

    k = np.zeros((HEIGHT, WIDTH))
    rate_h = img.shape[0] / HEIGHT
    rate_w = img.shape[1] / WIDTH
    for i in range(0, len(anno_list)):
        y_anno = min(int(anno_list[i][1] / rate_h), HEIGHT - 1)
        x_anno = min(int(anno_list[i][0] / rate_w), WIDTH - 1)
        k[y_anno, x_anno] = 1
    k = gaussian_filter(k, 3)

    k *= roi

    with h5py.File(img_path.replace('.jpg', '_resize.h5'), 'w') as hf:
        hf['density'] = k
        hf.close()

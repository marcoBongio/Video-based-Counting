import glob

import scipy.io
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from image import *
from variables import HEIGHT, WIDTH

# set the root to the path of Venice dataset you download
root = '../../mall_dataset/'

# now generate the Venice's ground truth
test_folder = os.path.join(root, 'frames/')
path_sets = [test_folder]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

gts = scipy.io.loadmat(os.path.join(root, 'mall_gt.mat'))
gts_frame = gts['frame'][0]
gts_count = gts['count']

for img_path in img_paths:
    print(img_path)
    img_name = os.path.basename(img_path)
    # print(img_path)
    index = img_name.split('.')[0]
    index = int(index.split('_')[1]) - 1
    img = plt.imread(img_path)

    anno_list = gts_frame[index]
    anno_list = anno_list.item(0)[0]
    k = np.zeros((HEIGHT, WIDTH))
    rate_h = img.shape[0] / HEIGHT
    rate_w = img.shape[1] / WIDTH
    for i in range(0, len(anno_list)):
        y_anno = min(int(anno_list[i][1] / rate_h), HEIGHT - 1)
        x_anno = min(int(anno_list[i][0] / rate_w), WIDTH - 1)
        k[y_anno, x_anno] = 1
    k = gaussian_filter(k, 3)

    with h5py.File(img_path.replace('.jpg', '_resize.h5'), 'w') as hf:
        hf['density'] = k
        hf.close()

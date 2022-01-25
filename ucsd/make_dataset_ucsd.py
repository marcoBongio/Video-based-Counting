import glob

import scipy.io
import skimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage import transform

from image import *
from variables import HEIGHT, WIDTH

# set the root to the path of Venice dataset you download
root = '../../ucsdpeds/'

# now generate the Venice's ground truth
test_folder = os.path.join(root, 'vidf/')
path_sets = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]

roi_path = os.path.join(root, 'vidf-cvpr/vidf1_33_roi_mainwalkway.mat')
roi = scipy.io.loadmat(roi_path)['roi'][0]
roi = roi.item(0)[-1]
roi = skimage.transform.resize(roi, (HEIGHT, WIDTH), order=0)

for path in path_sets:
    gt_path = path.replace("/vidf/", "/vidf-cvpr/")
    gt_path = gt_path.replace(".y", "_frame_full.mat")
    gts = scipy.io.loadmat(gt_path)
    gts_frame = gts['frame'][0]

    img_paths = []

    for img_path in glob.glob(os.path.join(path, '*.png')):
        img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        img_name = os.path.basename(img_path)
        # print(img_path)
        index = img_name.split('.')[0]
        index = int(index.split('f')[-1]) - 1
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

        # plt.imshow(k)
        # plt.show()

        k *= roi

        # plt.imshow(k)
        # plt.show()

        with h5py.File(img_path.replace('.png', '_resize.h5'), 'w') as hf:
            hf['density'] = k
            hf.close()

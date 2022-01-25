import json
import os
from os.path import join
import random

if __name__ == '__main__':
    # root is the path to your code, which is current directory
    root = ''
    # root_data is where you download the FDST dataset
    root_data = '../../mall_dataset/'
    test_folders = join(root_data, 'frames/')
    output_test = join(root, 'test.json')

    test_img_list = []

    for root, dirs, files in os.walk(test_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                test_img_list.append(join(root, file_name))

    with open(output_test, 'w') as f:
        json.dump(test_img_list, f)

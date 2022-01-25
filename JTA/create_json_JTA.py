import json
import os
from os.path import join
import random

if __name__ == '__main__':
    # root is the path to your code, which is current directory
    root = ''
    # root_data is where you download the FDST dataset
    root_data = '../../JTA-Dataset/frames/'
    train_folders = join(root_data, 'train/')
    test_folders = join(root_data, 'test/')
    val_folders = join(root_data, 'val/')
    output_train = join(root, 'train.json')
    output_val = join(root, 'val.json')
    output_test = join(root, 'test.json')

    train_img_list = []
    test_img_list = []
    val_img_list = []

    for root, dirs, files in os.walk(train_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                train_img_list.append(join(root, file_name))

    for root, dirs, files in os.walk(test_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                test_img_list.append(join(root, file_name))

    for root, dirs, files in os.walk(val_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                val_img_list.append(join(root, file_name))

    train_num = 9000
    random.shuffle(train_img_list)
    train_img_list = train_img_list[:train_num]

    val_num = 3600
    random.shuffle(val_img_list)
    val_img_list = val_img_list[:val_num]

    test_num = 6000
    random.shuffle(test_img_list)
    test_img_list = test_img_list[:test_num]

    with open(output_train, 'w') as f:
        json.dump(train_img_list, f)

    with open(output_val, 'w') as f:
        json.dump(val_img_list, f)

    with open(output_test, 'w') as f:
        json.dump(test_img_list, f)

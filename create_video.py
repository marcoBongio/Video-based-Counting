import csv
import json
import os

import PIL
import cv2
import numpy as np
import glob

from variables import HEIGHT, WIDTH, MODEL_NAME

root = ''
# root_data is where you download the FDST dataset
root_data = ''

scene = ''
image_folder = 'test_data/' + scene
dm_folder = 'plot/' + MODEL_NAME + '/' + scene

if scene == '':
    path_sets = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                 os.path.isdir(os.path.join(image_folder, f))]
else:
    path_sets = [image_folder]

test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

try:
    os.mkdir(os.path.dirname('video/'))
except:
    pass

pred = []

# load results of the model
filename = open("results/model_best_" + MODEL_NAME + ".csv", 'r')
file = csv.DictReader(filename)

for col in file:
    pred.append(float(col["Prediction"]))

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1660, 30)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 1
lineType = 1

j = 0
for path in path_sets:
    if scene == '':
        video_name = 'video/' + path.split('/')[-1] + '.mp4'
    else:
        video_name = 'video/' + scene + '.mp4'
    print(video_name)
    images = [img for img in os.listdir(path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, fps=10, frameSize=(width, height))

    for image in images:
        img = cv2.imread(os.path.join(path, image))
        if scene == '':
            dm = cv2.imread(os.path.join(dm_folder, path.split('/')[-1], image.replace('.jpg', '_pred.jpg')))
        else:
            dm = cv2.imread(os.path.join(dm_folder, image.replace('.jpg', '_pred.jpg')))
        dm = cv2.resize(dm, (width, height), interpolation=cv2.INTER_CUBIC)
        img = cv2.addWeighted(img, 0.5, dm, 0.5, 0)

        cv2.putText(img, 'Count = ' + str(round(pred[j], 2)),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        j += 1

        video.write(img)

    cv2.destroyAllWindows()
    video.release()

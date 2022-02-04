import json
import math

import numpy as np
import csv

import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt, pyplot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.distributions.empirical_distribution import ECDF

from variables import MODEL_NAME

errs = []
gt = []
pred = []

# load results of the model
filename = open("results/model_best_" + MODEL_NAME + ".csv", 'r')
file = csv.DictReader(filename)

for col in file:
    errs.append(float(col["Error"]))
    gt.append(float(col["GT"]))
    pred.append(float(col["Prediction"]))

# compute MAE and RMSE of the model
mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))

print('MAE: ', mae)
print('RMSE: ', rmse, '\n')

# compute stats on the errors
mean = np.mean(errs)
var = np.var(errs)
std = np.sqrt(var)

conf_int = st.t.interval(0.95, len(errs) - 1, loc=mean, scale=st.sem(errs))

print("Error Mean = " + str(mean))
print("Error Standard deviation = " + str(std))
print("Minimum = " + str(np.min(errs)))
print("Maximum = " + str(np.max(errs)) + "\n")

errs2 = list(range(len(pred)))
for j in range(0, len(pred)):
    errs2[j] = pred[j] - gt[j]

bins = math.ceil((np.max(errs2) - np.min(errs2)))
# plot errors histogram
plt.hist(errs2, bins=bins, edgecolor='black')
plt.xlabel('Errors (prediction - ground-truth)')
plt.ylabel('Frequency')
plt.axvline(np.mean(errs2), color='r', linestyle='dashed', linewidth=1, label='mean')
plt.axvline(np.median(errs2), color='y', linestyle='dashed', linewidth=1, label='median')
plt.legend(loc="upper left")
plt.show()

mean = np.mean(errs2)
median = np.median(errs2)

print("Error Mean = " + str(mean))
print("Error Median = " + str(median) + "\n")

# compute stats on the prediction
mean = np.mean(pred)
var = np.var(pred)
std = np.sqrt(var)

conf_int = st.t.interval(0.95, len(pred) - 1, loc=mean, scale=st.sem(pred))

print("Prediction Mean = " + str(mean))
print("Prediction Standard deviation = " + str(std))
print("Minimum = " + str(np.min(pred)))
print("Maximum = " + str(np.max(pred)) + "\n")

# plot predictions histogram
# plt.hist(pred)
# plt.show()

test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

indexes = list(range(len(errs)))
# print error|GT|prediction in order according to the error
print("Image \t | Error \t | Ground-Truth \t | Prediction")
errs, gt, pred, indexes = zip(*sorted(zip(errs, gt, pred, indexes), reverse=False))
for i in range(len(errs)):
    print(img_paths[indexes[i]], errs[i], "|", gt[i], "| ", pred[i])

bins = math.ceil((np.max(errs) - np.min(errs)))
plt.hist(errs, bins, density=True, histtype='step',
                            cumulative=True, label='ECDF')
plt.xlabel('Absolute Error (|prediction - ground-truth|)')
plt.ylabel('Probability')
plt.axvline(np.mean(errs), color='r', linestyle='dashed', linewidth=1, label='mean')
plt.legend(loc="lower right")
plt.show()

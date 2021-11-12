import numpy as np
import csv
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

model_name = "model_best_244_7200_lucidrain"

errs = []
gt = []
pred = []

# load results of the model
filename = open("results/" + model_name + ".csv", 'r')
file = csv.DictReader(filename)

for col in file:
    errs.append(float(col["Error"]))
    gt.append(float(col["GT"]))
    pred.append(float(col["Prediction"]))

# compute MAE and RMSE of the model
mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))

print('MAE: ', mae)
print('RMSE: ', rmse)

# compute stats on the errors
mean = np.mean(errs)
var = np.var(errs)
std = np.sqrt(var)

conf_int = st.t.interval(0.95, len(errs) - 1, loc=mean, scale=st.sem(errs))

print("Error Mean = " + str(mean))
print("Error Standard deviation = " + str(std))
print("Error Confidence Interval(95%) = " + str(conf_int) + "\n")

# compute stats on the prediction
mean = np.mean(pred)
var = np.var(pred)
std = np.sqrt(var)

conf_int = st.t.interval(0.95, len(pred) - 1, loc=mean, scale=st.sem(pred))

print("Prediction Mean = " + str(mean))
print("Prediction Standard deviation = " + str(std))
print("Prediction Confidence Interval(95%) = " + str(conf_int) + "\n")

# print error|GT|prediction in reverse order according to the error
print("Error \t | Ground-Truth \t | Prediction")
errs, gt, pred = zip(*sorted(zip(errs, gt, pred), reverse=True))
for i in range(len(errs)):
    print(errs[i], "|", gt[i], "| ", pred[i])

# plot errors histogram
plt.hist(errs, bins=100)
plt.show()

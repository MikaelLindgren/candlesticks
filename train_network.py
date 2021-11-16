import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(40,1)))
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.MaxPool1D())
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.Conv1D(32,kernel_size=4,strides=1, activation = "relu"))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(22))
model.summary()
model.compile(optimizer='adam', 
    loss='mean_absolute_error', 
    metrics=['mean_absolute_error'])
df = pd.read_csv("finance_data.csv")
df.pop(df.columns[0])
labels = df.pop("40")
weights = df.pop("41")
weights = weights
log_label = []
def z_maker(num):
    return np.zeros(num+1)
def one_hot(x):
    mi = min(x)
    x = x - mi
    ma = max(x)
    hotx = []
    for xx in x:
        temp = z_maker(ma)
        temp[xx] = 1
        hotx.append(temp)
    return hotx
for i in range(len(weights)):
    temp = weights[i]*labels[i]
    if temp > 0:
        temp = np.log(temp)
        temp = int(math.floor(temp))
    elif temp < 0:
        temp = - np.log(-temp)
    else:
        temp = int(temp)
    temp = int(math.ceil(temp))
    log_label.append(temp)
log_label = np.asarray(log_label)
hot_label = one_hot(log_label)
#print(hot_label)
nn = np.asarray(df)
hot_label = np.asarray(hot_label)
print(len(nn))
print(len(hot_label))
X_train = nn[:40000]
X_test = nn[40000:]
y_train = hot_label[:40000]
y_test = hot_label[40000:]

def quickshape(x):
    print(np.shape(x))
quickshape(X_train)
quickshape(X_test)
quickshape(y_train)
quickshape(y_test)
model.fit(x = X_train, y = y_train, epochs = 30)
pred_y = model.predict(x = X_test)
pred_y = pd.DataFrame(pred_y)
test_y = pd.DataFrame(y_test)
pred_y.to_csv("predy.csv")
test_y.to_csv("testy.csv")
quickshape(pred_y)
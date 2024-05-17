import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import  Conv2D , MaxPooling2D , Dense , Flatten
#
X_train = np.loadtxt(r"C:\Users\nimis\OneDrive\Desktop\ML Projects\CatsDogs\input.csv" , delimiter= ",")
Y_train = np.loadtxt(r"C:\Users\nimis\OneDrive\Desktop\ML Projects\CatsDogs\labels.csv" , delimiter=",")
X_test  =  np.loadtxt(r"C:\Users\nimis\OneDrive\Desktop\ML Projects\CatsDogs\input_test.csv" , delimiter=",")
Y_test = np.loadtxt(r"C:\Users\nimis\OneDrive\Desktop\ML Projects\CatsDogs\labels_test.csv" ,delimiter=", ")

print("shape of X_train", X_train.shape)
print("shape of X_test", X_test.shape)
print("shape of Y_train", Y_train.shape)
print("shape of Y_train", Y_test.shape)

X_train = X_train.reshape(len(X_train) , 100 , 100,3)
X_test = X_test.reshape(len(X_test), 100,100,3)
Y_train = Y_train.reshape(len(Y_train), 1)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train/255.0
X_test =X_test/255.0

model = Sequential([
    Conv2D(32, (3,3) , activation="relu" , input_shape=(100,100,3) ),
    MaxPooling2D((2,2)),
    Conv2D(32 , (3,3) , activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64 , activation="relu"),
    Dense(1, activation="sigmoid")

])

model.compile(loss="binary_crossentropy" ,  optimizer= "adam",  metrics= ['accuracy'])
#till now keras using the tensorlfow have created the conputational graph for the model
model.fit(X_train ,Y_train , epochs= 10 , batch_size=64)


model.evaluate(X_test , Y_test )
#Problems :
# I was reloading the data again on each run how to fix it ?
# While loading the data set i have used csv file , How to directly load images ?
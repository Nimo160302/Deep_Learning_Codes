import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
data =  keras.datasets.fashion_mnist
# keras method for train test data
(train_images , train_labels) , (test_images , test_labels) = data.load_data()
#from the data set on tenserfloe we got class_names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_labels[0]) #print a label
print(train_images[0].shape) #print the 28x28 values on image
plt.imshow(train_images[7]) #imshow and show together is used to show the image
# plt.show()
print(type(train_images[0]))
#since the train image is stored in numpy array from range of 0 to 255 we can directly divide
train_images = train_images/255.0
test_images = test_images /255.0
# now since we need to create a neural network and we have data of 2D array we need to convert into
# one d array since the basic neural layer takes one dimension data (FLATTENING)
# so 28 * 28 = 784  input layer , for output layer we have labels 0 to 9 i.e 10 output neurons
# we can introduce a hidden layer  what would be the size : any generally we take 15-20 % of prev layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"] )

model.fit(train_images, train_labels , epochs=5)
#test_loss , test_acc =  model.evaluate(test_images, test_labels)
#print("tested acc: " ,test_acc, "test_loss :" , test_loss )

prediction= model.predict([test_images])
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("actual : " + class_names[test_labels[i]])
    plt.title("prediction" + class_names[np.argmax(prediction[i])] )
    plt.show()
print(class_names[np.argmax(prediction[0])])
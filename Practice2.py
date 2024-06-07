import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

train_images.shape
train_images[0,23,23]

train_labels[:10]

class_names=["T-shirt/top","Trouser","Pillover",'Dress','Coat']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

#Data preprocessing
train_images=train_images/255.0
test_images=test_images/255.0

#model
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128,activation='relu'),keras.layers.Dense(10,activation='softmax')]) #output layer 10 = number of class names

#compiling
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#fitting
model.fit(train_images,train_labels,epochs=10)

test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=1)
print(test_acc)

predictions=model.predict(test_images)
test_images.shape
print(predictions) #address array
print(predictions[0]) #probability dist.
#use np.argmax to get index of class
print(class_names[mp.argmax(predictions[0])])
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#verifying predictions

#code
import math
from os import listdir

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
#from scipy import misc
import imageio

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

print(tf.__version__)


mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

#train_images = np.array([train_images])

'''
for x in range(len(train_images)):
    for y in range(len(train_images[x])):
        for i in range(len(train_images[x][y])):
            train_images[x][y][i] = [train_images[x][y][i]]
            #train_images[x, y, i] = [train_images[x, y, i]]
'''


#print(train_images)
train_images = train_images / 255.0
test_images = test_images / 255.0
#print(images[0].shape)

#sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=1.0, nesterov=True)

'''
model = Sequential()
#add model layers
model.add(Conv2D(60, kernel_size=7, activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(Conv2D(60, kernel_size=7, activation=tf.nn.relu))
model.add(Conv2D(60, kernel_size=7, activation=tf.nn.relu))
model.add(Flatten())
model.add(Dense(10, activation=tf.nn.sigmoid))
'''

model = tf.keras.models.load_model('model1.h5')

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#for i in range(len(labels)):
#model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

#print(test_images[0])

#model.save("model1.h5")

train_images = train_images.reshape(len(train_images), 28, 28)
test_images = test_images.reshape(len(test_images), 28, 28)

fail_images = []
fail_labels = []
fail_predictions = []

for i in range(len(test_labels)):
    if(test_labels[i] != np.argmax(predictions[i])):
        fail_images.append(test_images[i])
        fail_labels.append(test_labels[i])
        fail_predictions.append(predictions[i])


plt.figure(figsize=(10,10))
for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(fail_images[i], cmap=plt.cm.binary)
    plt.xlabel(str(fail_labels[i]) + " " + str(np.argmax(fail_predictions[i])))
plt.show()

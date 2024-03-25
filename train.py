import math
from os import listdir

import pygame
import pygame.camera
from pygame.locals import *

from PIL import Image, ImageDraw, ImageFont

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt
#from scipy import misc
import imageio

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

print(tf.__version__)

train_images = np.load("X.npy")
train_labels = np.load("Y.npy")

test_images = np.load("X_test.npy")
test_labels = np.load("Y_test.npy")

train_labels = np.array(train_labels)

print(train_labels)

train_images = np.reshape(train_images, (len(train_images), 128, 128, 1))
test_images = np.reshape(test_images, (len(test_images), 128, 128, 1))

#train_images = np.array([train_images])

'''

plt.figure(figsize=(10,10))
for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(str(train_labels[i]))
plt.show()

'''

#print(train_images)
#train_images = train_images / 255.0
#test_images = test_images / 255.0
#print(images[0].shape)

model = Sequential()

model.add(Conv2D(60, kernel_size=12, activation=tf.nn.relu, input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(30, kernel_size=6, activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, kernel_size=3, activation=tf.nn.relu))
model.add(Flatten())
model.add(Dense(6, activation=tf.nn.sigmoid))

#model = tf.keras.models.load_model('model1.h5')

sgd = tf.keras.optimizers.Adam(lr=0.000001)

model.compile(optimizer=sgd, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#for i in range(len(labels)):
while True:
    model.fit(train_images, train_labels, epochs=10)
    model.save("model1.h5")

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

#print(test_images[0])


train_images = train_images.reshape(len(train_images), 128, 128)
test_images = test_images.reshape(len(test_images), 128, 128)
#test_images = test_images.reshape(len(test_images), 28, 28)



pygame.init()
pygame.camera.init()

win = pygame.display.set_mode((1000, 1000))

pygame.display.set_caption("stuf")

imageNum = 0

myfont = pygame.font.SysFont('Comic Sans MS', 50)

running = True

size = (640,480)
#size = (128,128)
# create a display surface. standard pygame stuff

# this is the same as what we saw before
clist = pygame.camera.list_cameras()
if not clist:
    raise ValueError("Sorry, no cameras detected.")
cam = pygame.camera.Camera(clist[0], size)
cam.start()

# create a surface to capture to.  for performance purposes
# bit depth is the same as that of the display surface.
snapshot = pygame.surface.Surface(size, 0, win)

while running:
    pygame.time.delay(100)

    win.fill((0,255,255))

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                running = False;
            if event.key == pygame.K_LEFT:
                imageNum -= 1
            if event.key == pygame.K_RIGHT:
                imageNum += 1
                

    #keys = pygame.keys.get_pressed()


    # if you don't want to tie the framerate to the camera, you can check
    # if the camera has an image ready.  note that while this works
    # on most cameras, some will never return true.
    if cam.query_image():
        snapshot = cam.get_image(snapshot)

    snapshot2 = snapshot.subsurface((80, 0, 480, 480))
    snapshot2 = pygame.transform.scale(snapshot2, (128, 128))

    # blit it to the display surface.  simple!
    win.blit(snapshot2, (100,100))
    pygame.display.flip()


    currentImage = []

    for y in range(128):
        currentLayer = []
        for x in range(128):
            color = snapshot2.get_at((x, y))[:3]
            currentLayer.append((color[0] + color[1] + color[2])/3)
        currentImage.append(currentLayer)


    
    newImg = Image.new('L', (128, 128), color = 100)
    pixels = newImg.load()

    for x in range(128):
        for y in range(128):
            pixels[x, y] = int(currentImage[y][x])
    
    newImg.save('image.png')

    img = pygame.image.load("image.png")

    img = pygame.transform.scale(img, (256, 256))

    win.blit(img, (150, 410))


    currentImage = np.array([currentImage])

    currentImage = np.reshape(currentImage, (1, 128, 128, 1))


    predictions = model.predict(currentImage)

    print(predictions)

    numText = myfont.render("real: " + str(np.argmax(predictions[imageNum])) + "  computer: " + str(test_labels[imageNum]), False, (0, 0, 0))

    win.blit(numText, (600, 500))



    pygame.display.update()
    

pygame.quit()


'''

    newImg = Image.new('L', (128, 128), color = 100)
    pixels = newImg.load()

    for x in range(128):
        for y in range(128):
            pixels[x, y] = int(test_images[imageNum][y][x])
    
    newImg.save('image.png')

    img = pygame.image.load("image.png")

    img = pygame.transform.scale(img, (256, 256))

    win.blit(img, (150, 410))

'''


'''

plt.figure(figsize=(10,10))
for i in range(0, 64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(str(test_labels[i]) + " " + str(np.argmax(predictions[i])))
plt.show()

'''

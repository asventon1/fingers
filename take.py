import math
from os import listdir
import calendar
import time

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

saveImageNumber = 0
shouldTake = 0

while running:
    pygame.time.delay(100)

    win.fill((0,255,255))


    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                running = False;
            if event.key == pygame.K_t:
                shouldTake = 1000
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
#    pygame.display.flip()


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

    if(shouldTake > 0):
        newImg.save("images/train2/" + str(calendar.timegm(time.gmtime())) + "_" + str(saveImageNumber) + "_" + str(imageNum) + ".png")
        saveImageNumber += 1
        shouldTake -= 1
        numText = myfont.render("number left: " + str(shouldTake), False, (0, 0, 0))
        win.blit(numText, (300, 200))
        


    img = pygame.image.load("image.png")

    img = pygame.transform.scale(img, (256, 256))

    win.blit(img, (150, 410))
#    pygame.display.flip()


    currentImage = np.array([currentImage])

    currentImage = np.reshape(currentImage, (1, 128, 128, 1))

    
    numText = myfont.render(str(imageNum), False, (0, 0, 0))

    win.blit(numText, (600, 500))

    '''
    predictions = model.predict(currentImage)

    print(predictions)

    numText = myfont.render("computer: " + str(np.argmax(predictions[0])), False, (0, 0, 0))

    win.blit(numText, (600, 500))
   '''


    pygame.display.flip()

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

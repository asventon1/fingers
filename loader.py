from os import listdir
import imageio
import numpy as np

def test():
    images = listdir("images/test")

    #images = np.array([])
    imagesList = []

    #images = np.append(images, 10)

    #print(images)

    for i in images:
            arr = imageio.imread("images/test/"+i)
            imagesList.append(arr.tolist())

    imagesList = np.array(imagesList)

    np.save("X_test", imagesList)

    labelsList = []

    for i in images:
        labelsList.append(i[len(i)-5])

    np.save("Y_test", labelsList)

test()


def train():
    images = listdir("images/train")

    #images = np.array([])
    imagesList = []

    #images = np.append(images, 10)

    #print(images)

    for i in images:
            arr = imageio.imread("images/train/"+i)
            imagesList.append(arr.tolist())

    imagesList = np.array(imagesList)

    np.save("X", imagesList)

    labelsList = []

    for i in images:
        labelsList.append(i[len(i)-5])


    np.save("Y", labelsList)

train()

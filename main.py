import numpy as np

x = np.load("X.npy").tolist()
y = np.load("Y.npy").tolist()

Xfile = open("X.txt", "w")
Xfile.write(str(x))


Yfile = open("Y.txt", "w")
Yfile.write(str(y))

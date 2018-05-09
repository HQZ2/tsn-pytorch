import h5py
import numpy as np
import matplotlib.pyplot as plt

a = h5py.File('/Users/joy.he/Desktop/train_feature.h5','r')


list(a.keys())

f1 = a['sJFgo9H6zNo/2048'][...]
f2 = a['sJFgo9H6zNo/201'][...]

for i in range(201):
    plt.clf()
    plt.plot(f2[:,i])
    plt.ylim(0,1)
    plt.savefig('/tmp/picture/{:3d}.jpg'.format(i))
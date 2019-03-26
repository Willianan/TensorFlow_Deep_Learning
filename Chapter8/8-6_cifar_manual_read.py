# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/21 9:59
文   件：8-6_cifar_manual_read.py
E-mail:willianan@hotmail.com
"""

import numpy as np
from scipy.misc import imsave

filename = '/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin'

bytestream = open(filename,'rb')
buf = bytestream.read(10000*(1+32*32*3))
bytestream.close()

data = np.frombuffer(buf,dtype = np.int8)
data = data.reshape(10000,1+32*32*3)
labels_images = np.hsplit(data,[1])
labels = labels_images[0].reshape(10000)
images = labels_images[1].reshape(10000,32,32,3)

img = np.reshape(images[0],(3,32,32))
img = img.transpose(1,2,0)

import pylab
print(labels[0])
pylab.imshow(img)
pylab.show()
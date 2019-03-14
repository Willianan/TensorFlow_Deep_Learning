from tensorflow.examples.tutorials.mnist import input_data
#下载MNIST数据集
mniist = input_data.read_data_sets("MNIST_Data/",one_hot=True)

print('输入数据：',mniist.train.images)
print('输入数据打shape：',mniist.train.images.shape)

import pylab
im = mniist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
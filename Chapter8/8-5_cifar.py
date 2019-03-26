# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/21 9:42
文   件：8-5_cifar.py
E-mail:willianan@hotmail.com
"""

import cifar10_input
import tensorflow as tf
import pylab

#取数据
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
images_test,labels_test = cifar10_input.inputs(eval_data = True,data_dir = data_dir,batch_size = batch_size)

sess = tf.Session()
tf.global_variables_initializer().run(session = sess)
tf.train.start_queue_runners(sess = sess)
image_batch,label_batch = sess.run([images_test,labels_test])
print("___\n",image_batch[0])
print("___\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()